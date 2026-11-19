"""Evaluate a frozen model on held-out sequences with explicit initialization."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
from io_drift.data import load_sequence
from io_drift.inference import Predictor
from io_drift.provenance import content_hash
from io_drift.metrics import velocity_metrics, trajectory_metrics
from io_drift.navigation import integrate_imu, integrate_velocity
from train import fit_baseline,summarize


def evaluate(prepared,data,output):
    run=json.loads((output/'run.json').read_text());config=run['config']
    for filename,digest in run['prepared_hashes'].items():
        if content_hash(prepared/filename)!=digest:
            raise ValueError(f'Prepared data changed since training: {filename}')
    test=np.load(prepared/'test.npz');train=np.load(prepared/'train.npz')
    if set(test['sequence'])!=set(config['test']):
        raise ValueError('Test sequences differ from frozen config')
    predictor=Predictor(output/'velocity_cnn.pt')
    scale,ridge=fit_baseline(train)
    predictions={'Zero':np.zeros_like(test['y']),
                 'Training mean':np.broadcast_to(train['y'].mean(0),test['y'].shape),
                 'Ridge':ridge.predict(scale.transform(summarize(test['x']))),
                 'CNN':predictor.predict(test['x'])}
    velocity_rows=[];trajectory_rows=[];all_predictions=[]
    for name in config['test']:
        mask=test['sequence']==name;endpoints=test['endpoint'][mask]
        seq=load_sequence(data,name,config['max_gap_s'])
        for method,pred in predictions.items():
            velocity_rows.append({'sequence':name,'method':method,
                'selected_on_validation':method==run['selected_on_validation'],
                **velocity_metrics(test['y'][mask],pred[mask])})
        # Segment by raw IMU gaps; no correction/reset inside a continuous segment.
        segments=np.cumsum(np.r_[0,np.diff(seq.time)>config['max_gap_s']])
        for segment in np.unique(segments[endpoints]):
            keep=np.flatnonzero(segments[endpoints]==segment)
            if len(keep)<2:
                continue
            index=endpoints[keep];a,b=index[0],index[-1]
            relative=index-a;times=seq.time[index]
            q,v,raw_position=integrate_imu(seq.time[a:b+1],seq.imu[a:b+1],
                seq.quaternion[a],seq.velocity[a],seq.position[a])
            rotations=Rotation.from_quat(q[relative])
            methods={method:integrate_velocity(times,rotations.apply(pred[mask][keep]),seq.position[a])
                     for method,pred in predictions.items()}
            methods['Raw IMU integration']=raw_position[relative]
            # Constant reference initial velocity is a baseline, never an ongoing input.
            methods['Constant initial velocity']=seq.position[a]+(times-times[0])[:,None]*seq.velocity[a]
            for method,position in methods.items():
                trajectory_rows.append({'sequence':name,'segment':int(segment),'method':method,
                    'duration_s':float(times[-1]-times[0]),'samples':len(times),
                    **trajectory_metrics(times,seq.position[index],position)})
                if method=='CNN':
                    for t,ref,est in zip(times,seq.position[index],position):
                        all_predictions.append([name,int(segment),t,*ref,*est])
    velocity=pd.DataFrame(velocity_rows);trajectory=pd.DataFrame(trajectory_rows)
    velocity.to_csv(output/'test_velocity_metrics.csv',index=False)
    velocity.groupby('method')[['rmse_mean','mae_mean']].mean().to_csv(output/'test_velocity_macro.csv')
    trajectory.to_csv(output/'test_trajectory_metrics.csv',index=False)
    pd.DataFrame(all_predictions,columns=['sequence','segment','time_s','ref_x','ref_y','ref_z','cnn_x','cnn_y','cnn_z']).to_csv(output/'test_trajectories.csv',index=False)
    print(velocity.groupby('method')[['rmse_mean','mae_mean']].mean().to_string())
    print(trajectory[['sequence','method','position_rmse_m','final_error_m']].to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',type=Path,default=ROOT/'data/prepared')
    p.add_argument('--data',type=Path,default=ROOT/'data/euroc')
    p.add_argument('--output',type=Path,default=ROOT/'results/first_run')
    a=p.parse_args();evaluate(a.prepared,a.data,a.output)
