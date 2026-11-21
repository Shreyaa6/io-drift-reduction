"""Evaluate frozen selection once; reused test sequences are development comparisons."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from io_drift.data import load_sequence
from io_drift.inference import Predictor
from io_drift.optimization import common_windows
from io_drift.calibration import fit_training_bias,apply_bias
from io_drift.metrics import velocity_metrics,trajectory_metrics
from io_drift.navigation import integrate_imu,integrate_velocity
from train import fit_baseline,summarize


def evaluate(output):
    selection=json.loads((output/'selection.json').read_text())
    if hashlib.sha256((output/'best.pt').read_bytes()).hexdigest()!=selection['checkpoint_sha256']:
        raise ValueError('Selected checkpoint changed after validation freeze')
    if json.loads((ROOT/'data/euroc/provenance.json').read_text())['sha256']!=selection['dataset_sha256']:
        raise ValueError('Dataset archive differs from training source')
    chosen=Predictor(output/'best.pt');original=Predictor(ROOT/'results/first_run/velocity_cnn.pt')
    seq_train=[load_sequence(ROOT/'data/euroc',n) for n in selection['train']]
    calibration=fit_training_bias(seq_train)
    (output/'training_calibration.json').write_text(json.dumps(calibration,indent=2))
    common={'common_window':selection['common_window'],'stride':selection['stride']}
    training=[common_windows(s,200,2,**common) for s in seq_train]
    train={'x':np.concatenate([a[0] for a in training]),'y':np.concatenate([a[1] for a in training])}
    scale,ridge=fit_baseline(train)
    velocity_rows=[];trajectory_rows=[];predictions=[]
    for name in selection['test']:
        seq=load_sequence(ROOT/'data/euroc',name)
        x,y,end=common_windows(seq,chosen.config['window'],chosen.config['sample_step'],**common)
        original_x,_,original_end=common_windows(seq,200,2,**common)
        if not np.array_equal(end,original_end):
            raise ValueError('Endpoint mismatch')
        methods={'Optimized DL':chosen.predict(x),'Original CNN':original.predict(original_x),
                 'Ridge':ridge.predict(scale.transform(summarize(original_x))),
                 'Training mean':np.broadcast_to(train['y'].mean(0),y.shape),'Zero':np.zeros_like(y)}
        for method,pred in methods.items():
            velocity_rows.append({'sequence':name,'method':method,**velocity_metrics(y,pred)})
        segments=np.cumsum(np.r_[0,np.diff(seq.time)>.025])
        for segment in np.unique(segments[end]):
            keep=np.flatnonzero(segments[end]==segment)
            if len(keep)<2:
                continue
            index=end[keep];a,b=index[0],index[-1];relative=index-a;t=seq.time[index]
            raw=seq.imu[a:b+1];corrected=apply_bias(raw,calibration)
            rotations={}
            for mode,sensors in [('raw',raw),('training-calibrated',corrected)]:
                q,_,p=integrate_imu(seq.time[a:b+1],sensors,seq.quaternion[a],seq.velocity[a],seq.position[a])
                rotations[mode]=Rotation.from_quat(q[relative])
                trajectory_rows.append({'sequence':name,'segment':int(segment),
                    'method':f'IMU integration ({mode})',**trajectory_metrics(t,seq.position[index],p[relative])})
                for method in ('Optimized DL','Original CNN','Zero'):
                    position=integrate_velocity(t,rotations[mode].apply(methods[method][keep]),seq.position[a])
                    trajectory_rows.append({'sequence':name,'segment':int(segment),
                        'method':f'{method} ({mode})',**trajectory_metrics(t,seq.position[index],position)})
                    if method=='Optimized DL':
                        for ts,ref,est in zip(t,seq.position[index],position):
                            predictions.append([name,int(segment),mode,ts,*ref,*est])
    velocity=pd.DataFrame(velocity_rows);trajectory=pd.DataFrame(trajectory_rows)
    velocity.to_csv(output/'velocity_metrics.csv',index=False)
    velocity.groupby('method')[['rmse_mean','mae_mean']].mean().to_csv(output/'velocity_macro.csv')
    trajectory.to_csv(output/'trajectory_metrics.csv',index=False)
    pd.DataFrame(predictions,columns=['sequence','segment','orientation','time_s','ref_x','ref_y','ref_z','pred_x','pred_y','pred_z']).to_csv(output/'trajectories.csv',index=False)
    (output/'evaluation_protocol.json').write_text(json.dumps({'status':selection['test_status'],
        'initialization':'Reference p/v/q once per segment; no ongoing reference attitude/bias',
        'calibration':'Fixed gyro/accel biases learned on training references only; also report uncorrected orientation',
        'common_window':selection['common_window'],'stride':selection['stride'],
        'selection_checkpoint_sha256':selection['checkpoint_sha256']},indent=2))
    print(velocity.groupby('method')[['rmse_mean','mae_mean']].mean().to_string())
    print(trajectory[['sequence','method','position_rmse_m','final_error_m']].to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'results/optimized')
    evaluate(p.parse_args().output)
