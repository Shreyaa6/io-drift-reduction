"""Select a causal position pipeline on validation, freeze, then compare reused test."""
import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from io_drift.data import load_sequence
from io_drift.inference import Predictor
from io_drift.optimization import common_windows
from io_drift.calibration import apply_bias
from io_drift.position_filter import propagate_attitude,filter_velocity
from io_drift.navigation import integrate_velocity
from io_drift.metrics import velocity_metrics,trajectory_metrics


def get_tracks(names,predictor,calibration,gains):
    tracks=[]
    for name in names:
        seq=load_sequence(ROOT/'data/euroc',name)
        x,body,end=common_windows(seq,predictor.config['window'],predictor.config['sample_step'])
        pred=predictor.predict(x)
        segments=np.cumsum(np.r_[0,np.diff(seq.time)>.025])
        for segment in np.unique(segments[end]):
            keep=np.flatnonzero(segments[end]==segment)
            if len(keep)<2:
                continue
            index=end[keep];a,b=index[0],index[-1]
            corrected=apply_bias(seq.imu[a:b+1],calibration)
            attitudes={gain:Rotation.from_quat(propagate_attitude(seq.time[a:b+1],corrected,
                    seq.quaternion[a],gain)[index-a]) for gain in gains}
            tracks.append({'sequence':name,'segment':int(segment),'time':seq.time[index],
                'reference':seq.position[index],'reference_velocity':seq.velocity[index],
                'reference_body':body[keep],'body':pred[keep], 'attitudes':attitudes})
    return tracks


def assess(tracks,config,offset,save_positions=False):
    rows=[];positions=[]
    for track in tracks:
        body=filter_velocity(track['time'],track['body'],config['scale'],config['smoothing_s'],
                             offset if config['body_offset'] else None)
        world=track['attitudes'][config['gravity_gain']].apply(body)
        position=integrate_velocity(track['time'],world,track['reference'][0])
        metric=trajectory_metrics(track['time'],track['reference'],position)
        metric.update(velocity_metrics(track['reference_velocity'],world))
        metric['estimated_path_length_m']=float(np.linalg.norm(np.diff(position,axis=0),axis=1).sum())
        metric['reference_speed_mean']=float(np.linalg.norm(track['reference_velocity'],axis=1).mean())
        metric['estimated_speed_mean']=float(np.linalg.norm(world,axis=1).mean())
        zero=np.broadcast_to(track['reference'][0],position.shape)
        metric['zero_position_rmse_m']=trajectory_metrics(track['time'],track['reference'],zero)['position_rmse_m']
        metric['zero_velocity_rmse']=velocity_metrics(track['reference_velocity'],np.zeros_like(world))['rmse_mean']
        rows.append({'sequence':track['sequence'],'segment':track['segment'],**metric})
        if save_positions:
            for t,ref,est in zip(track['time'],track['reference'],position):
                positions.append([track['sequence'],track['segment'],t,*ref,*est])
    return pd.DataFrame(rows),positions


def macro(rows,column):
    return float(rows.groupby('sequence')[column].mean().mean())


def run(output):
    output.mkdir(parents=True,exist_ok=True)
    if (output/'selection.json').exists():
        raise ValueError('Completed selection exists: choose another output directory')
    plan=json.loads((ROOT/'configs/position_search.json').read_text())
    previous=json.loads((ROOT/'results/optimized/selection.json').read_text())
    checkpoint=ROOT/'results/optimized/best.pt'
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=previous['checkpoint_sha256']:
        raise ValueError('Optimized checkpoint changed')
    predictor=Predictor(checkpoint)
    calibration=json.loads((ROOT/'results/optimized/training_calibration.json').read_text())
    # Fit velocity bias using training sequences only. No evaluation reference used.
    offsets=[];train_audit=[]
    for name in previous['train']:
        seq=load_sequence(ROOT/'data/euroc',name)
        x,y,_=common_windows(seq,predictor.config['window'],predictor.config['sample_step'])
        residual=predictor.predict(x)-y
        offset=np.mean(residual,axis=0);offsets.append(offset)
        train_audit.append({'sequence':name,'mean_body_velocity_error':offset.tolist()})
    offset=np.mean(offsets,axis=0)
    (output/'training_velocity_offset.json').write_text(json.dumps({'offset':offset.tolist(),
        'training_audit':train_audit,'method':'Equal-sequence mean prediction-minus-reference body velocity on training only'},indent=2))
    # Load validation only for parameter selection. Position is now the primary objective.
    validation=get_tracks(previous['validation'],predictor,calibration,plan['gravity_gain'])
    candidates=[]
    for scale,bias,tau,gain in itertools.product(plan['scales'],plan['body_offset'],plan['smoothing_s'],plan['gravity_gain']):
        config={'scale':scale,'body_offset':bias,'smoothing_s':tau,'gravity_gain':gain}
        rows,_=assess(validation,config,offset)
        velocity=macro(rows,'rmse_mean');zero_velocity=macro(rows,'zero_velocity_rmse')
        eligible=velocity<zero_velocity if plan['require_velocity_better_than_zero'] else True
        candidates.append({**config,'position_rmse_m':macro(rows,'position_rmse_m'),
                           'velocity_rmse':velocity,'eligible':eligible})
    table=pd.DataFrame(candidates).sort_values('position_rmse_m')
    eligible=table[table.eligible]
    if eligible.empty:
        raise ValueError('No nonzero candidate passes velocity baseline constraint')
    best=eligible.iloc[0]
    chosen={'scale':float(best.scale),'body_offset':bool(best.body_offset),
            'smoothing_s':float(best.smoothing_s),'gravity_gain':float(best.gravity_gain)}
    nondamped=eligible[eligible.scale==1.].iloc[0]
    motion_preserving={'scale':1.,'body_offset':bool(nondamped.body_offset),
                       'smoothing_s':float(nondamped.smoothing_s),'gravity_gain':float(nondamped.gravity_gain)}
    prior={'scale':1.,'body_offset':False,'smoothing_s':0.,'gravity_gain':0.}
    validation_prior,_=assess(validation,prior,offset)
    selection={'config':chosen,'motion_preserving_config':motion_preserving,'search_plan':plan,'validation_position_rmse':float(best.position_rmse_m),
        'validation_prior_position_rmse':macro(validation_prior,'position_rmse_m'),
        'validation_zero_position_rmse':macro(validation_prior,'zero_position_rmse_m'),
        'checkpoint_sha256':previous['checkpoint_sha256'],'training_offset':offset.tolist(),
        'test_status':'Previously inspected development sequences; no fresh independent confirmation.',
        'initialization':'Reference position and orientation once per continuous segment; no ongoing reference inputs',
        'notes':'Postprocessing/attitude integration search; no new DL weights trained in this experiment'}
    table.to_csv(output/'validation_search.csv',index=False)
    (output/'selection.json').write_text(json.dumps(selection,indent=2))
    # Frozen configuration; only now load reused evaluation sequences.
    test=get_tracks(previous['test'],predictor,calibration,sorted(set([0.,chosen['gravity_gain'],motion_preserving['gravity_gain']])))
    results=[];saved_positions=[]
    for split,tracks in [('validation',validation),('development_test',test)]:
        for method,config in [('Prior optimized pipeline',prior),('Position-selected pipeline',chosen),('Motion-preserving pipeline',motion_preserving)]:
            rows,positions=assess(tracks,config,offset,save_positions=True)
            rows.insert(0,'split',split);rows.insert(1,'method',method);results.append(rows)
            for position in positions:
                saved_positions.append([split,method,*position])
    result=pd.concat(results,ignore_index=True)
    result.to_csv(output/'metrics.csv',index=False)
    pd.DataFrame(saved_positions,columns=['split','method','sequence','segment','time_s','ref_x','ref_y','ref_z','pred_x','pred_y','pred_z']).to_csv(output/'trajectories.csv',index=False)
    print(json.dumps(selection,indent=2),flush=True)
    print(result.groupby(['split','method'])[['position_rmse_m','zero_position_rmse_m','rmse_mean','estimated_path_length_m']].mean().to_string())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'results/position_reduction')
    run(p.parse_args().output)
