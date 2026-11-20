"""Predict body velocity from an IMU CSV only (EuRoC column order); no reference file."""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from io_drift.inference import Predictor


def predict_csv(source,checkpoint,output):
    model=Predictor(checkpoint);config=model.config
    frame=pd.read_csv(source)
    if len(frame.columns)!=7:
        raise ValueError('Expected timestamp(ns), gyro xyz(rad/s), specific force xyz(m/s²)')
    stamp=frame.iloc[:,0].to_numpy(np.int64)
    if len(stamp)<2 or (np.diff(stamp)<=0).any():
        raise ValueError('Strictly increasing timestamps required')
    dt=np.r_[0.,np.diff(stamp)/1e9]
    features=np.column_stack([frame.iloc[:,1:7].to_numpy(float),dt])
    if not np.isfinite(features).all():
        raise ValueError('Nonfinite IMU measurements')
    bounds=np.r_[0,np.flatnonzero(dt>config['max_gap_s']),len(stamp)]
    offsets=np.arange(config['window']-1,-1,-config['sample_step'])[::-1]
    starts=np.concatenate([np.arange(a,b-config['window']+1,config['stride'],dtype=int)
                           for a,b in zip(bounds[:-1],bounds[1:])])
    if len(starts)==0:
        raise ValueError('No full windows')
    predictions=[]
    for a in range(0,len(starts),256):
        predictions.append(model.predict(features[starts[a:a+256,None]+offsets]))
    velocity=np.concatenate(predictions)
    result=pd.DataFrame(velocity,columns=['body_vx_m_s','body_vy_m_s','body_vz_m_s'])
    result.insert(0,'timestamp_ns',stamp[starts+config['window']-1])
    output.parent.mkdir(parents=True,exist_ok=True);result.to_csv(output,index=False)
    print(f'Wrote {len(result)} body-velocity estimates to {output}; no reference data read')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('imu_csv',type=Path)
    p.add_argument('--checkpoint',type=Path,default=ROOT/'results/first_run/velocity_cnn.pt')
    p.add_argument('--output',type=Path,default=ROOT/'results/imu_predictions.csv')
    a=p.parse_args();predict_csv(a.imu_csv,a.checkpoint,a.output)
