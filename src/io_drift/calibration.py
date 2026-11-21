"""Estimate fixed IMU biases from TRAINING reference motion, never evaluation labels."""
import numpy as np
from scipy.spatial.transform import Rotation


def fit_training_bias(sequences,max_gap_s=.025):
    gyro_residuals=[];accel_residuals=[];per_sequence=[]
    for seq in sequences:
        dt=np.diff(seq.time)
        valid=(dt>0)&(dt<=max_gap_s)
        rotations=Rotation.from_quat(seq.quaternion)
        angular=(rotations[:-1].inv()*rotations[1:]).as_rotvec()/dt[:,None]
        mid=rotations[:-1]*Rotation.from_rotvec(angular*dt[:,None]*.5)
        force=mid.inv().apply(np.diff(seq.velocity,axis=0)/dt[:,None]-np.array([0,0,-9.81]))
        gyro=.5*(seq.imu[:-1,:3]+seq.imu[1:,:3])-angular
        accel=.5*(seq.imu[:-1,3:6]+seq.imu[1:,3:6])-force
        gyro_residuals.append(gyro[valid]);accel_residuals.append(accel[valid])
        per_sequence.append({'sequence':seq.name,'gyro_bias':np.median(gyro[valid],axis=0).tolist(),
                             'accel_bias':np.median(accel[valid],axis=0).tolist()})
    # Equal weight across flights rather than allowing long sequences to dominate.
    return {'gyro_bias':np.median([x['gyro_bias'] for x in per_sequence],axis=0).tolist(),
            'accel_bias':np.median([x['accel_bias'] for x in per_sequence],axis=0).tolist(),
            'training_sequences':per_sequence,
            'method':'Median training-reference kinematic residual per sequence, then median across sequences',
            'limitations':'Fixed sensor calibration; may not transfer to another sensor, date or temperature'}


def apply_bias(imu,calibration):
    corrected=np.array(imu,copy=True)
    corrected[:,:3]-=calibration['gyro_bias']
    corrected[:,3:6]-=calibration['accel_bias']
    return corrected
