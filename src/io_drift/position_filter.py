"""Causal attitude propagation and velocity regularization for position integration."""
import numpy as np
from scipy.spatial.transform import Rotation


def propagate_attitude(time, corrected_imu, initial_quaternion, gravity_gain=0.):
    """Gyro propagation with optional accelerometer tilt correction, never heading aiding.

    Gravity correction is attenuated during force-magnitude deviations. This heuristic
    does not prove the vehicle is stationary and cannot resolve yaw or constant velocity.
    """
    q=np.empty((len(time),4));q[0]=initial_quaternion
    rotation=Rotation.from_quat(initial_quaternion)
    filtered=corrected_imu[0,3:6].copy()
    for i in range(1,len(time)):
        dt=time[i]-time[i-1]
        if dt<=0:
            raise ValueError('Positive elapsed time required')
        alpha=1-np.exp(-dt/.3)
        filtered+=alpha*(corrected_imu[i,3:6]-filtered)
        magnitude=np.linalg.norm(filtered)
        correction=np.zeros(3)
        if gravity_gain and magnitude>1e-8:
            trust=np.exp(-((magnitude-9.81)/.6)**2)
            expected=rotation.inv().apply([0.,0.,1.])
            correction=gravity_gain*trust*np.cross(filtered/magnitude,expected)
        omega=.5*(corrected_imu[i-1,:3]+corrected_imu[i,:3])+correction
        rotation=rotation*Rotation.from_rotvec(omega*dt)
        q[i]=rotation.as_quat()
    return q


def filter_velocity(time, velocity, scale=1., smoothing_s=0., body_offset=None):
    if scale<=0 or smoothing_s<0:
        raise ValueError('Positive nonzero scale and nonnegative smoothing required')
    result=np.asarray(velocity,dtype=float).copy()
    if body_offset is not None:
        result-=np.asarray(body_offset)
    result*=scale
    if smoothing_s:
        for i in range(1,len(time)):
            dt=time[i]-time[i-1]
            if dt<=0:
                raise ValueError('Positive elapsed time required')
            alpha=1-np.exp(-dt/smoothing_s)
            result[i]=result[i-1]+alpha*(result[i]-result[i-1])
    return result
