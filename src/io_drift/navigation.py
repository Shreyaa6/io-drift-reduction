"""Reference used only for explicit one-time p/v/q initialization per segment."""
import numpy as np
from scipy.spatial.transform import Rotation


def integrate_imu(time, imu, initial_quaternion, initial_velocity, initial_position):
    n = len(time)
    rotations = np.empty((n,4))
    velocity, position = np.empty((n,3)), np.empty((n,3))
    rotations[0], velocity[0], position[0] = initial_quaternion, initial_velocity, initial_position
    current = Rotation.from_quat(initial_quaternion)
    for i in range(1,n):
        dt = time[i]-time[i-1]
        if dt <= 0:
            raise ValueError('Nonpositive navigation dt')
        angular = .5*(imu[i-1,:3]+imu[i,:3])
        middle = current * Rotation.from_rotvec(angular*dt*.5)
        acceleration = middle.apply(.5*(imu[i-1,3:6]+imu[i,3:6])) + [0,0,-9.81]
        velocity[i] = velocity[i-1]+acceleration*dt
        position[i] = position[i-1]+.5*(velocity[i-1]+velocity[i])*dt
        current = current * Rotation.from_rotvec(angular*dt)
        rotations[i] = current.as_quat()
    return rotations, velocity, position


def integrate_velocity(time, world_velocity, initial_position):
    delta = .5*(world_velocity[1:]+world_velocity[:-1])*np.diff(time)[:,None]
    return np.vstack([initial_position, initial_position+np.cumsum(delta,axis=0)])
