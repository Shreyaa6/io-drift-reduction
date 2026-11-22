import sys
import unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from io_drift.position_filter import propagate_attitude,filter_velocity
from io_drift.navigation import integrate_imu


class PositionFilterTests(unittest.TestCase):
    def test_zero_gain_matches_prior_gyro_propagation(self):
        time=np.arange(101)*.005
        imu=np.zeros((101,7));imu[:,:3]=[.1,-.05,.2];imu[:,5]=9.81
        q=propagate_attitude(time,imu,[0,0,0,1],0.)
        previous,_,_=integrate_imu(time,imu,[0,0,0,1],[0,0,0],[0,0,0])
        np.testing.assert_allclose(q,previous,atol=1e-12)

    def test_gravity_correction_reduces_static_tilt_error(self):
        time=np.arange(1001)*.01
        true=Rotation.from_rotvec([.2,0,0])
        imu=np.zeros((len(time),7));imu[:,3:6]=true.inv().apply([0,0,9.81])
        result=propagate_attitude(time,imu,[0,0,0,1],1.)
        error=(Rotation.from_quat(result[-1]).inv()*true).magnitude()
        self.assertLess(error,.001)

    def test_smoothing_has_no_future_dependency(self):
        time=np.arange(20)*.1;velocity=np.arange(60,dtype=float).reshape(20,3)
        a=filter_velocity(time,velocity,.75,.3)
        velocity[10:]=99999
        b=filter_velocity(time,velocity,.75,.3)
        np.testing.assert_equal(a[:10],b[:10])

    def test_constant_velocity_and_nonzero_scale(self):
        time=np.arange(10)*.1;velocity=np.tile([1.,2.,3.],(10,1))
        result=filter_velocity(time,velocity,1.,.3,body_offset=[.1,.2,.3])
        np.testing.assert_allclose(result,np.tile([.9,1.8,2.7],(10,1)))
        with self.assertRaises(ValueError):
            filter_velocity(time,velocity,0.)


if __name__=='__main__':
    unittest.main()
