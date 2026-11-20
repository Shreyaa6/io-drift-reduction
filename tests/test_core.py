import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from io_drift.data import Sequence,windows,validate_splits,load_sequence
from io_drift.navigation import integrate_imu,integrate_velocity
from io_drift.metrics import trajectory_metrics,velocity_metrics
from io_drift.model import VelocityCNN
from io_drift.inference import Predictor
from io_drift.provenance import content_hash


class CoreTests(unittest.TestCase):
    def sequence(self):
        t=np.r_[np.arange(4)*.005,1+np.arange(4)*.005]
        x=np.tile(np.arange(8)[:,None],(1,7)).astype(float)
        y=np.zeros((8,3));q=np.tile([0.,0.,0.,1.],(8,1))
        return Sequence('fake',t,x,y.copy(),y.copy(),q,y.copy())

    def test_gap_and_past_only_windows(self):
        seq=self.sequence();x,y,ends=windows(seq,4,1,1)
        np.testing.assert_equal(ends,[3,7]);np.testing.assert_equal(x[:,0,0],[0,4])
        seq.imu[4:]=99999
        other,_,_=windows(seq,4,1,1)
        np.testing.assert_equal(x[0],other[0])

    def test_reference_never_input(self):
        seq=self.sequence();x,_,_=windows(seq,4,1,1)
        seq.body_velocity[:]=999;seq.quaternion[:]=999;seq.position[:]=999
        other,_,_=windows(seq,4,1,1)
        np.testing.assert_equal(x,other)

    def test_split_overlap_rejected(self):
        with self.assertRaises(ValueError):
            validate_splits({'train':['a'],'validation':['b'],'test':['a']})

    def test_stationary_specific_force(self):
        t=np.arange(101)*.01;imu=np.zeros((101,7));imu[:,5]=9.81
        q,v,p=integrate_imu(t,imu,[0,0,0,1],[0,0,0],[0,0,0])
        np.testing.assert_allclose(v,0,atol=1e-10);np.testing.assert_allclose(p,0,atol=1e-10)
        np.testing.assert_allclose(q[-1],[0,0,0,1],atol=1e-10)

    def test_gyro_body_to_world(self):
        t=np.linspace(0,1,101);imu=np.zeros((101,7));imu[:,2]=np.pi/2
        q,_,_=integrate_imu(t,imu,[0,0,0,1],[0,0,0],[0,0,0])
        np.testing.assert_allclose(Rotation.from_quat(q[-1]).apply([1,0,0]),[0,1,0],atol=1e-8)

    def test_constant_velocity_position(self):
        t=np.array([0.,.1,.2]);v=np.tile([2.,0,0],(3,1))
        p=integrate_velocity(t,v,np.zeros(3))
        np.testing.assert_allclose(p[:,0],[0,.2,.4])
        self.assertEqual(trajectory_metrics(t,p,p)['position_rmse_m'],0.)
        self.assertEqual(velocity_metrics(v,v)['rmse_mean'],0.)

    def test_checkpoint_inference_without_labels(self):
        model=VelocityCNN().eval()
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'model.pt'
            torch.save({'state_dict':model.state_dict(),'config':{'window':20,'sample_step':2},
                        'input_mean':[0.]*7,'input_std':[1.]*7,
                        'target_mean':[0.]*3,'target_std':[1.]*3},path)
            result=Predictor(path).predict(np.zeros((2,10,7)))
            self.assertEqual(result.shape,(2,3));self.assertTrue(np.isfinite(result).all())

    def test_prepared_content_hash_ignores_zip_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            a,b=Path(temp)/'a.npz',Path(temp)/'b.npz'
            np.savez(a,x=np.arange(4),y=np.ones(2))
            np.savez_compressed(b,y=np.ones(2),x=np.arange(4))
            self.assertEqual(content_hash(a),content_hash(b))
            np.savez(b,x=np.arange(4)+1,y=np.ones(2))
            self.assertNotEqual(content_hash(a),content_hash(b))

    def test_loader_alignment_and_quaternion_order(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp)/'synthetic/mav0'
            (base/'imu0').mkdir(parents=True);(base/'state_groundtruth_estimate0').mkdir()
            stamp=1400000000000000000
            # Identity attitude; velocity [1,2,3] preserved in body frame.
            (base/'imu0/data.csv').write_text('t,gx,gy,gz,ax,ay,az\n'+ '\n'.join(
                f'{stamp+i*5000000},0,0,0,0,0,9.81' for i in range(5)))
            (base/'state_groundtruth_estimate0/data.csv').write_text('t,px,py,pz,qw,qx,qy,qz,vx,vy,vz,b1,b2,b3,b4,b5,b6\n'+ '\n'.join(
                f'{stamp+i*5000000},0,0,0,1,0,0,0,1,2,3,0,0,0,0,0,0' for i in range(1,4)))
            seq=load_sequence(temp,'synthetic')
            self.assertEqual(len(seq.time),3)
            np.testing.assert_allclose(seq.body_velocity,[[1,2,3]]*3)
            np.testing.assert_allclose(np.diff(seq.time),.005,atol=1e-12)


if __name__=='__main__':
    unittest.main()
