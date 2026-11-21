import copy
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from io_drift.advanced_models import make_model
from io_drift.optimization import augment_raw,common_windows
from io_drift.data import Sequence
from io_drift.inference import Predictor
from io_drift.calibration import fit_training_bias,apply_bias


class OptimizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_all_architectures_train_and_change_weights(self):
        for name in ('cnn','resnet','tcn','gru'):
            with self.subTest(architecture=name):
                torch.manual_seed(1);model=make_model(name)
                initial=[p.detach().clone() for p in model.parameters()]
                opt=torch.optim.AdamW(model.parameters(),lr=.001)
                pred=model(torch.randn(4,100,7))
                self.assertEqual(tuple(pred.shape),(4,3))
                loss=(pred-torch.randn(4,3)).square().mean();loss.backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
                opt.step()
                self.assertGreater(sum(float((p-i).square().sum().detach()) for p,i in zip(model.parameters(),initial)),0.)

    def test_rotation_preserves_norm_and_transforms_labels(self):
        x=torch.randn(8,100,7);y=x[:,0,:3].clone()
        augmented,labels=augment_raw(x,y,rotation=True)
        torch.testing.assert_close(augmented[:,0,:3],labels)
        torch.testing.assert_close(x[:,:,:3].norm(dim=-1),augmented[:,:,:3].norm(dim=-1))
        torch.testing.assert_close(x[:,:,3:6].norm(dim=-1),augmented[:,:,3:6].norm(dim=-1))
        torch.testing.assert_close(x[:,:,6],augmented[:,:,6])

    def test_contexts_have_identical_endpoints(self):
        n=1800;t=np.arange(n)*.005
        values=np.tile(np.arange(n)[:,None],(1,7));v=np.zeros((n,3));q=np.tile([0,0,0,1],(n,1))
        seq=Sequence('synthetic',t,values,v,v,q,v)
        previous=None
        for window,step in [(200,2),(400,4),(800,8)]:
            x,_,end=common_windows(seq,window,step)
            self.assertEqual(x.shape[1:],(100,7))
            np.testing.assert_equal(x[:,-1,0],end)
            if previous is not None:
                np.testing.assert_equal(end,previous)
            previous=end

    def test_ensemble_and_shrinkage_checkpoint(self):
        torch.manual_seed(2);model=make_model('resnet').eval()
        x=np.zeros((3,100,7),np.float32)
        member={'state_dict':model.state_dict(),'config':{'architecture':'resnet','window':400,'sample_step':4},
                'input_mean':[0.]*7,'input_std':[1.]*7,'target_mean':[0.]*3,'target_std':[1.]*3}
        saved={**member,'members':[member,copy.deepcopy(member)],'shrink_alpha':.5,'shrink_center':[0.]*3}
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'model.pt';torch.save(saved,path)
            result=Predictor(path).predict(x)
        with torch.no_grad():
            expected=.5*model(torch.from_numpy(x)).numpy()
        np.testing.assert_allclose(result,expected,rtol=1e-5,atol=1e-6)

    def test_training_calibration_recovers_synthetic_bias(self):
        t=np.arange(401)*.005
        q=Rotation.from_rotvec(np.column_stack([t*0,t*0,.2*t])).as_quat()
        imu=np.zeros((len(t),7));imu[:,:3]=[.01,.02,.23];imu[:,3:6]=[.1,-.1,10.01]
        velocity=np.zeros((len(t),3))
        seq=Sequence('train',t,imu,velocity,velocity,q,velocity)
        calibration=fit_training_bias([seq])
        np.testing.assert_allclose(calibration['gyro_bias'],[.01,.02,.03],atol=1e-10)
        np.testing.assert_allclose(calibration['accel_bias'],[.1,-.1,.2],atol=1e-10)
        corrected=apply_bias(imu,calibration)
        np.testing.assert_allclose(corrected[0,:6],[0,0,.2,0,0,9.81],atol=1e-10)
        np.testing.assert_equal(imu[0,:3],[.01,.02,.23])


if __name__=='__main__':
    unittest.main()
