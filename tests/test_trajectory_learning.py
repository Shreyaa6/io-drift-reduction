import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from io_drift.model import VelocityCNN
from io_drift.inference import Predictor
from io_drift.trajectory_learning import TrajectoryPredictor


class TrajectoryLearningTests(unittest.TestCase):
    def test_zero_residual_reproduces_base_without_labels(self):
        torch.manual_seed(42);base=VelocityCNN().eval();correction=VelocityCNN().eval()
        torch.nn.init.zeros_(correction.head[-1].weight);torch.nn.init.zeros_(correction.head[-1].bias)
        with tempfile.TemporaryDirectory() as folder:
            base_path=Path(folder)/'base.pt';residual_path=Path(folder)/'residual.pt'
            torch.save({'state_dict':base.state_dict(),'config':{'window':200,'sample_step':2},
                'input_mean':[0.]*7,'input_std':[1.]*7,'target_mean':[0.]*3,'target_std':[1.]*3},base_path)
            torch.save({'state_dict':correction.state_dict(),'input_mean':[0.]*7,'input_std':[1.]*7,
                        'target_std':[1.]*3},residual_path)
            values=np.zeros((3,100,7),np.float32)
            np.testing.assert_allclose(TrajectoryPredictor(base_path,residual_path).predict(values),
                                       Predictor(base_path).predict(values),atol=1e-8)

    def test_displacement_loss_has_finite_velocity_gradients(self):
        velocity=torch.ones((2,51,3),requires_grad=True)
        dt=torch.full((2,50,1),.1)
        displacement=torch.cumsum(.5*(velocity[:,1:]+velocity[:,:-1])*dt,dim=1)
        loss=displacement.square().mean();loss.backward()
        self.assertTrue(torch.isfinite(velocity.grad).all())
        self.assertGreater(float(velocity.grad.abs().sum()),0.)


if __name__=='__main__':
    unittest.main()
