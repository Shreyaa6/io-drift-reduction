"""Residual DL correction trained with local trajectory loss; label-free inference."""
import numpy as np
import torch
from .model import VelocityCNN
from .inference import Predictor


class TrajectoryPredictor:
    def __init__(self,base_checkpoint,residual_checkpoint):
        self.base=Predictor(base_checkpoint);self.config=self.base.config
        saved=torch.load(residual_checkpoint,map_location='cpu',weights_only=True)
        self.residual=VelocityCNN().eval();self.residual.load_state_dict(saved['state_dict'])
        self.mean=np.asarray(saved['input_mean']);self.std=np.asarray(saved['input_std'])
        self.scale=np.asarray(saved['target_std'])

    def predict(self,features):
        base=self.base.predict(features);corrections=[]
        with torch.no_grad():
            for a in range(0,len(features),256):
                x=torch.tensor((features[a:a+256]-self.mean)/self.std,dtype=torch.float32)
                corrections.append(self.residual(x).numpy()*self.scale)
        return base+np.concatenate(corrections)
