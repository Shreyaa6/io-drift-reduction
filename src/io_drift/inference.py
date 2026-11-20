"""Label-free body velocity inference from timestamped raw IMU windows."""
import numpy as np
import torch
from .model import VelocityCNN


class Predictor:
    def __init__(self, checkpoint):
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        self.config = saved['config']
        self.mean, self.std = np.array(saved['input_mean']), np.array(saved['input_std'])
        self.ymean, self.ystd = np.array(saved['target_mean']), np.array(saved['target_std'])
        self.model = VelocityCNN().eval()
        self.model.load_state_dict(saved['state_dict'])

    def predict(self, features):
        """NxLx7 raw gyro, accel, dt; caller constructs past-only gap-safe windows."""
        values = np.asarray(features)
        length = len(range(self.config['window']-1,-1,-self.config['sample_step']))
        if values.ndim != 3 or values.shape[1:] != (length,7) or not np.isfinite(values).all():
            raise ValueError(f'Expected finite Nx{length}x7 windows')
        with torch.no_grad():
            pred = self.model(torch.tensor((values-self.mean)/self.std,dtype=torch.float32)).numpy()
        return pred*self.ystd+self.ymean
