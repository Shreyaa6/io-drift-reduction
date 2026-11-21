"""Label-free body velocity inference; supports original CNN and seed ensembles."""
import numpy as np
import torch
from .advanced_models import make_model


class Predictor:
    def __init__(self, checkpoint):
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        self.config = saved['config']
        self.members=[]
        for member in saved.get('members',[saved]):
            model=make_model(member['config'].get('architecture','cnn')).eval()
            model.load_state_dict(member['state_dict'])
            self.members.append((model,np.array(member['input_mean']),np.array(member['input_std']),
                                 np.array(member['target_mean']),np.array(member['target_std'])))
        self.model,self.mean,self.std,self.ymean,self.ystd=self.members[0]
        self.alpha=float(saved.get('shrink_alpha',1.))
        self.center=np.array(saved.get('shrink_center',self.ymean))

    def predict(self, features):
        """NxLx7 raw gyro, accel, dt; caller constructs past-only gap-safe windows."""
        values = np.asarray(features)
        length = len(range(self.config['window']-1,-1,-self.config['sample_step']))
        if values.ndim != 3 or values.shape[1:] != (length,7) or not np.isfinite(values).all():
            raise ValueError(f'Expected finite Nx{length}x7 windows')
        if len(values)==0:
            raise ValueError('At least one window required')
        predictions=[]
        with torch.no_grad():
            for model,mean,std,ym,ys in self.members:
                batches=[]
                for a in range(0,len(values),256):
                    x=torch.tensor((values[a:a+256]-mean)/std,dtype=torch.float32)
                    batches.append(model(x).numpy()*ys+ym)
                predictions.append(np.concatenate(batches))
        pred=np.mean(predictions,axis=0)
        return self.center+self.alpha*(pred-self.center)
