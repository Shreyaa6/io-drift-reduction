"""Matched endpoints and physically consistent training-only augmentation."""
import numpy as np
import torch


def common_windows(seq, window, sample_step, common_window=800, stride=20, max_gap_s=.025):
    if window > common_window or window < 2 or sample_step < 1 or stride < 1:
        raise ValueError('Invalid context/window configuration')
    bounds=np.r_[0,np.flatnonzero(np.diff(seq.time)>max_gap_s)+1,len(seq.time)]
    endpoints=np.concatenate([np.arange(a+common_window-1,b,stride,dtype=int)
                              for a,b in zip(bounds[:-1],bounds[1:])])
    if len(endpoints)==0:
        raise ValueError('No supported common windows')
    offsets=np.arange(window-1,-1,-sample_step)[::-1]
    x=seq.imu[endpoints[:,None]-window+1+offsets].astype(np.float32)
    return x,seq.body_velocity[endpoints].astype(np.float32),endpoints


def augment_raw(x,y,rotation=False,noise=False):
    """Change sensor xy axes consistently in gyro, accel and body-velocity label."""
    x=x.clone();y=y.clone()
    if rotation:
        angle=(torch.rand(len(x),device=x.device)*2-1)*torch.pi
        c,s=angle.cos(),angle.sin()
        for start in (0,3):
            a,b=x[:,:,start].clone(),x[:,:,start+1].clone()
            x[:,:,start]=c[:,None]*a-s[:,None]*b
            x[:,:,start+1]=s[:,None]*a+c[:,None]*b
        a,b=y[:,0].clone(),y[:,1].clone()
        y[:,0]=c*a-s*b;y[:,1]=s*a+c*b
    if noise:
        # Random fixed-in-window bias plus per-sample noise; labels unchanged.
        scales=x.new_tensor([.003]*3+[.03]*3)
        x[:,:,:6]+=torch.randn_like(x[:,:,:6])*scales
        x[:,:,:6]+=torch.randn_like(x[:,:1,:6])*scales*.5
    return x,y
