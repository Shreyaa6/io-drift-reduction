"""Validation-position-selected residual learning with multi-step displacement loss."""
import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from io_drift.data import load_sequence
from io_drift.optimization import common_windows
from io_drift.calibration import apply_bias
from io_drift.position_filter import propagate_attitude
from io_drift.inference import Predictor
from io_drift.model import VelocityCNN
from io_drift.navigation import integrate_velocity
from io_drift.metrics import trajectory_metrics,velocity_metrics


class Chunks(Dataset):
    def __init__(self,tracks,length=51):
        self.tracks=tracks;self.length=length
        self.starts=[(i,j) for i,t in enumerate(tracks) for j in range(len(t['x'])-length+1)]
        counts={i:sum(k==i for k,_ in self.starts) for i in range(len(tracks))}
        self.weights=[1/counts[i] for i,_ in self.starts]

    def __len__(self):
        return len(self.starts)

    def __getitem__(self,index):
        i,j=self.starts[index];t=self.tracks[i];sl=slice(j,j+self.length)
        return tuple(torch.from_numpy(np.asarray(t[k][sl],dtype=np.float32)) for k in ['x','base','body','rotation','position','time'])


def tracks_for(names,base,calibration):
    tracks=[]
    for name in names:
        seq=load_sequence(ROOT/'data/euroc',name)
        x,y,end=common_windows(seq,base.config['window'],base.config['sample_step'])
        pred=base.predict(x)
        groups=np.cumsum(np.r_[0,np.diff(seq.time)>.025])
        for segment in np.unique(groups[end]):
            keep=np.flatnonzero(groups[end]==segment);index=end[keep];a,b=index[0],index[-1]
            q=propagate_attitude(seq.time[a:b+1],apply_bias(seq.imu[a:b+1],calibration),seq.quaternion[a],0.)
            tracks.append({'name':name,'segment':int(segment),'x':x[keep],'body':y[keep],
                'base':pred[keep].astype(np.float32),'rotation':Rotation.from_quat(q[index-a]).as_matrix().astype(np.float32),
                'position':seq.position[index].astype(np.float32),'time':(seq.time[index]-seq.time[index[0]]).astype(np.float32),
                'velocity':seq.velocity[index]})
    return tracks


def evaluate(model,tracks,mean,std,scale,device):
    rows=[];predictions=[];model.eval()
    with torch.no_grad():
        for track in tracks:
            delta=[]
            for a in range(0,len(track['x']),256):
                x=torch.tensor(track['x'][a:a+256],device=device)
                delta.append((model((x-mean.to(device))/std.to(device))*scale.to(device)).cpu().numpy())
            body=track['base']+np.concatenate(delta)
            world=np.einsum('nij,nj->ni',track['rotation'],body)
            position=integrate_velocity(track['time'],world,track['position'][0])
            rows.append({'sequence':track['name'],'segment':track['segment'],
                         **trajectory_metrics(track['time'],track['position'],position),
                         **velocity_metrics(track['velocity'],world)})
            predictions.append((track,position,body))
    return pd.DataFrame(rows),predictions


def score(rows,key):
    return float(rows.groupby('sequence')[key].mean().mean())


def train(output):
    if (output/'selection.json').exists():
        raise ValueError('Selection already exists; use another directory')
    output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    previous=json.loads((ROOT/'results/optimized/selection.json').read_text())
    base=Predictor(ROOT/'results/optimized/best.pt')
    calibration=json.loads((ROOT/'results/optimized/training_calibration.json').read_text())
    training=tracks_for(previous['train'],base,calibration)
    validation=tracks_for(previous['validation'],base,calibration)
    dataset=Chunks(training)
    x=np.concatenate([t['x'] for t in training]);y=np.concatenate([t['body'] for t in training])
    mean=torch.tensor(x.mean((0,1)));std=torch.tensor(x.std((0,1)).clip(1e-6));scale=torch.tensor(y.std(0).clip(1e-6))
    torch.manual_seed(42)
    zero=VelocityCNN().to(device)
    nn.init.zeros_(zero.head[-1].weight);nn.init.zeros_(zero.head[-1].bias)
    prior_rows,_=evaluate(zero,validation,mean,std,scale,device)
    prior_position=score(prior_rows,'position_rmse_m');prior_velocity=score(prior_rows,'rmse_mean')
    global_best=prior_position;global_state={k:v.detach().cpu().clone() for k,v in zero.state_dict().items()};selected='Unchanged baseline'
    leaderboard=[]
    for weight in [.25,1.,4.]:
        torch.manual_seed(42);np.random.seed(42)
        model=VelocityCNN().to(device);nn.init.zeros_(model.head[-1].weight);nn.init.zeros_(model.head[-1].bias)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.001)
        sampler=WeightedRandomSampler(dataset.weights,512,replacement=True)
        loader=DataLoader(dataset,batch_size=8,sampler=sampler)
        best=prior_position;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};stale=0;history=[];best_epoch=0
        for epoch in range(30):
            model.train();losses=[]
            for xb,base_velocity,target_body,rotation,position,time in loader:
                xb=xb.to(device);base_velocity=base_velocity.to(device);target_body=target_body.to(device)
                rotation=rotation.to(device);position=position.to(device);time=time.to(device)
                batch,length=xb.shape[:2]
                correction=model(((xb-mean.to(device))/std.to(device)).reshape(-1,100,7)).reshape(batch,length,3)*scale.to(device)
                body=base_velocity+correction
                body_loss=((body-target_body)/scale.to(device)).square().mean()
                world=torch.einsum('btij,btj->bti',rotation,body)
                dt=time[:,1:]-time[:,:-1]
                displacement=torch.cumsum(.5*(world[:,1:]+world[:,:-1])*dt[:,:,None],dim=1)
                reference=position[:,1:]-position[:,:1]
                # Average-motion error over 1..5 second horizons; no absolute position input.
                elapsed=(time[:,1:]-time[:,:1]).clamp_min(1.)[:,:,None]
                position_loss=((displacement-reference)/elapsed/scale.to(device)).square().mean()
                loss=body_loss+weight*position_loss
                optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                losses.append(float(loss.detach()))
            rows,_=evaluate(model,validation,mean,std,scale,device)
            value=score(rows,'position_rmse_m');velocity=score(rows,'rmse_mean')
            eligible=velocity<=prior_velocity*1.02
            history.append({'epoch':epoch+1,'loss':float(np.mean(losses)),
                            'validation_position_rmse_m':value,'validation_world_velocity_rmse':velocity,'eligible':eligible})
            if eligible and value<best:
                best=value;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};stale=0;best_epoch=epoch+1
            else:
                stale+=1
            if epoch==0 or (epoch+1)%5==0:
                print(f'position weight {weight} epoch {epoch+1}: pos={value:.4f}, vel={velocity:.4f}, eligible={eligible}',flush=True)
            if stale>=7:
                break
        folder=output/f'lambda_{weight}';folder.mkdir(exist_ok=True)
        pd.DataFrame(history).to_csv(folder/'history.csv',index=False)
        torch.save({'state_dict':best_state,'input_mean':mean.tolist(),'input_std':std.tolist(),'target_std':scale.tolist()},folder/'residual.pt')
        leaderboard.append({'weight':weight,'best_epoch':best_epoch,'validation_position_rmse':best})
        if best<global_best:
            global_best=best;global_state=copy.deepcopy(best_state);selected=f'position_weight_{weight}'
    torch.save({'state_dict':global_state,'input_mean':mean.tolist(),'input_std':std.tolist(),'target_std':scale.tolist()},output/'best_residual.pt')
    selection={'selected':selected,'validation_prior_position_rmse':prior_position,'validation_best_position_rmse':global_best,
        'validation_prior_world_velocity_rmse':prior_velocity,'constraint':'validation world-velocity RMSE <= 1.02 times prior',
        'base_checkpoint_sha256':previous['checkpoint_sha256'],
        'residual_sha256':hashlib.sha256((output/'best_residual.pt').read_bytes()).hexdigest(),
        'initialization':'Reference p/q once per segment; calibrated gyro orientation; no velocity damping',
        'test_status':'Reused development sequences; no independent final confirmation'}
    (output/'selection.json').write_text(json.dumps(selection,indent=2));pd.DataFrame(leaderboard).to_csv(output/'leaderboard.csv',index=False)
    # Only now inspect the previously used evaluation data for this frozen residual.
    model=VelocityCNN().to(device);model.load_state_dict(global_state)
    test=tracks_for(previous['test'],base,calibration)
    result=[];positions=[]
    for split,tracks in [('validation',validation),('development_test',test)]:
        for method,net in [('Prior ensemble',zero),('Trajectory-trained correction',model)]:
            rows,pred=evaluate(net,tracks,mean,std,scale,device);rows.insert(0,'split',split);rows.insert(1,'method',method)
            for track,position,body in pred:
                zero_position=np.broadcast_to(track['position'][0],track['position'].shape)
                z=trajectory_metrics(track['time'],track['position'],zero_position)['position_rmse_m']
                rows.loc[(rows.sequence==track['name'])&(rows.segment==track['segment']),'zero_position_rmse_m']=z
                for t,ref,est in zip(track['time'],track['position'],position):
                    positions.append([split,method,track['name'],t,*ref,*est])
            result.append(rows)
    result=pd.concat(result,ignore_index=True);result.to_csv(output/'metrics.csv',index=False)
    pd.DataFrame(positions,columns=['split','method','sequence','time_s','ref_x','ref_y','ref_z','pred_x','pred_y','pred_z']).to_csv(output/'trajectories.csv',index=False)
    print(json.dumps(selection,indent=2),flush=True)
    print(result.groupby(['split','method'])[['position_rmse_m','zero_position_rmse_m','rmse_mean']].mean().to_string())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'results/trajectory_training')
    train(p.parse_args().output)
