"""Fit CNN; select epoch and baseline on validation only. Never reads test windows."""
import argparse
import copy
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from io_drift.provenance import content_hash
from io_drift.metrics import velocity_metrics
from io_drift.model import VelocityCNN


def summarize(x):
    return np.concatenate([x.mean(1),x.std(1),x[:,-1]],axis=1)


def macro_rmse(y,pred,names):
    return float(np.mean([velocity_metrics(y[names==n],pred[names==n])['rmse_mean'] for n in np.unique(names)]))


def fit_baseline(train):
    summary=summarize(train['x'])
    scale=StandardScaler().fit(summary)
    ridge=Ridge(alpha=100.).fit(scale.transform(summary),train['y'])
    return scale,ridge


def train(prepared,output,epochs=None):
    config=json.loads((prepared/'config.json').read_text())
    if epochs is not None:
        config['epochs']=epochs
    if config['epochs']<1:
        raise ValueError('epochs must be positive')
    random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed'])
    torch.set_num_threads(config['threads'])
    tr=np.load(prepared/'train.npz');va=np.load(prepared/'validation.npz')
    mean=tr['x'].mean((0,1));std=tr['x'].std((0,1)).clip(1e-6)
    ym=tr['y'].mean(0);ys=tr['y'].std(0).clip(1e-6)
    tx=torch.from_numpy(((tr['x']-mean)/std).astype(np.float32))
    vx=torch.from_numpy(((va['x']-mean)/std).astype(np.float32))
    loader=DataLoader(TensorDataset(tx,torch.from_numpy((tr['y']-ym)/ys)),
                      batch_size=config['batch_size'],shuffle=True)
    model=VelocityCNN()
    opt=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    best=float('inf');stale=0;history=[];state=None;start=time.perf_counter()
    for epoch in range(config['epochs']):
        model.train();losses=[]
        for xb,yb in loader:
            opt.zero_grad();loss=nn.functional.mse_loss(model(xb),yb);loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            pred=model(vx).numpy()*ys+ym
        score=macro_rmse(va['y'],pred,va['sequence'])
        history.append({'epoch':epoch+1,'train_scaled_mse':float(np.mean(losses)),
                        'validation_macro_rmse':score})
        if score<best:
            best=score;state=copy.deepcopy(model.state_dict());best_epoch=epoch+1;stale=0
        else:
            stale+=1
        if epoch==0 or (epoch+1)%5==0:
            print(f'Epoch {epoch+1}: validation macro RMSE={score:.4f}',flush=True)
        if stale>=config['patience']:
            break
    model.load_state_dict(state);model.eval()
    with torch.no_grad():
        cnn=model(vx).numpy()*ys+ym
    scale,ridge=fit_baseline(tr)
    predictions={'Zero':np.zeros_like(va['y']),'Training mean':np.broadcast_to(ym,va['y'].shape),
                 'Ridge':ridge.predict(scale.transform(summarize(va['x']))),'CNN':cnn}
    scores={name:macro_rmse(va['y'],p,va['sequence']) for name,p in predictions.items()}
    selected=min(scores,key=scores.get)
    output.mkdir(parents=True,exist_ok=True)
    torch.save({'state_dict':state,'config':config,'input_mean':mean.tolist(),'input_std':std.tolist(),
                'target_mean':ym.tolist(),'target_std':ys.tolist()},output/'velocity_cnn.pt')
    pd.DataFrame(history).to_csv(output/'learning_curve.csv',index=False)
    provenance=json.loads((prepared.parent/'euroc/provenance.json').read_text()) if (prepared.parent/'euroc/provenance.json').exists() else None
    run={'created_utc':datetime.now(timezone.utc).isoformat(),'config':config,'best_epoch':best_epoch,
         'selected_on_validation':selected,'validation_macro_rmse':scores,
         'training_seconds':time.perf_counter()-start,'dataset':provenance,
         'prepared_hashes':{f:content_hash(prepared/f) for f in ['train.npz','validation.npz','config.json']},
         'versions':{'torch':str(torch.__version__),'numpy':np.__version__,'pandas':pd.__version__},
         'task':'body-frame reference velocity from gyro/accel/dt only',
         'initialization':'trajectory evaluation uses one reference p/v/q per continuous segment; no reference bias or ongoing attitude'}
    (output/'run.json').write_text(json.dumps(run,indent=2))
    print(json.dumps({'best_epoch':best_epoch,'validation_macro_rmse':scores,'selected':selected},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',type=Path,default=ROOT/'data/prepared')
    p.add_argument('--output',type=Path,default=ROOT/'results/first_run')
    p.add_argument('--epochs',type=int)
    a=p.parse_args();train(a.prepared,a.output,a.epochs)
