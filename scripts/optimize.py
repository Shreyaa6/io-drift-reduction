"""Predeclared validation-only architecture/context search, EMA and seed ensemble."""
import argparse
import copy
import hashlib
import json
import random
import sys
import time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset,WeightedRandomSampler
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
from io_drift.advanced_models import make_model
from io_drift.data import load_sequence,validate_splits
from io_drift.inference import Predictor
from io_drift.optimization import common_windows,augment_raw
from train import macro_rmse


def dataset(sequences,trial,search):
    xs,ys,names,ends=[],[],[],[]
    for seq in sequences:
        x,y,e=common_windows(seq,trial['window'],trial['sample_step'],search['common_window'],search['stride'])
        xs.append(x);ys.append(y);names.extend([seq.name]*len(y));ends.append(e)
    return np.concatenate(xs),np.concatenate(ys),np.array(names),np.concatenate(ends)


def predict_batches(model,x,mean,std,ym,ys):
    result=[]
    model.eval()
    with torch.no_grad():
        for batch in torch.split(torch.from_numpy(x),256):
            result.append(model(((batch-mean)/std).to(next(model.parameters()).device)).cpu().numpy()*ys.numpy()+ym.numpy())
    return np.concatenate(result)


def fit(tr,va,trial,search,output):
    seed=trial['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if (output/'result.json').exists():
        cached=json.loads((output/'result.json').read_text())
        cached.setdefault('device','cpu')
        if cached['trial'] != trial:
            raise ValueError('Resume configuration mismatch')
        checkpoint=torch.load(output/'model.pt',map_location='cpu',weights_only=True)
        print(f"Resuming completed trial {trial['id']}",flush=True)
        return cached,checkpoint,np.load(output/'validation_predictions.npy')
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    x,y,names,_=tr;vx,vy,vnames,_=va
    mean=torch.tensor(x.mean((0,1)));std=torch.tensor(x.std((0,1)).clip(1e-6))
    ym=torch.tensor(y.mean(0));ys=torch.tensor(y.std(0).clip(1e-6))
    counts={n:int((names==n).sum()) for n in np.unique(names)}
    weights=torch.tensor([1/counts[n] for n in names],dtype=torch.double)
    sampler=WeightedRandomSampler(weights,len(y),replacement=True)
    loader=DataLoader(TensorDataset(torch.from_numpy(x),torch.from_numpy(y)),batch_size=search['batch_size'],sampler=sampler)
    model=make_model(trial['architecture']).to(device);ema=copy.deepcopy(model).eval()
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    opt=torch.optim.AdamW(model.parameters(),lr=search['learning_rate'],weight_decay=search['weight_decay'])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=search['scheduler_patience'],factor=.5,min_lr=1e-5)
    best=float('inf');stale=0;best_state=None;history=[];start=time.perf_counter()
    for epoch in range(search['epochs']):
        model.train();losses=[];gradient_norm=[]
        for xb,yb in loader:
            xb,yb=augment_raw(xb,yb,rotation='rotation' in trial['augmentation'],noise='noise' in trial['augmentation'])
            opt.zero_grad();pred=model(((xb-mean)/std).to(device));target=((yb-ym)/ys).to(device)
            loss=nn.functional.smooth_l1_loss(pred,target,beta=.5) if trial['loss']=='huber' else nn.functional.mse_loss(pred,target)
            loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            gradient_norm.append(float(norm));losses.append(float(loss.detach()))
            with torch.no_grad():
                for ep,mp in zip(ema.parameters(),model.parameters()):
                    ep.mul_(.98).add_(mp,alpha=.02)
        raw_pred=predict_batches(model,vx,mean,std,ym,ys)
        ema_pred=predict_batches(ema,vx,mean,std,ym,ys)
        raw_score=macro_rmse(vy,raw_pred,vnames);ema_score=macro_rmse(vy,ema_pred,vnames)
        score=min(raw_score,ema_score);chosen=model if raw_score<=ema_score else ema
        scheduler.step(score)
        history.append({'epoch':epoch+1,'train_loss':float(np.mean(losses)),
                        'validation_raw_rmse':raw_score,'validation_ema_rmse':ema_score,
                        'learning_rate':opt.param_groups[0]['lr'],'gradient_norm_mean':float(np.mean(gradient_norm))})
        if score<best:
            best=score;best_state={k:v.detach().cpu().clone() for k,v in chosen.state_dict().items()};best_epoch=epoch+1
            best_type='raw' if raw_score<=ema_score else 'ema';stale=0
        else:
            stale+=1
        if epoch==0 or (epoch+1)%10==0:
            print(f"{trial['id']} seed {seed} epoch {epoch+1}: validation={score:.4f}, best={best:.4f}",flush=True)
        if stale>=search['patience']:
            break
    state_delta=float(torch.sqrt(sum(torch.sum((best_state[k]-initial[k])**2) for k in initial)))
    initial_norm=float(torch.sqrt(sum(torch.sum(v**2) for v in initial.values())))
    result={'trial':trial,'best_epoch':best_epoch,'best_weights':best_type,'validation_macro_rmse':best,
            'epochs_run':len(history),'seconds':time.perf_counter()-start,'device':str(device),
            'parameter_count':sum(p.numel() for p in model.parameters()),
            'weight_change_l2':state_delta,'initial_weight_l2':initial_norm,
            'relative_weight_change':state_delta/initial_norm,'all_weights_finite':all(bool(torch.isfinite(v).all()) for v in best_state.values())}
    config={**json.loads((ROOT/'configs/euroc.json').read_text()),**trial,'stride':search['stride']}
    checkpoint={'state_dict':best_state,'config':config,'input_mean':mean.tolist(),'input_std':std.tolist(),
                'target_mean':ym.tolist(),'target_std':ys.tolist()}
    output.mkdir(parents=True,exist_ok=True)
    torch.save(checkpoint,output/'model.pt');pd.DataFrame(history).to_csv(output/'history.csv',index=False)
    (output/'result.json').write_text(json.dumps(result,indent=2))
    loaded=make_model(trial['architecture']);loaded.load_state_dict(best_state)
    prediction=predict_batches(loaded,vx,mean,std,ym,ys)
    np.save(output/'validation_predictions.npy',prediction)
    return result,checkpoint,prediction


def optimize(output,search):
    if (output/'selection.json').exists():
        raise ValueError('Completed search exists; use a new output directory')
    config=json.loads((ROOT/'configs/euroc.json').read_text());validate_splits(config)
    torch.set_num_threads(search['threads'])
    # Test sequence files and labels are never opened during search.
    seqs={split:[load_sequence(ROOT/'data/euroc',n) for n in config[split]] for split in ('train','validation')}
    output.mkdir(parents=True,exist_ok=True)
    if (output/'search_plan.json').exists() and json.loads((output/'search_plan.json').read_text()) != search:
        raise ValueError('Search plan changed; use a new directory')
    (output/'search_plan.json').write_text(json.dumps(search,indent=2))
    cache={}
    def arrays(trial):
        key=(trial['window'],trial['sample_step'])
        if key not in cache:
            cache[key]=(dataset(seqs['train'],trial,search),dataset(seqs['validation'],trial,search))
        return cache[key]
    results=[];candidates={}
    for trial in search['trials']:
        trial={**trial,'seed':search['seed']}
        tr,va=arrays(trial)
        result,checkpoint,pred=fit(tr,va,trial,search,output/'trials'/trial['id'])
        results.append(result);candidates[trial['id']]=(checkpoint,pred)
        pd.DataFrame([{**r['trial'],**{k:v for k,v in r.items() if k!='trial'}} for r in results]).to_csv(output/'leaderboard.csv',index=False)
    winner=min(results,key=lambda r:r['validation_macro_rmse'])
    trial=winner['trial'];tr,va=arrays(trial)
    members=[candidates[trial['id']][0]];predictions=[candidates[trial['id']][1]]
    for seed in search['repeat_seeds']:
        repeat={**trial,'seed':seed,'id':f"{trial['id']}_seed{seed}"}
        result,checkpoint,pred=fit(tr,va,repeat,search,output/'trials'/repeat['id'])
        results.append(result);candidates[repeat['id']]=(checkpoint,pred)
        members.append(checkpoint);predictions.append(pred)
    # Original checkpoint is a candidate so a failed search does not force regression.
    original_trial={'window':200,'sample_step':2}
    original_x,original_y,original_names,_=dataset(seqs['validation'],original_trial,search)
    original=Predictor(ROOT/'results/first_run/velocity_cnn.pt').predict(original_x)
    scores={'Original CNN':macro_rmse(original_y,original,original_names)}
    choices={'Original CNN':('original',None,None)}
    ym=tr[1].mean(0)
    # Select seed or same-architecture seed ensemble and mild shrinkage using validation only.
    member_specs=[(f"seed_{m['config']['seed']}",[m],p) for m,p in zip(members,predictions)]
    member_specs.append(('seed_ensemble',members,np.mean(predictions,axis=0)))
    for label,items,pred in member_specs:
        for alpha in (0.5,0.75,1.):
            name=f'{label}_shrink_{alpha}'
            shrunk=ym+alpha*(pred-ym)
            scores[name]=macro_rmse(va[1],shrunk,va[2]);choices[name]=(items,alpha,ym)
    selected=min(scores,key=scores.get)
    if selected=='Original CNN':
        chosen=torch.load(ROOT/'results/first_run/velocity_cnn.pt',weights_only=True)
    else:
        items,alpha,center=choices[selected]
        chosen={**items[0],'members':items,'shrink_alpha':alpha,'shrink_center':center.tolist()}
    torch.save(chosen,output/'best.pt')
    summary={'created_utc':datetime.now(timezone.utc).isoformat(),'selected':selected,'winner_trial':trial,
             'selection_metric':'equal-sequence mean axis body-velocity RMSE on common validation endpoints',
             'validation_candidates':scores,'baseline_validation_rmse':scores['Original CNN'],
             'selected_validation_rmse':scores[selected],'common_window':search['common_window'],'stride':search['stride'],
             'train':config['train'],'validation':config['validation'],'test':config['test'],
             'test_status':'Previously inspected in first run. Subsequent comparisons are development-set reuse, not fresh confirmation.',
             'dataset_sha256':json.loads((ROOT/'data/euroc/provenance.json').read_text())['sha256'],
             'checkpoint_sha256':hashlib.sha256((output/'best.pt').read_bytes()).hexdigest()}
    (output/'selection.json').write_text(json.dumps(summary,indent=2))
    pd.DataFrame([{**r['trial'],**{k:v for k,v in r.items() if k!='trial'}} for r in results]).to_csv(output/'leaderboard.csv',index=False)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'results/optimized')
    p.add_argument('--search',type=Path,default=ROOT/'configs/optimization.json')
    a=p.parse_args();optimize(a.output,json.loads(a.search.read_text()))
