"""Audit sequence-level splits and prepare causal training windows."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from io_drift.data import load_sequence, windows, validate_splits


def prepare(config, data_root, output):
    validate_splits(config)
    output.mkdir(parents=True,exist_ok=True)
    report=[]
    for split in ('train','validation','test'):
        xs,ys,names,ends=[],[],[],[]
        for name in config[split]:
            seq=load_sequence(data_root,name,config['max_gap_s'])
            x,y,index=windows(seq,**{k:config[k] for k in ('window','stride','sample_step','max_gap_s')})
            xs.append(x);ys.append(y);names.extend([name]*len(x));ends.append(index)
            report.append({'split':split,'sequence':name,'raw_supported_rows':len(seq.time),
                           'windows':len(x),'duration_s':float(seq.time[-1]-seq.time[0]),
                           'gaps':int((np.diff(seq.time)>config['max_gap_s']).sum())})
        np.savez_compressed(output/f'{split}.npz',x=np.concatenate(xs),y=np.concatenate(ys),
                            sequence=np.array(names),endpoint=np.concatenate(ends))
    (output/'config.json').write_text(json.dumps(config,indent=2))
    pd.DataFrame(report).to_csv(output/'audit.csv',index=False)
    print(pd.DataFrame(report).to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/euroc.json')
    p.add_argument('--data',type=Path,default=ROOT/'data/euroc')
    p.add_argument('--output',type=Path,default=ROOT/'data/prepared')
    a=p.parse_args()
    prepare(json.loads(a.config.read_text()),a.data,a.output)
