"""Report position gains alongside velocity, local error and motion-amplitude tradeoffs."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]


def table(frame):
    lines=['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
    for _,row in frame.iterrows():
        lines.append('| '+' | '.join(f'{v:.4f}' if isinstance(v,float) else str(v) for v in row)+' |')
    return '\n'.join(lines)


def report(output):
    metrics=pd.read_csv(output/'metrics.csv');selection=json.loads((output/'selection.json').read_text())
    test=metrics[metrics.split=='development_test']
    summary=test.groupby('method')[['position_rmse_m','zero_position_rmse_m','rmse_mean',
                                  'translation_rpe_1s_m','estimated_path_length_m','path_length_m']].mean()
    old=summary.loc['Prior optimized pipeline','position_rmse_m']
    damp=summary.loc['Position-selected pipeline','position_rmse_m']
    preserve=summary.loc['Motion-preserving pipeline','position_rmse_m']
    zero=summary.zero_position_rmse_m.iloc[0]
    reduction=100*(1-damp/zero);preserve_reduction=100*(1-preserve/zero)
    velocity_old=summary.loc['Prior optimized pipeline','rmse_mean']
    velocity_damp=summary.loc['Position-selected pipeline','rmse_mean']
    summary.to_csv(output/'macro_metrics.csv')
    fig,ax=plt.subplots(figsize=(9,4.5))
    labels=['Zero velocity','Prior pipeline','No velocity damping','Position-selected damping']
    ax.bar(labels,[zero,old,preserve,damp],color=['#a8b8b4','#8ca6a8','#457b9d','#b78854'])
    ax.set(ylabel='Position RMSE (m)',title='Position error — reused development trajectories')
    ax.tick_params(axis='x',labelrotation=12);fig.tight_layout()
    fig.savefig(output/'position_comparison.png',dpi=160);plt.close(fig)
    trajectories=pd.read_csv(output/'trajectories.csv')
    trajectories=trajectories[trajectories.split=='development_test']
    names=trajectories.sequence.unique();fig,axes=plt.subplots(1,len(names),figsize=(12,5),squeeze=False)
    for ax,name in zip(axes[0],names):
        data=trajectories[trajectories.sequence==name]
        reference=data[data.method=='Prior optimized pipeline']
        ax.plot(reference.ref_x,reference.ref_y,label='Reference',color='black')
        for method,group in data.groupby('method'):
            ax.plot(group.pred_x,group.pred_y,label=method,alpha=.8)
        ax.set(title=name,xlabel='World x (m)',ylabel='World y (m)');ax.axis('equal');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(output/'trajectories_xy.png',dpi=160);plt.close(fig)
    text=f'''# Position-drift follow-up

## What changed

The optimized DL weights are unchanged. This experiment changes **causal velocity postprocessing and attitude propagation**, selected using validation position RMSE. It does not claim another neural-network accuracy gain.

A predeclared grid compared 54 candidates: velocity scales 0.5/0.75/1; optional training-fitted velocity offset; causal smoothing constants 0/0.3/1 s; accelerometer tilt gains 0/0.1/0.5. Every candidate had to beat zero velocity on validation velocity RMSE. No scale-zero, forced return to origin, future-sample filtering or ongoing reference corrections were allowed.

Validation selected `{selection['config']}`. It also selected a separate scale-one candidate `{selection['motion_preserving_config']}` to expose how much improvement depends on velocity damping. The damped policy was first evaluated exploratorily; that exposed its motion-underestimation tradeoff. The motion-preserving policy was then selected from the existing validation grid before its comparison. All evaluation recordings were already inspected in earlier project stages; this is not fresh confirmation.

## Position metrics

- Zero-velocity position RMSE: **{zero:.4f} m**.
- Previous optimized calibrated pipeline: **{old:.4f} m**.
- Position-selected damping pipeline: **{damp:.4f} m**, **{reduction:.2f}% lower than zero velocity**, and {100*(1-damp/old):.2f}% lower than the prior pipeline.
- Motion-preserving pipeline (scale 1): **{preserve:.4f} m**, **{preserve_reduction:.2f}% reduction versus zero velocity** (negative means worse), and {100*(1-preserve/old):.2f}% reduction versus prior pipeline.

{table(test[['sequence','method','position_rmse_m','final_error_m']])}

## The tradeoff: smaller position error is not automatically better odometry

The selected scale is {selection['config']['scale']}. Damping halves velocity magnitude and estimated travelled distance. Estimated mean path length is {summary.loc['Position-selected pipeline','estimated_path_length_m']:.2f} m versus reference {summary.loc['Position-selected pipeline','path_length_m']:.2f} m. World-velocity RMSE changes from **{velocity_old:.4f} to {velocity_damp:.4f} m/s** ({100*(velocity_damp/velocity_old-1):.2f}% worse if positive). The one-second translation error is reported below and can also worsen.

On these bounded/returning trajectories, conservative movement can reduce global position RMSE while underestimating actual motion. This is a benchmark-specific regularization tradeoff, not evidence of universally accurate tracking. Do not quote the damped percentage alone as a general drift-reduction result. The no-damping candidate is the more appropriate ablation for orientation improvements.

{table(summary.reset_index())}

## Method boundaries

Reference position and orientation are supplied once at each continuous segment start. Gyro/accelerometer calibration and optional velocity offset are learned from training references only. Accelerometer feedback corrects tilt heuristically, attenuated when force magnitude deviates from gravity. It does not observe heading, initial velocity or distinguish stationary motion from constant-velocity travel. Dynamic acceleration may corrupt the gravity assumption. No zero-velocity detector was used.

No body-velocity weights or earlier result files were overwritten. The prior pipeline is reproduced on the same endpoints. Velocity metric here is in world coordinates after estimated attitude; do not directly compare it with the earlier body-velocity RMSE of 0.3641 m/s.

## Reproduce

```bash
python scripts/reduce_position_drift.py --output results/new_position_search
python scripts/report_position.py --output results/new_position_search
```

Existing completed selections cannot be overwritten silently. Fresh, more varied flights and trajectory-aware training/filtering are needed before claiming robust IO. See the versioned configuration, validation table, initialization notes and all metrics in this directory.

Relevant research direction: TLIO fuses learned relative displacement and uncertainty with a filter, rather than relying only on velocity integration: https://arxiv.org/abs/2007.01867 . That published system was not implemented or reproduced in this follow-up.
'''
    (output/'REPORT.md').write_text(text)
    print(f'Zero={zero:.6f}, prior={old:.6f}, no-damping={preserve:.6f}, damped={damp:.6f}')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'results/position_checked')
    report(p.parse_args().output)
