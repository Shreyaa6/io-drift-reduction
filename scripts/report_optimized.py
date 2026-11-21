"""Create comparison plots and an evidence-based report from stored search results."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]


def markdown_table(frame):
    columns=list(frame.columns)
    lines=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    for _,row in frame.iterrows():
        lines.append('| '+' | '.join(f'{v:.4f}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |')
    return '\n'.join(lines)


def report(output):
    selection=json.loads((output/'selection.json').read_text())
    leaderboard=pd.read_csv(output/'leaderboard.csv').sort_values('validation_macro_rmse')
    leaderboard['device']=leaderboard['device'].fillna('cpu')
    velocity=pd.read_csv(output/'velocity_macro.csv').set_index('method')
    trajectory=pd.read_csv(output/'trajectory_metrics.csv')
    original=velocity.loc['Original CNN','rmse_mean'];optimized=velocity.loc['Optimized DL','rmse_mean']
    relative=100*(1-optimized/original)
    val_relative=100*(1-selection['selected_validation_rmse']/selection['baseline_validation_rmse'])
    fig,ax=plt.subplots(figsize=(8,4.5));values=velocity.sort_values('rmse_mean')
    ax.bar(values.index,values.rmse_mean,color=['#28666e' if x=='Optimized DL' else '#8ca6a8' for x in values.index])
    ax.set(ylabel='Mean axis body-velocity RMSE (m/s)',title='Matched timestamps — reused evaluation sequences')
    fig.tight_layout();fig.savefig(output/'velocity_comparison.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5))
    labels=[f"{r.id} (seed {r.seed})" for _,r in leaderboard.iterrows()]
    ax.barh(labels,leaderboard.validation_macro_rmse,color='#457b9d');ax.invert_yaxis()
    ax.axvline(selection['baseline_validation_rmse'],color='#c05640',linestyle='--',label='Original CNN, matched endpoints')
    ax.set(xlabel='Validation macro velocity RMSE (m/s)',title='Architecture/context trials — validation only');ax.legend()
    fig.tight_layout();fig.savefig(output/'validation_search.png',dpi=160);plt.close(fig)
    data=pd.read_csv(output/'trajectories.csv')
    names=data.sequence.unique();fig,axes=plt.subplots(len(names),2,figsize=(12,4*len(names)),squeeze=False)
    for i,name in enumerate(names):
        for j,mode in enumerate(['raw','training-calibrated']):
            d=data[(data.sequence==name)&(data.orientation==mode)]
            axes[i,j].plot(d.ref_x,d.ref_y,label='Reference')
            axes[i,j].plot(d.pred_x,d.pred_y,label='Optimized DL')
            axes[i,j].set(title=f'{name}: {mode}',xlabel='World x (m)',ylabel='World y (m)')
            axes[i,j].axis('equal');axes[i,j].legend()
    fig.tight_layout();fig.savefig(output/'trajectories_xy.png',dpi=160);plt.close(fig)
    val_table=leaderboard[['id','seed','best_epoch','best_weights','validation_macro_rmse','weight_change_l2','device']]
    pos_table=trajectory[trajectory.method.str.startswith(('Optimized DL','Original CNN','Zero'))][['sequence','method','position_rmse_m','final_error_m']]
    notes=f'''# Deep-learning optimization results

## Scope and headline

Eight predeclared architecture/context/loss/augmentation configurations were trained, followed by two additional seeds for the validation-winning configuration. The selected predictor is **{selection['selected']}**, based on **{selection['winner_trial']['id']}**. Every network consumes only gyro/accelerometer/dt history. No images, GPS, or ongoing reference orientation enter the velocity network.

On common validation endpoints, selected RMSE is **{selection['selected_validation_rmse']:.4f} m/s**, versus **{selection['baseline_validation_rmse']:.4f} m/s** for the original CNN ({val_relative:.2f}% reduction).

On the already-inspected evaluation sequences, selected RMSE is **{optimized:.4f} m/s**, versus **{original:.4f} m/s** for the original CNN at the **same timestamps** ({relative:.2f}% reduction; a negative reduction means worse). Do not compare against the old 0.4157 score, which includes different startup samples.

{markdown_table(velocity.reset_index())}

## All completed training runs

{markdown_table(val_table)}

Each weight-change norm compares the trained checkpoint with that run's own initial weights. Original weight-audit details are stored separately. Training and validation curves, gradient norms, learning-rate history, chosen raw/EMA checkpoint, and finite-weight checks are retained in each trial directory. A completed CPU trial was retained while subsequent models used available MPS acceleration; an unfinished CPU residual run was restarted for acceleration and was not scored as a completed trial.

## Position tracking must be reported separately

{markdown_table(pos_table)}

Fixed gyro/accelerometer biases were fitted from training reference kinematics only, then applied unchanged. Both raw and training-calibrated orientation propagation are shown. Compare the same orientation mode when attributing improvement to DL. Each segment still receives reference position/orientation and classical initial velocity once. There is no ongoing reference correction or fitted alignment. These results do not establish unaided startup or reliable navigation; inspect all position errors and stationary baselines before making a drift-reduction claim.

## What was and was not optimized

Tested: longer temporal context, residual and dilated CNNs, a GRU with attention, balanced flight sampling, sensor-noise augmentation, coordinate-consistent rotation augmentation, MSE/Huber loss, dropout, AdamW, gradient clipping, learning-rate reduction, early stopping, averaged weights, multiple seeds, ensembling and validation-selected shrinkage. Every candidate and failed improvement remains visible in the records.

This is not an exhaustive search or a claim of the best possible neural network. Training data, motion observability, initialization, sensor calibration, and domain diversity impose limits that model size alone cannot solve. Adding every available DL technique can increase overfitting and compute without improving unseen-flight performance.

## Evaluation status

**{selection['test_status']}** All architecture, checkpoint, ensemble, and shrinkage selection used validation only, with selection saved and checkpoint hashed before this evaluation. These reused public sequences cannot provide fresh independent confirmation. Repeated validation searches also increase overfitting risk. Reserve new flight trajectories or another audited dataset for final confirmation; do not select further changes using these evaluation scores.

Reproduction: see `docs/OPTIMIZATION.md`, `search_plan.json`, `selection.json`, `leaderboard.csv`, and `evaluation_protocol.json`. The original first run remains unchanged.
'''
    (output/'REPORT.md').write_text(notes)
    print(f'Original={original:.6f}, optimized={optimized:.6f}, relative reduction={relative:.2f}%')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'results/optimized')
    report(p.parse_args().output)
