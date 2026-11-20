"""Render stored results without retraining."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]


def plot(output):
    table=pd.read_csv(output/'test_velocity_macro.csv').sort_values('rmse_mean')
    fig,ax=plt.subplots(figsize=(7,4))
    ax.bar(table.method,table.rmse_mean,color=['#52796f' if m!='CNN' else '#457b9d' for m in table.method])
    ax.set(ylabel='Mean axis velocity RMSE (m/s)',title='EuRoC held-out sequences: body velocity')
    fig.tight_layout();fig.savefig(output/'velocity_comparison.png',dpi=160);plt.close(fig)
    curves=pd.read_csv(output/'learning_curve.csv')
    fig,ax=plt.subplots(figsize=(7,4));ax.plot(curves.epoch,curves.validation_macro_rmse)
    ax.set(xlabel='Epoch',ylabel='Validation macro RMSE (m/s)',title='Checkpoint selection uses validation only')
    fig.tight_layout();fig.savefig(output/'learning_curve.png',dpi=160);plt.close(fig)
    positions=pd.read_csv(output/'test_trajectories.csv')
    groups=list(positions.groupby(['sequence','segment']))
    fig,axes=plt.subplots(len(groups),1,figsize=(8,4*len(groups)),squeeze=False)
    for ax,((name,segment),data) in zip(axes[:,0],groups):
        ax.plot(data.ref_x,data.ref_y,label='Reference');ax.plot(data.cnn_x,data.cnn_y,label='CNN + gyro integration')
        ax.set(xlabel='World x (m)',ylabel='World y (m)',title=f'{name}, segment {segment}: initialized once')
        ax.axis('equal');ax.legend()
    fig.tight_layout();fig.savefig(output/'trajectories_xy.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'results/first_run')
    plot(p.parse_args().output)
