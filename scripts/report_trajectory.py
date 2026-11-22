"""Report the trajectory-loss experiment, including an unchanged fallback."""
import argparse
import json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]


def report(output):
    selected=json.loads((output/'selection.json').read_text())
    leader=pd.read_csv(output/'leaderboard.csv')
    rows=pd.read_csv(output/'metrics.csv')
    lines=['# Trajectory-aware DL follow-up','',f"Selected policy: **{selected['selected']}**.",'',
           'Three loss weights (0.25, 1, 4) were tested for a small residual CNN on top of the frozen ensemble. The correction starts at zero, so unchanged inference is an explicit candidate. Each trial combines body-velocity supervision with integrated displacement loss over contiguous five-second chunks. No future sample enters a prediction; future reference motion appears only in training loss.', '',
           'Validation macro position RMSE selects checkpoints, subject to world-velocity RMSE remaining within 2% of the prior model. Orientation uses the same fixed training calibration and gyro propagation. There is no velocity damping, ongoing reference orientation, or test-driven tuning.', '',
           f"Baseline validation position RMSE: {selected['validation_prior_position_rmse']:.4f} m. Selected: {selected['validation_best_position_rmse']:.4f} m.", '',
           '## Trial outcomes','',leader.to_csv(index=False),'',
           '## Interpretation','',
           'None of the three trials beat the unchanged baseline on validation before early stopping. The exported residual is therefore the zero correction. This is not a successful extra DL improvement, and the original ensemble remains the recommended velocity model. All loss curves are retained; best_epoch=0 means the unchanged fallback won.', '',
           'The separate position-filter experiment can lower global position RMSE by damping velocities, but it worsens movement amplitude and local velocity accuracy. It does not resolve robust IO. More model complexity has not yet demonstrated a clean position gain on this data.', '',
           '## Measured comparison','',rows.groupby(['split','method'])[['position_rmse_m','zero_position_rmse_m','rmse_mean']].mean().to_csv(), '',
           'Float32 preparation causes tiny rounding differences from older double-precision reports. Evaluation sequences were inspected previously; these are development comparisons, not fresh confirmation. Reference position/orientation initialize each continuous segment once. No earlier checkpoint or result was overwritten.', '',
           '## Reproduce','', '```bash','python scripts/train_trajectory.py --output results/new_trajectory_training','python scripts/report_trajectory.py --output results/new_trajectory_training','```', '',
           'A completed selection cannot be overwritten. See training_plan.json, selection.json, the individual loss histories, and the frozen base-checkpoint hash. A full uncertainty-aware state estimator remains future work, not a claimed implementation here.']
    (output/'REPORT.md').write_text('\n'.join(lines))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'results/trajectory_training')
    report(p.parse_args().output)
