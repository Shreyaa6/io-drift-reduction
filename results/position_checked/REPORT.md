# Position-drift follow-up

## What changed

The optimized DL weights are unchanged. This experiment changes **causal velocity postprocessing and attitude propagation**, selected using validation position RMSE. It does not claim another neural-network accuracy gain.

A predeclared grid compared 54 candidates: velocity scales 0.5/0.75/1; optional training-fitted velocity offset; causal smoothing constants 0/0.3/1 s; accelerometer tilt gains 0/0.1/0.5. Every candidate had to beat zero velocity on validation velocity RMSE. No scale-zero, forced return to origin, future-sample filtering or ongoing reference corrections were allowed.

Validation selected `{'scale': 0.5, 'body_offset': False, 'smoothing_s': 0.0, 'gravity_gain': 0.1}`. It also selected a separate scale-one candidate `{'scale': 1.0, 'body_offset': False, 'smoothing_s': 0.0, 'gravity_gain': 0.1}` to expose how much improvement depends on velocity damping. The damped policy was first evaluated exploratorily; that exposed its motion-underestimation tradeoff. The motion-preserving policy was then selected from the existing validation grid before its comparison. All evaluation recordings were already inspected in earlier project stages; this is not fresh confirmation.

## Position metrics

- Zero-velocity position RMSE: **5.1762 m**.
- Previous optimized calibrated pipeline: **5.1788 m**.
- Position-selected damping pipeline: **3.3450 m**, **35.38% lower than zero velocity**, and 35.41% lower than the prior pipeline.
- Motion-preserving pipeline (scale 1): **5.2243 m**, **-0.93% reduction versus zero velocity** (negative means worse), and -0.88% reduction versus prior pipeline.

| sequence | method | position_rmse_m | final_error_m |
| --- | --- | --- | --- |
| MH_05_difficult | Prior optimized pipeline | 7.9809 | 12.1812 |
| V2_03_difficult | Prior optimized pipeline | 2.3767 | 4.8024 |
| MH_05_difficult | Position-selected pipeline | 4.9192 | 6.1280 |
| V2_03_difficult | Position-selected pipeline | 1.7708 | 3.3721 |
| MH_05_difficult | Motion-preserving pipeline | 7.9178 | 12.1531 |
| V2_03_difficult | Motion-preserving pipeline | 2.5309 | 5.0635 |

## The tradeoff: smaller position error is not automatically better odometry

The selected scale is 0.5. Damping halves velocity magnitude and estimated travelled distance. Estimated mean path length is 34.83 m versus reference 91.44 m. World-velocity RMSE changes from **0.3603 to 0.4020 m/s** (11.58% worse if positive). The one-second translation error is reported below and can also worsen.

On these bounded/returning trajectories, conservative movement can reduce global position RMSE while underestimating actual motion. This is a benchmark-specific regularization tradeoff, not evidence of universally accurate tracking. Do not quote the damped percentage alone as a general drift-reduction result. The no-damping candidate is the more appropriate ablation for orientation improvements.

| method | position_rmse_m | zero_position_rmse_m | rmse_mean | translation_rpe_1s_m | estimated_path_length_m | path_length_m |
| --- | --- | --- | --- | --- | --- | --- |
| Motion-preserving pipeline | 5.2243 | 5.1762 | 0.3605 | 0.5803 | 69.6627 | 91.4385 |
| Position-selected pipeline | 3.3450 | 5.1762 | 0.4020 | 0.6962 | 34.8314 | 91.4385 |
| Prior optimized pipeline | 5.1788 | 5.1762 | 0.3603 | 0.5797 | 69.6626 | 91.4385 |

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
