# EuRoC IMU-only first experiment — 22 November 2026

## Findings

A validation-selected CNN improves body-frame velocity estimation on the two held-out sequences. It does not yet deliver reliable position odometry. Velocity and position results must be reported together.

| Method | Test macro mean axis velocity RMSE (m/s) |
|---|---:|
| CNN | **0.415732** |
| Ridge on window summaries | 0.457824 |
| Training mean | 0.526191 |
| Zero velocity | 0.538798 |

CNN velocity RMSE is 9.19% lower than Ridge, 20.99% lower than training mean, and 22.84% lower than zero. Relative reductions use `(baseline - CNN) / baseline`. This is a new task and dataset, so none of these figures indicates an improvement over the old project's PX4–GPS residual score.

## Per-sequence check

| Sequence | Ridge RMSE | CNN RMSE | CNN position RMSE | Zero-velocity position RMSE |
|---|---:|---:|---:|---:|
| MH_05_difficult | 0.510340 | 0.427240 | 16.3648 m | 8.1126 m |
| V2_03_difficult | 0.405307 | 0.404224 | 7.2787 m | 2.1528 m |

The velocity gain is concentrated in MH_05; on V2_03 it is marginal. CNN velocity integration drifts and is worse than stationary and constant-initial-velocity position baselines. Trajectories can return near the starting point, so a small endpoint error for zero velocity does not imply accurate motion tracking; position RMSE over the whole path is also reported.

Raw uncalibrated IMU integration produces position RMSE around 28–30 km over these sequences, exposing its sensitivity to bias/orientation error. It receives no reference bias correction. Its failure is retained as a diagnostic, not used to advertise a misleadingly large navigation improvement. It is not a competitive modern IO baseline.

## Frozen protocol

- Train: six sequences; validation: MH_04 and V1_03; test: MH_05 and V2_03. V1_01 excluded for known reference-orientation concerns.
- 7,688 training windows, 2,015 validation windows and 2,240 test windows.
- Full 200-row (~1 s) histories, downsampled by two inside the window; stride 20 (~10 predictions/s). No reference values enter features.
- Training normalization only. Seed 42; AdamW; maximum 50 epochs; patience 8. Best epoch 12; early stopping after epoch 20. Checkpoint selected using validation macro RMSE.
- CNN validation macro RMSE: 0.434660; Ridge: 0.483758. Test evaluation followed model freeze. No test-driven model tuning was performed.
- Position/orientation and, for the classical integration baseline, velocity initialized from reference once per segment. No ongoing reference attitude or bias supplied. No fitted trajectory alignment or loop closure.
- Eight core tests passed before evaluation; additional content-provenance and inference checks are recorded in the repository tests. A label-free raw-IMU prediction command is also provided.

Exact configuration, versions, content hashes and source archive checksum are in `run.json`; the model checkpoint, raw metrics, trajectories and learning curve are included alongside this report. Dataset files are intentionally excluded from Git.

## Limits and next work

Single seed, small public dataset, shared training/test environments, and only two held-out trajectories limit conclusions. Body-velocity learning cannot universally resolve missing initial conditions or heading from arbitrary inertial measurements. The current gyro-only attitude propagation and biased velocity estimates produce poor position tracking.

Next: audit calibration/initialization, implement causal bias/orientation handling, compare a proper published IO baseline, broaden flight diversity, and evaluate on fresh reserved sequences. Selecting another architecture based on these test results would consume this test set; use validation or establish a new independent evaluation.

Team responsibilities are assignments for further work and review, documented in `docs/TEAM_PLAN.md`; they are not retroactive author credits.
