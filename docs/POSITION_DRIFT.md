# Reducing position error: what the follow-up established

The user asked whether the calibrated position error (5.1788 m) could be reduced below zero velocity (5.1762 m). Two separate experiments were performed; results should not be merged into a misleading success claim.

## Causal filter search

`configs/position_search.json` declares a 54-candidate grid over nonzero velocity scale, training-derived body-velocity offset, past-only exponential smoothing and accelerometer tilt correction. Selection uses validation position RMSE and requires validation world-velocity RMSE better than zero velocity. Parameters and unchanged base-checkpoint hash are frozen before each reported comparison.

The main selection halves velocity and uses a small accelerometer tilt gain. It reduces evaluation global position RMSE to 3.3450 m (35.38% below zero velocity), but also halves travelled distance relative to the prior model, underestimates reference movement, worsens world-velocity RMSE and local translation error. The result is benchmark-specific position regularization, not proof of more accurate odometry. Returning/contained paths can reward under-motion in global RMSE. No forced loop closure or reference-based return to origin is applied.

A second candidate fixes velocity scale at one and is selected from the same validation table. It gives 5.2243 m on the reused evaluation sequences, worse than both prior and zero velocity. Optional training-derived velocity-bias subtraction and smoothing did not win. Therefore no filter-based change is promoted as a robust replacement for the default ensemble.

Accelerometer correction uses a causal force low-pass estimate and estimated gravity direction, attenuated if magnitude deviates from 9.81 m/s². Its correction sign and causality are tested. It provides no heading observation, cannot distinguish rest from constant velocity, and can be corrupted by dynamic acceleration. It is not an EKF or a reliable stationary detector.

Reproduce with:

```bash
python scripts/reduce_position_drift.py --output results/new_position_search
python scripts/report_position.py --output results/new_position_search
```

## Trajectory-aware DL training

A new residual CNN starts at zero correction on top of the frozen three-seed ensemble. Its training objective combines body-velocity supervision with accumulated displacement error over five-second chunks. Rotation in the loss uses gyro propagation initialized once from reference, with fixed training-derived sensor calibration. Rotation/reference positions are loss/evaluation information, not network inputs. Each per-time prediction sees only its own past IMU window, even though contiguous predictions share a training chunk.

Three displacement-loss weights (0.25, 1, 4), 30 maximum epochs, patience seven and flight-balanced sampling were tested. Checkpoint selection minimizes validation position RMSE while requiring validation world-velocity error within 2% of the prior. No test values select a trial. None beat validation baseline; the exported correction remains zero and the prior model is retained. This negative result is recorded, not presented as a gain.

```bash
python scripts/train_trajectory.py --output results/new_trajectory_training
python scripts/report_trajectory.py --output results/new_trajectory_training
```

The four new filter tests cover integration equivalence at zero gain, tilt-correction sign, future independence, and constant-velocity preservation/nonzero-scale constraints. Two trajectory-learning tests cover the exact zero-residual fallback without labels and finite gradients through displacement integration. Total suite: 20 passing tests.

## What to claim and what to do next

The velocity model's earlier 12.3% matched-sample body-velocity gain remains valid as a development comparison. A position-RMSE reduction can be quoted only with the velocity-damping tradeoff and reused-data caveat. Do not claim reliable, generally improved drift reduction from these results. Keep global position RMSE, local translation error, world/body velocity error, motion amplitude and initialization in reports.

Reference pose still initializes each continuous segment once. Test recordings and validation sequences have been used repeatedly, so another round cannot supply pristine confirmation. More varied reserved flight data and an uncertainty-aware motion/state estimator are stronger next steps than repeatedly searching these same sequences. TLIO demonstrates the research direction of learned displacement/uncertainty fused with a state estimator, but its full method has not been reproduced here: https://arxiv.org/abs/2007.01867 .
