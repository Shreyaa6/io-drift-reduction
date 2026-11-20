# Method and limitations

## Task

Predict body-frame velocity from a past-only raw IMU window using a compact temporal CNN. Compare zero velocity, training mean body velocity, and Ridge over window mean/std/latest features. Model normalization is fitted only on training windows. Early stopping and method selection minimize equal-sequence mean axis RMSE on validation. The trainer never loads test windows.

The network has three temporal convolution layers followed by global mean/latest feature pooling and an MLP output head. Convolution padding is only within a fully observed historical window. It does not expose samples after the prediction endpoint. Training uses AdamW, normalized MSE, gradient clipping and dropout.

## Velocity metrics

For each axis, RMSE = sqrt(mean((prediction-reference)^2)); report the arithmetic average of three axis RMSEs. First compute per-sequence scores, then average sequences equally for the primary macro result. Also retain per-axis MAE and RMSE. This differs from pooled vector error. Test samples are identical for all methods.

## Trajectory evaluation: explicitly initialized, not unaided startup

Each continuous segment starts at the first full-window endpoint. Both raw integration and learned navigation receive reference position, velocity and orientation **once at this endpoint**. Zero reference biases are not read; raw gyro and accelerometer measurements remain uncorrected. The learned velocity itself does not use initial velocity, although it is supplied to the common integration routine for the classical baseline. Position is anchored once, with no subsequent alignment, loop closure, GPS, reference attitude, or resets within a continuous segment.

Gyro midpoint integration propagates body-to-world orientation. Classical navigation rotates specific force, adds [0,0,-9.81] gravity, and integrates acceleration. Learned body velocity is rotated with the same propagated orientation, then trapezoidally integrated at prediction times. This deliberately simple orientation propagation can accumulate severe bias error. Learned translation does not automatically fix orientation drift. Full startup orientation, heading, and initial velocity are not universally observable from arbitrary IMU-only motion.

Metrics: position RMSE in the initialized frame (no fitted SE3/Sim3 alignment), endpoint position error, endpoint drift as a percentage of reference path length, and translation-only 1-second relative error. The latter is computed at the first endpoint at least one second later; it is not full SE(3) pose RPE. No rotation accuracy or real-time hardware claim is made. Baselines also include constant initial reference velocity. Gaps create independent initialized segments; report each rather than joining trajectories across missing data.

## Accuracy claims

Report the frozen validation-selected method and CNN even when the CNN loses. Do not claim improvement from a different target, different dataset, or cherry-picked sequence. Any comparison against raw integration must state that the baseline is uncalibrated and has no ongoing bias correction; it is a weak physical baseline, not a state-of-the-art estimator. Improvement over it does not establish superiority over AirIO or production IO.

One seed and two held-out sequences are an initial experiment, not statistical proof. Further model comparisons should use validation or a new evaluation protocol. Final external confirmation needs new untouched flight trajectories and documented initialization.
