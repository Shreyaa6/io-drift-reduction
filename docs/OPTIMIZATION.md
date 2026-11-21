# Bounded deep-learning optimization protocol

## What trained weights mean

Weights are optimized by backpropagation and AdamW; hyperparameters specify how training proceeds. Manually changing saved weights without evidence is not a route to accuracy. The original checkpoint has 40,899 parameter elements; reproducing its seeded initialization confirms all 40,899 differ in the saved trained model (L2 change 7.7567). The optimization pipeline additionally records each trial's initial weight norm, L2 update norm, finite-weight check, gradient norms, and epoch-by-epoch validation errors.

No finite experiment can certify “everything possible” or a global best. This protocol compares a predeclared, reproducible set of practical options and records failures as well as wins.

## Search plan

`configs/optimization.json` declares eight model/context/loss/augmentation configurations before running them:

- Original-sized CNN with improved training on one-second context.
- Residual CNN on one-, two-, and four-second contexts.
- Dilated residual temporal network on four seconds.
- Convolutional encoder plus two-layer GRU and attention on two seconds.
- Huber-loss variant on two seconds.
- Huber loss plus physically consistent random sensor xy rotations on four seconds.

All histories contain 100 sampled observations but cover different durations; this bounds computational cost. Raw elapsed-time values retain the underlying IMU cadence. Context length and sample spacing are recorded in the checkpoint. The models see no future sample beyond the prediction endpoint. Convolution padding and GRU attention operate within that already observed history.

Every trial uses flight-balanced sampling, training-only input/target normalization, AdamW, gradient clipping, dropout, validation-driven learning-rate reduction, early stopping, and a moving average of weights. Validation chooses raw versus averaged weights. Noise augmentation adds modest raw-sensor noise and fixed-in-window bias to training inputs only. Rotation augmentation rotates gyro, accelerometer and target body velocity consistently; it never rotates only the input or leaves its label inconsistent.

Choose the best architecture/context on validation, repeat it with seeds 17 and 123, and compare single seeds with their equal-weight ensemble. A small predeclared shrinkage grid (0.5/0.75/1 toward training mean) is also selected on validation. The original saved CNN remains a candidate so an unsuccessful search does not force a worse replacement. The final checkpoint is frozen and hashed before evaluating the reused test sequences.

## Fair comparison and leakage boundaries

Different history lengths otherwise discard different startup samples. All trials and baselines therefore predict at identical endpoints that have at least four seconds of continuous history. The original CNN is reevaluated at these same endpoints. Do not compare the new numbers directly with its earlier all-endpoint score (0.4157 m/s).

The search script loads only training/validation sequences, never test recordings or labels. The final evaluator loads test after `selection.json` and `best.pt` exist, verifies the checkpoint hash, and records that this test set was already inspected in the first run. Its results are development comparisons, not fresh external confirmation. Two repeatedly used validation sequences can also be overfit by extensive search; larger searches are not automatically more trustworthy.

## Position tracking and calibration

A network that improves body velocity can still worsen position after integration. The evaluator reports velocity and trajectory errors separately. It includes both raw gyro orientation propagation and a fixed training-derived gyro/accelerometer bias correction. The correction is estimated from reference kinematics in training flights only, using robust medians, and applied unchanged at evaluation. It does not read test reference biases or ongoing reference attitude. It may not transfer across devices, temperature, or recording sessions.

For each navigation segment, reference p/q and classical initial v are supplied once, as in the first run. This remains reference-initialized IO, not unaided startup. Compare original and optimized networks with the same orientation method to isolate model changes; compare raw versus calibrated propagation to isolate calibration changes. No result should combine these effects into an unexplained “DL improvement.”

## Reproduction

```bash
source .venv/bin/activate
python -m unittest discover -s tests -v
python scripts/optimize.py --output results/another_search
python scripts/evaluate_optimized.py --output results/another_search
python scripts/report_optimized.py --output results/another_search
python scripts/predict.py data/euroc/MH_05_difficult/mav0/imu0/data.csv \
  --checkpoint results/another_search/best.pt --output results/body_velocity.csv
```

Completed trial files are reused when resuming the same unfinished plan. A completed search directory cannot be silently overwritten. Use a new output directory for another search. Training uses Apple MPS acceleration when available (CPU otherwise); each trial records its device. A completed CPU trial can be resumed alongside subsequent accelerated trials. This is a bounded eight-configuration search plus two seed repetitions, not an exhaustive hyperparameter sweep or a reproduction of a published state-of-the-art system.

## Method references

- Temporal convolution motivation: Bai, Kolter and Koltun, *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling*, https://arxiv.org/abs/1803.01271 . This implementation is independent and not an exact reproduction.
- PyTorch learning-rate scheduling: https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html .
- Drone IO context and body-frame representation: https://air-io.github.io/ . Its reported improvements are not this project's results, and no ongoing ground-truth attitude is passed to our network.
