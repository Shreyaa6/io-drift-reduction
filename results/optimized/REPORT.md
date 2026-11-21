# Deep-learning optimization results

## Scope and headline

Eight predeclared architecture/context/loss/augmentation configurations were trained, followed by two additional seeds for the validation-winning configuration. The selected predictor is **seed_ensemble_shrink_1.0**, based on **resnet_1s**. Every network consumes only gyro/accelerometer/dt history. No images, GPS, or ongoing reference orientation enter the velocity network.

On common validation endpoints, selected RMSE is **0.3769 m/s**, versus **0.4240 m/s** for the original CNN (11.11% reduction).

On the already-inspected evaluation sequences, selected RMSE is **0.3641 m/s**, versus **0.4153 m/s** for the original CNN at the **same timestamps** (12.33% reduction; a negative reduction means worse). Do not compare against the old 0.4157 score, which includes different startup samples.

| method | rmse_mean | mae_mean |
| --- | --- | --- |
| Optimized DL | 0.3641 | 0.2781 |
| Original CNN | 0.4153 | 0.3207 |
| Ridge | 0.4635 | 0.3610 |
| Training mean | 0.5327 | 0.4067 |
| Zero | 0.5458 | 0.4140 |

## All completed training runs

| id | seed | best_epoch | best_weights | validation_macro_rmse | weight_change_l2 | device |
| --- | --- | --- | --- | --- | --- | --- |
| resnet_1s_seed17 | 17 | 11 | ema | 0.3894 | 6.3178 | mps |
| resnet_1s | 42 | 7 | raw | 0.3936 | 5.3511 | mps |
| resnet_1s_seed123 | 123 | 18 | raw | 0.4018 | 6.9012 | mps |
| cnn_1s | 42 | 20 | raw | 0.4258 | 6.9416 | cpu |
| resnet_2s_huber | 42 | 2 | raw | 0.4262 | 3.3674 | mps |
| resnet_4s | 42 | 8 | ema | 0.4302 | 5.4617 | mps |
| gru_2s | 42 | 3 | ema | 0.4315 | 6.2991 | mps |
| tcn_4s | 42 | 3 | raw | 0.4348 | 4.1568 | mps |
| resnet_2s | 42 | 2 | raw | 0.4352 | 3.2562 | mps |
| resnet_4s_rotation | 42 | 17 | ema | 0.4769 | 10.0295 | mps |

Each weight-change norm compares the trained checkpoint with that run's own initial weights. Original weight-audit details are stored separately. Training and validation curves, gradient norms, learning-rate history, chosen raw/EMA checkpoint, and finite-weight checks are retained in each trial directory. A completed CPU trial was retained while subsequent models used available MPS acceleration; an unfinished CPU residual run was restarted for acceleration and was not scored as a completed trial.

## Position tracking must be reported separately

| sequence | method | position_rmse_m | final_error_m |
| --- | --- | --- | --- |
| MH_05_difficult | Optimized DL (raw) | 12.5896 | 12.0462 |
| MH_05_difficult | Original CNN (raw) | 14.8198 | 20.4911 |
| MH_05_difficult | Zero (raw) | 8.1709 | 0.3300 |
| MH_05_difficult | Optimized DL (training-calibrated) | 7.9809 | 12.1812 |
| MH_05_difficult | Original CNN (training-calibrated) | 12.0032 | 15.8364 |
| MH_05_difficult | Zero (training-calibrated) | 8.1709 | 0.3300 |
| V2_03_difficult | Optimized DL (raw) | 3.7310 | 6.0361 |
| V2_03_difficult | Original CNN (raw) | 7.3454 | 8.4215 |
| V2_03_difficult | Zero (raw) | 2.1815 | 2.2243 |
| V2_03_difficult | Optimized DL (training-calibrated) | 2.3767 | 4.8024 |
| V2_03_difficult | Original CNN (training-calibrated) | 5.7620 | 7.6542 |
| V2_03_difficult | Zero (training-calibrated) | 2.1815 | 2.2243 |

Fixed gyro/accelerometer biases were fitted from training reference kinematics only, then applied unchanged. Both raw and training-calibrated orientation propagation are shown. Compare the same orientation mode when attributing improvement to DL. Each segment still receives reference position/orientation and classical initial velocity once. There is no ongoing reference correction or fitted alignment. These results do not establish unaided startup or reliable navigation; inspect all position errors and stationary baselines before making a drift-reduction claim.

## What was and was not optimized

Tested: longer temporal context, residual and dilated CNNs, a GRU with attention, balanced flight sampling, sensor-noise augmentation, coordinate-consistent rotation augmentation, MSE/Huber loss, dropout, AdamW, gradient clipping, learning-rate reduction, early stopping, averaged weights, multiple seeds, ensembling and validation-selected shrinkage. Every candidate and failed improvement remains visible in the records.

This is not an exhaustive search or a claim of the best possible neural network. Training data, motion observability, initialization, sensor calibration, and domain diversity impose limits that model size alone cannot solve. Adding every available DL technique can increase overfitting and compute without improving unseen-flight performance.

## Evaluation status

**Previously inspected in first run. Subsequent comparisons are development-set reuse, not fresh confirmation.** All architecture, checkpoint, ensemble, and shrinkage selection used validation only, with selection saved and checkpoint hashed before this evaluation. These reused public sequences cannot provide fresh independent confirmation. Repeated validation searches also increase overfitting risk. Reserve new flight trajectories or another audited dataset for final confirmation; do not select further changes using these evaluation scores.

Reproduction: see `docs/OPTIMIZATION.md`, `search_plan.json`, `selection.json`, `leaderboard.csv`, and `evaluation_protocol.json`. The original first run remains unchanged.
