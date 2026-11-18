# Dataset decision and protocol

## Selected: EuRoC MAV, IMU/reference subset

No dataset is perfect. EuRoC is a suitable first benchmark for this scope because it records real micro-aerial-vehicle motion, gyroscope/specific-force IMU at 200 Hz, calibration and reference state. Images are available in the full original dataset but are not downloaded or used in this IO project. There are no GPS model inputs or GPS labels in this pipeline.

- Original dataset: https://projects.asl.ethz.ch/datasets/euroc-mav/
- Original dataset record: https://doi.org/10.3929/ethz-b-000690084
- Paper: Burri et al., *The EuRoC micro aerial vehicle datasets*, IJRR, 2016, https://doi.org/10.1177/0278364915620033
- Downloaded subset maintained by AirIO: https://github.com/Air-IO/Air-IO/releases/download/datasets/EuRoC-Dataset.zip
- AirIO preparation documentation: https://github.com/Air-IO/Air-IO
- SHA-256: `de09fdd03e9147461fd678aa6763454bdf8d1f07172753a9ec80073e9be38e4a`

This repository uses an independently implemented small CNN, not the AirIO architecture, pretrained weights, or published performance. AirIO is credited for packaging the source data. Dataset rights and terms belong to the original owners; raw data and prepared windows are not committed or redistributed here. See the original dataset record before reuse. The archive contains calibration files but does not provide the full original images.

## Why not reuse the previous CSV?

It has no reference position or orientation and only four short logs. Its labels are PX4-minus-GPS residuals, which are a different prediction task from IO. No claimed percentage improvement over its published RMSE can be inferred from this dataset's scores.

## Fixed splits

Training: MH_01_easy, MH_02_easy, MH_03_medium, V1_02_medium, V2_01_easy, V2_02_medium.

Validation: MH_04_difficult, V1_03_difficult.

Held-out test: MH_05_difficult, V2_03_difficult.

These are project-specific splits, not a claimed official EuRoC learning split. Held-out sequences include difficult motion and share environments with training; this is not an unseen-environment test. EuRoC V1_01_easy is excluded because orientation-reference issues are documented by AirIO/OpenVINS. Changing splits after seeing test scores invalidates the held-out interpretation.

## Input and label definitions

Input: body-frame angular velocity xyz (rad/s), specific force xyz (m/s²), raw elapsed sample time (s). No absolute timestamps, IDs, position, velocity, quaternion, bias reference, or previous target enters the model.

Label: reference world velocity rotated into the current IMU body frame using reference orientation. Reference orientation is used for label construction only; it is not an ongoing inference feature. Quaternion CSV order is wxyz; SciPy order is xyzw. World gravity is [0,0,-9.81] m/s² for the navigation baseline.

Timestamps are parsed as int64 nanoseconds and reduced by an integer origin before conversion to seconds. Labels are linearly interpolated for position/velocity and interpolated with SLERP for rotations, strictly within reference coverage. Reference gaps >25 ms are excluded unless the timestamp is an exact reference sample. Windows reset across IMU gaps >25 ms. A full 200-row history (~1 s) is required; every second row is supplied to the network, ending at the current sample. Predictions occur every 20 raw rows (~10 Hz). The cadence is a design choice, not a claim of high-rate real-time navigation.

## Reference limitations

The source reference uses motion capture/laser tracking and aligned/estimated states; it is not exact physical truth. EuRoC documents reference synchronization and accuracy limitations. Interpolation labels can use reference samples around a timestamp; this is offline supervision, not an online sensor input. The dataset has limited diversity and this is a single-seed first experiment.

## Next dataset

Blackbird UAV is a suitable candidate to broaden drone IO motion: https://github.com/mit-aera/Blackbird-Dataset . Its IMU is real and camera imagery rendered; cameras are irrelevant to the IO-only experiment. A Blackbird adapter is not implemented here. Audit its sampling, units, sensor frame, initialization and split independence before mixing it with EuRoC.
