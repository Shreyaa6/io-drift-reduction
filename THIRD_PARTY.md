# Data and software acknowledgements

EuRoC MAV data: Burri et al., *The EuRoC micro aerial vehicle datasets*, International Journal of Robotics Research, 2016. https://doi.org/10.1177/0278364915620033 . Original data and terms: https://doi.org/10.3929/ethz-b-000690084 . Download instructions: https://projects.asl.ethz.ch/datasets/euroc-mav/ . The project does not claim ownership of these recordings.

The downloaded IMU/reference archive is packaged by the AirIO authors at https://github.com/Air-IO/Air-IO/releases/tag/datasets . AirIO: Qiu et al., *AirIO: Learning Inertial Odometry with Enhanced IMU Feature Observability*, IEEE RA-L, 2025. https://doi.org/10.1109/LRA.2025.3581130 . No AirIO source code or pretrained weights have been copied into this project. Dataset terms are separate from the AirIO code license.

PyTorch, NumPy, SciPy, pandas, scikit-learn and Matplotlib are installed as dependencies, retain their respective licenses, and are not vendored. The checked-in CNN checkpoint was trained by this project's pipeline on the referenced data. Raw data and prepared windows are excluded from Git.
