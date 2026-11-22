# Results

`first_run/` stores the initial trained CNN, exact config/provenance, validation learning curve, test metrics, plots, and limitations. These results solve a different task from the previous PX4–GPS residual project and must not be compared numerically as an accuracy gain over that project.

Further runs should use separate output directories and explicitly report whether the test data has already influenced development.

`optimized/` stores the bounded architecture/context search, all ten fitted checkpoints and validation histories, selected ensemble, weight audit, calibrated-navigation ablations, matched-sample evaluation, figures and report. The same evaluation sequences were previously inspected; results are development comparisons.

`position_checked/` retains a validation-selected position filter and an explicit no-damping comparison. Its lower global position error comes with substantial motion underestimation. `trajectory_training/` retains three displacement-loss DL trials; unchanged baseline won validation. Neither experiment overwrites the optimized ensemble.
