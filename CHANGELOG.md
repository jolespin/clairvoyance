# Change  Log
* [2023.10.12] - Replaced `np.mean` with `np.nanmean` to handle `nan` values in scores (e.g., `precision_score`)
* [2023.10.10] - Made `method="asymmetric"` the new default
* [2023.6.9] - Added `plot_scores_comparison` and `get_balanced_class_subset` for evaluating testing datasets.
* [2023.5.25] - Added `X_testing` and `y_testing` to `recursive_feature_elimination` functions/methods.