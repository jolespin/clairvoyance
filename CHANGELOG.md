#### Change  Log
* [2025.7.26] - Updated `training_testing_weights`
* [2025.2.21] - Changed `verbose` behavior for `BayesianClairvoyanceBase` to supress logs.  Also in `ClairvoyanceRecursiveFeatureAddition` and `ClairvoyanceRecursiveFeatureElimination` to supress progressbar for `verbose=0`
* [2025.2.21] - Replaced `StratifiedKFold` with `StratifiedGroupKFold` in `format_cross_validation`
* [2025.2.19] - Added support for `check_parameter_space` and `compile_parameter_space` to handle single point values and types.  Allows `check_parameter_space` to not need estimator
* [2025.2.19] - Added `pyexeggutor` support
* [2025.2.4] - Moved `_check_param_space` and `_compile_param_space` to `.utils` and renamed `check_parameter_space` and `compile_parameter_space`, respectively.
* [2024.8.13] - Made `matplotlib`, `seaborn`, and `xarray` optional as they are currently only used in `clairvoyance.legacy`
* [2024.7.8] - Added `memory_profiler` to `.fit` and `.fit_transform` in `clairvoyance.bayesian` classes
* [2024.7.6] - Using `get_feature_importances` from `Feature-Engine` instead of `format_weights`.
* [2024.7.6] - Demoted `ClairvoyanceBase`, `ClairvoyanceRegression`, `ClairvoyanceClassification`, and `ClairvoyanceRecursive` to `clairvoyance.legacy.`
* [2024.7.6] - Added `BayesianClairvoyanceBase`, `BayesianClairvoyanceClassification`, and `BayesianClairvoyanceRegression` in `clairvoyance.bayesian. which use `Optuna` for hyperparameter tuning and  `ClairvoyanceRecursiveFeatureAddition` or `ClairvoyanceRecursiveElimination` for feature selection.
* [2024.7.6] - Added `ClairvoyanceRecursiveFeatureAddition` and `ClairvoyanceRecursiveElimination` in `clairvoyance.feature_selection` which are mods of `Feature-Engine` classes that can handle transformations during feature selection along with some other conveniences. 
* [2024.6.14] - Changed `recursive_feature_inclusion` to `recursive_feature_addition` to be consistent with common usage.
* [2023.12.4] - Added support for models that do not converge or `nan` values in resulting scores.
* [2023.10.12] - Replaced `np.mean` with `np.nanmean` to handle `nan` values in scores (e.g., `precision_score`)
* [2023.10.10] - Made `method="asymmetric"` the new default
* [2023.6.9] - Added `plot_scores_comparison` and `get_balanced_class_subset` for evaluating testing datasets.
* [2023.5.25] - Added `X_testing` and `y_testing` to `recursive_feature_elimination` functions/methods.

#### Pending:
* Add `n_cv_jobs` to all relevant classes
* Test using `sktime`: 
    * [TimeSeriesForestClassifier](https://www.sktime.net/en/stable/api_reference/auto_generated/sktime.classification.interval_based.TimeSeriesForestClassifier.html)
    * [TimeSeriesForestRegressor](https://www.sktime.net/en/stable/api_reference/auto_generated/sktime.regression.interval_based.TimeSeriesForestRegressor.html)
    
#### Future:
* Use SHAP?
