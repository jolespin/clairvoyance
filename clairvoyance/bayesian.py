# Built-ins
import os, sys, copy, warnings
from collections import OrderedDict
from collections.abc import Mapping

# from multiprocessing import cpu_count

# PyData
import pandas as pd
import numpy as np
# import xarray as xr

# Machine learning
from scipy import stats
from sklearn.metrics import get_scorer #, make_scorer
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.base import clone, is_classifier, is_regressor

# Feature selection
from feature_engine.selection import DropConstantFeatures, DropDuplicateFeatures # , RecursiveFeatureElimination, RecursiveFeatureAddition

# Hyperparameter selection
import optuna

from .utils import *
from .transformations import (
    closure, 
    clr, 
    clr_with_multiplicative_replacement,
)
from .feature_selection import (
    ClairvoyanceRecursiveFeatureAddition, 
    ClairvoyanceRecursiveFeatureElimination,
)

# Memory profiling
from memory_profiler import profile

_bayesianclairvoyancebase_docstring = """
        # Modeling parameters:
        # ====================
        estimator: 
                sklearn-compatible estimator with either .feature_importances_ or .coef_
            
        param_space:

                dict with {name_param: [suggestion_type, *]}
        
                suggestion_types:  {"categorical", "discrete_uniform", "float", "int", "loguniform", "uniform"}
                
                categorical suggestion types must contain 2 items (e.g., [categorical, ['a','b','c']])
                uniform/loguniform suggestion types must contain 3 items [uniform/loguniform, low, high]
                float/int suggestion type must contain either 3 items [float/int, low, high]) or 
                4 items [float/int, low, high, {step:float/int, log:bool}]

        
        scorer:
                sklearn-compatible scorer or string with scorer name that can be used with `sklearn.metrics.get_scorer`
                
        n_iter [Default=10]:
                Number of iterations to run Optuna + feature selection
                
        n_trials [Default=50]:
                Number of trials for Optuna for each iteration
                
        transformation [Default=None]:
                Transformation to use for each time a feature is added or removed. 
                This is useful for compositional data analysis where the feature transformations
                are dependent on the other features.
                None: No transformation
                closure: Total sum-scaling where the values are normalized by the total counts
                clr: CLR transformation.  Cannot handle zeros so may require pseudocounts
                clr_with_multiplicative_replacement: Closure followed by a pseudocount 1/m**2 where m = number of features

        # Optuna
        # ======
        study_prefix [Default: "n_iter="]:
                Prefix to use for Optuna iterations
                
        study_timeout [Default=None]:
                Stop study after the given number of second(s). If this argument is set to None, the study is executed without time limitation. 
                If n_trials is also set to None, the study continues to create trials until it receives a termination signal such as Ctrl+C or SIGTERM.
                
        study_callbacks [Default=None]:
                List of callback functions that are invoked at the end of each trial. 
                Each function must accept two parameters with the following types in this order: Study and FrozenTrial.
        
        training_testing_weights [Default = [1.0,1.0]]:
                Training and testing multipliers to use in multi-objective optimization

        # Feature selection
        # =================
        feature_selection_method:str [Default = "addition"]:
                Feature selection method to use.  
                addition or elimination

        
        feature_selection_performance_threshold [Default = 0.0]:
                Minimum performance gain to consider a feature
                
        drop_constant_features [Default = True]:
                Use FeatureEngine to drop constant features in cross-validation splits
                
        threshold_constant_features [Default = 1]:
                Threshold to use for constant features
                
        drop_duplicate_features [Default = True]:
                Use FeatureEngine to drop duplicate features in cross-validation splits

        feature_importances_from_cv [Default = True]:
                If True, calculate feature importances from cross-validation splits
                If False, use a final fit for the feature importances

        # Zero weights
        # ============
        remove_zero_weighted_features [Default = True]:
                When changing the feature set on tree-based estimators it is common for certain features
                to have a zero-weight after refitting the model.  This refits iteratively refits the
                estimator until none of the features have zero weights.  It is possible that this will
                break poorly performing estimators and return zero features. 
                
        maximum_tries_to_remove_zero_weighted_features [Default = 100]:
                Maximum number of tries to remove zero weighted features during refits

        # Labeling
        # ========
        name:str=None,
        observation_type:str=None,
        feature_type:str=None,
        target_type:str=None,

        # Utility
        # =======
        early_stopping [Default = 5]:
                If the model does not improve in 5 iterations then stop.
                
        random_state [Default = 0]:
                Random seed state used for TPESampler, cross-validation[, and estimator if one is not already set.]
                
        n_jobs [Default = 1]:
                Number of processors to use.  -1 uses all available. 
        
        copy_X=True,
        copy_y=True,
        verbose=1,
        log=sys.stdout,
"""

class BayesianClairvoyanceBase(object):
    
    def __init__(
        self,
        # Modeling
        estimator,
        param_space:dict,
        scorer,
        n_iter=10,
        n_trials=50,
        transformation=None,

        # Optuna
        study_prefix="n_iter=",
        study_timeout=None,
        study_callbacks=None,
        training_testing_weights = [1.0,1.0],

        # Feature selection
        feature_selection_method:str="addition",
        feature_selection_performance_threshold=0.0, 
        drop_constant_features = True,
        threshold_constant_features=1,
        drop_duplicate_features = True,
        feature_importances_from_cv=True,        

        # Zero weights
        remove_zero_weighted_features=True,
        maximum_tries_to_remove_zero_weighted_features=100,

        # Labeling
        name=None,
        observation_type=None,
        feature_type=None,
        target_type=None,

        # Utility
        early_stopping=5,
        random_state=0,
        n_jobs=1,
        copy_X=True,
        copy_y=True,
        verbose=1,
        log=sys.stdout,
         ):         

        _bayesianclairvoyancebase_docstring
        
        # Method
        assert_acceptable_arguments(feature_selection_method, {"addition", "elimination"})
        self.feature_selection_method = {"addition":ClairvoyanceRecursiveFeatureAddition, "elimination":ClairvoyanceRecursiveFeatureElimination}[feature_selection_method]
            
        # Estimator
        self.estimator_name = estimator.__class__.__name__
        if is_classifier(estimator):
            self.estimator_type = "classifier"
        if is_regressor(estimator):
            self.estimator_type = "regressor"
        self.estimator = clone(estimator)

        if "random_state" in self.estimator.get_params(deep=True):
            query = self.estimator.get_params(deep=True)["random_state"]
            if query is None:
                if verbose > 0:
                    print("Updating `random_state=None` in `estimator` to `random_state={}`".format(random_state), file=log)
                self.estimator.set_params(random_state=random_state)
        self.feature_weight_attribute = get_feature_importance_attribute(estimator, "auto")
        assert len(param_space) > 0, "`param_space` must have at least 1 key:[value_1, value_2, ..., value_n] pair"
        self.param_space = self._check_param_space(estimator, param_space)

        # Training testing weights
        training_testing_weights = np.asarray(training_testing_weights).astype(float)
        msg = "`training_testing_weights` must be a float vector with values in the range [0,1]"
        assert training_testing_weights.size == 2, msg
        assert np.min(training_testing_weights) >= 0, msg
        assert np.max(training_testing_weights) <= 1, msg
        self.training_testing_weights = training_testing_weights
        

        # Set attributes
        self.name = name
        self.observation_type = observation_type
        self.feature_type = feature_type
        self.target_type = target_type
        self.is_fitted = False
        self.testing_set_provided = False
        self.n_iter = n_iter
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.early_stopping = early_stopping
        self.study_prefix= study_prefix
        self.study_timeout = study_timeout
        self.study_callbacks = study_callbacks

        self.random_state = random_state
        if isinstance(scorer, str):
            scorer = get_scorer(scorer)
        self.scorer = scorer
        self.scorer_name = scorer._score_func.__name__
        self.verbose = verbose
        self.log = log

        # Data
        if isinstance(transformation, str):
            assert_acceptable_arguments(transformation, {"closure", "clr", "clr_with_multiplicative_replacement"})
            transformation = globals()[transformation]

        if transformation is not None:
            assert hasattr(transformation, "__call__"), "If `transform` is not None, then it must be a callable function"

        self.transformation = transformation
        self.copy_X = copy_X
        self.copy_y = copy_y

        # Feature selection
        self.feature_selection_performance_threshold = feature_selection_performance_threshold
        self.feature_importances_from_cv = feature_importances_from_cv
        self.drop_constant_features = drop_constant_features
        self.threshold_constant_features = threshold_constant_features
        self.drop_duplicate_features = drop_duplicate_features
        self.remove_zero_weighted_features=remove_zero_weighted_features
        self.maximum_tries_to_remove_zero_weighted_features=maximum_tries_to_remove_zero_weighted_features
        
    def __repr__(self):
        pad = 4
        header = format_header("{}(Name:{})".format(self.__class__.__name__, self.name),line_character="=")
        n = len(header.split("\n")[0])
        fields = [
            header,
            pad*" " + "* Estimator: {}".format(self.estimator_name),
            pad*" " + "* Estimator Type: {}".format(self.estimator_type),
            pad*" " + "* Parameter Space: {}".format(self.param_space),
            pad*" " + "* Scorer: {}".format(self.scorer_name),

            pad*" " + "- -- --- ----- -------- -------------",
            
            pad*" " + "* n_iter: {}".format(self.n_iter),
            pad*" " + "* n_jobs: {}".format(self.n_jobs),
            pad*" " + "* early_stopping: {}".format(self.early_stopping),

            pad*" " + "* random_state: {}".format(self.random_state),

            pad*" " + "- -- --- ----- -------- -------------",
            pad*" " + "* Feature Selection Method: {}".format(self.feature_selection_method.__name__),
            pad*" " + "* Feature Weight Attribute: {}".format(self.feature_weight_attribute),
            pad*" " + "* Transformation: {}".format(self.transformation),
            pad*" " + "* Remove Zero Weighted Features: {}".format(self.remove_zero_weighted_features),
            pad*" " + "* Maximum Tries to Remove: {}".format(self.maximum_tries_to_remove_zero_weighted_features), 
            
            pad*" " + "- -- --- ----- -------- -------------",
            
            pad*" " + "* Observation Type: {}".format(self.observation_type),
            pad*" " + "* Feature Type: {}".format(self.feature_type),
            pad*" " + "* Target Type: {}".format(self.target_type),
 
            pad*" " + "- -- --- ----- -------- -------------",

            pad*" " + "* Fitted: {}".format(self.is_fitted),
            # pad*" " + "* Fitted(RCI): {}".format(self.is_fitted_rci),
            
            
            ]

        return "\n".join(fields)

    def _check_param_space(self, estimator, param_space):
        """
        estimator: A sklearn-compatible estimator
        param_space: dict with {name_param: [suggestion_type, *]}
        
        suggestion_types:  {"categorical", "discrete_uniform", "float", "int", "loguniform", "uniform"}
        
        categorical suggestion types must contain 2 items (e.g., [categorical, ['a','b','c']])
        uniform/loguniform suggestion types must contain 3 items [uniform/loguniform, low, high]
        float/int suggestion type must contain either 3 items [float/int, low, high]) or 4 items [float/int, low, high, {step:float/int, log:bool}]
        """
        # suggest_categorical()
        # Suggest a value for the categorical parameter.
        # suggest_discrete_uniform(name, low, high, q)
            # Suggest a value for the discrete parameter.
        # suggest_float(name, low, high, *[, step, log])
            # Suggest a value for the floating point parameter.
        # suggest_int(name, low, high, *[, step, log])
            # Suggest a value for the integer parameter.
        # suggest_loguniform(name, low, high)
            # Suggest a value for the continuous parameter.
        # suggest_uniform(name, low, high)
            # Suggest a value for the continuous parameter.
        
        # Check if parameter names are valid
        param_space = copy.deepcopy(param_space)
        estimator_params = set(estimator.get_params(deep=True).keys())
        query_params = set(param_space.keys())
        assert query_params <= estimator_params, "The following parameters are not recognized for estimator {}:\n{}".format(estimator.__class__.__name__, "\n".join(sorted(query_params - estimator_params)))

        suggestion_types = {"categorical", "discrete_uniform", "float", "int", "loguniform", "uniform"}
        for k, v in param_space.items():
            assert hasattr(v, "__iter__") & (not isinstance(v, str)), "space must be iterable"
            assert len(v) > 1, "space must use the following format: [suggestion_type, *values] (e.g., [categorical, ['a','b','c']]\n[int, 1, 100])"
            query_suggestion_type = v[0]
            assert_acceptable_arguments(query_suggestion_type, suggestion_types)
            if query_suggestion_type in {"categorical"}:
                assert len(v) == 2, "categorical suggestion types must contain 2 items (e.g., [categorical, ['a','b','c']])"
                assert hasattr(v[1], "__iter__") & (not isinstance(v[1], str)), "categorical suggestion types must contain 2 items [categorical, ['a','b','c']]"
            if query_suggestion_type in {"uniform", "loguniform"}:
                assert len(v) == 3, "uniform/loguniform suggestion types must contain 3 items [uniform/loguniform, low, high]"
            if query_suggestion_type in {"discrete_uniform"}:
                assert len(v) == 4, "discrete_uniform suggestion type must contain 4 items [discrete_uniform, low, high, q]"
            if query_suggestion_type in {"float", "int"}:
                suggest_float_int_error_message = "float/int suggestion type must contain either 3 items [float/int, low, high]) or 4 items [float/int, low, high, {step:float/int, log:bool}]"
                assert len(v) in {3,4}, suggest_float_int_error_message
                if len(v) == 3:
                    param_space[k] = [*v, {}]
                if len(v) == 4:
                    query_dict = v[-1]
                    assert isinstance(query_dict, Mapping), suggest_float_int_error_message
                    query_dict = dict(query_dict)
                    assert set(query_dict.keys()) <= {"step", "log"}, suggest_float_int_error_message
                    if "step" in query_dict:
                        assert isinstance(query_dict["step"], (float, int)), suggest_float_int_error_message
                    if "log" in query_dict:
                        assert isinstance(query_dict["log"], bool), suggest_float_int_error_message
                    param_space[k] = [*v[:-1], query_dict]
        return param_space

    def _compile_param_space(self, trial, param_space):
        params = dict()
        for k, v in param_space.items():
            suggestion_type = v[0]
            suggest = getattr(trial, f"suggest_{suggestion_type}")
            if suggestion_type in {"float", "int"}:
                suggestion = suggest(k, *v[1:-1], **v[-1])
            else:
                suggestion = suggest(k, *v[1:])
            params[k] = suggestion
        return params

    def _optimize_hyperparameters(self, X, y, study_name, sampler, **study_kws): # test set here?
        def _objective(trial):

            # Compile parameters
            params = self._compile_param_space(trial, self.param_space)

            estimator = clone(self.estimator)
            estimator.set_params(**params)

            cv_results = cross_val_score(estimator, X, y, scoring=self.scorer, n_jobs=self.n_jobs, cv=self.cv_splits_)

            return cv_results.mean()
            
        direction = "maximize"
        study = optuna.create_study(direction=direction, study_name=study_name, sampler=sampler, **study_kws)
        study.optimize(_objective, n_trials=self.n_trials, timeout=self.study_timeout, show_progress_bar=self.verbose >= 2, callbacks=self.study_callbacks, gc_after_trial=True)
        return study
        
    def _optimize_hyperparameters_include_testing(self, X, y,  X_testing, y_testing, study_name, sampler, **study_kws): # test set here?
        def _objective(trial):

            # Compile parameters
            params = self._compile_param_space(trial, self.param_space)
            estimator = clone(self.estimator)
            estimator.set_params(**params)
            estimator.fit(X, y)
            testing_score = self.scorer(estimator, X_testing, y_testing)
            
            cv_results = cross_val_score(estimator, X, y, scoring=self.scorer, n_jobs=self.n_jobs, cv=self.cv_splits_)

            return [cv_results.mean(), testing_score]
        # if direction == "auto":
        #     direction = {"regressor":"minimize", "classifier":"maximize"}[self.estimator_type]
        directions = ["maximize", "maximize"]
        study = optuna.create_study(directions=directions, study_name=study_name, sampler=sampler, **study_kws)
        study.optimize(_objective, n_trials=self.n_trials, timeout=self.study_timeout, show_progress_bar=self.verbose >= 2, callbacks=self.study_callbacks, gc_after_trial=True)
        return study

    def _feature_selection(self, estimator, X, y, X_testing, y_testing, study_name):

            # initial_testing_score = np.nan
            # if self.testing_set_provided:
            #     estimator.fit(X, y)
            #     initial_testing_score = self.scorer(estimator, X_testing, y_testing)
                
            # Feature selection
            model_fs = self.feature_selection_method(
                estimator=estimator, 
                scoring=self.scorer, 
                cv=self.cv_splits_, 
                threshold=self.feature_selection_performance_threshold,
                feature_importances_from_cv=self.feature_importances_from_cv,
                transformation=self.transformation,
                remove_zero_weighted_features=self.remove_zero_weighted_features,
                maximum_tries_to_remove_zero_weighted_features=self.maximum_tries_to_remove_zero_weighted_features,
                verbose=self.verbose,
            )
            model_fs.fit(X, y)

            selected_features = model_fs.selected_features_
            feature_selected_training_cv = model_fs.feature_selected_model_cv_
            feature_selected_testing_score = np.nan
            if self.testing_set_provided:
                X_training_query = X.loc[:,selected_features]
                X_testing_query = X_testing.loc[:,selected_features]
                if self.transformation is not None:
                    X_training_query = self.transformation(X_training_query)
                    X_testing_query = self.transformation(X_testing_query)

                estimator.fit(X_training_query, y)
                feature_selected_testing_score = self.scorer(estimator, X_testing_query, y_testing)
                
            # Show the feature weights be scaled? Before or after
            return (selected_features, model_fs.initial_feature_importances_, model_fs.feature_selected_importances_, model_fs.performance_drifts_, feature_selected_training_cv, feature_selected_testing_score)
                
    def _fit(self, X, y, cv, X_testing=None, y_testing=None, optimize_with_training_and_testing="auto", **study_kws): # How to use the test set here?
        if self.copy_X:
            self.X_ = X.copy()
        if self.copy_y:
            self.y_ = y.copy()
            
        # Cross validation
        self.cv_splits_, self.cv_labels_ = format_cross_validation(cv, X=X, y=y, random_state=self.random_state, stratify=self.estimator_type == "classifier")
        
        # Testing
        self.testing_set_provided = check_testing_set(X.columns, X_testing, y_testing)
        if optimize_with_training_and_testing == "auto":
            optimize_with_training_and_testing = self.testing_set_provided
        if optimize_with_training_and_testing:
            assert self.testing_set_provided, "If `optimize_with_training_and_testing=True` then X_testing and y_testing must be provided."
            
        self.studies_ = OrderedDict()
        self.results_ = OrderedDict()
        self.feature_weights_ = OrderedDict()
        self.feature_selection_performance_drifts_ = OrderedDict()
        self.feature_weights_initial_ = OrderedDict()
        # self.feature_selection_ = dict()

        if self.drop_constant_features:
            model_dcf = DropConstantFeatures(tol=self.threshold_constant_features)
            features_to_drop = set()
            for cv_labels, (indices_training, indices_testing) in zip(self.cv_labels_, self.cv_splits_):
                model_dcf.fit(X.iloc[indices_training])
                features_to_drop |= set(model_dcf.features_to_drop_)
            if self.verbose > 0:
                n_dropped_features = len(features_to_drop)
                if n_dropped_features > 0:
                    print("Dropping {} constant features based on {} threshold: {}".format(n_dropped_features, self.threshold_constant_features, sorted(features_to_drop)), file=self.log)
            X = X.drop(features_to_drop, axis=1)
            del model_dcf
                                                                       
        if self.drop_duplicate_features:
            model_ddf = DropConstantFeatures(tol=self.threshold_constant_features)
            features_to_drop = set()
            for cv_labels, (indices_training, indices_testing) in zip(self.cv_labels_, self.cv_splits_):
                model_ddf.fit(X.iloc[indices_training])
                features_to_drop |= set(model_ddf.features_to_drop_)
            if self.verbose > 0:
                n_dropped_features = len(features_to_drop)
                if n_dropped_features > 0:
                    print("Dropping {} duplicate features: {}".format(n_dropped_features, sorted(features_to_drop)), file=self.log)
            X = X.drop(features_to_drop, axis=1)
            del model_ddf
            
        query_features = X.columns
        best_score = -np.inf
        no_progress = 0
        for i in range(1, self.n_iter+1):
            if no_progress > self.early_stopping:
                warnings.warn(f"Stopping because score has not improved from {best_score} with {len(query_features)} features in {self.early_stopping} iterations")
                break
            else:
                if len(query_features) > 1:
                    # Study
                    study_name = f"{self.study_prefix}{i}"
                    sampler = optuna.samplers.TPESampler(seed=self.random_state + i)
                    if optimize_with_training_and_testing:
                        self.multiobjective_ = True
                        study = self._optimize_hyperparameters_include_testing(
                            X=X.loc[:,query_features], 
                            y=y, 
                            X_testing=X_testing.loc[:,query_features], 
                            y_testing=y_testing, 
                            study_name=study_name, 
                            sampler=sampler, 
                            **study_kws,
                        )
                    else:
                        self.multiobjective_ = False
                        study = self._optimize_hyperparameters(
                            X=X.loc[:,query_features], 
                            y=y,
                            study_name=study_name, 
                            sampler=sampler, 
                            **study_kws,
                        )
                    self.studies_[study_name] = study

                    # Fit
                    initial_testing_score = np.nan
                    best_estimator = clone(self.estimator)

                    if optimize_with_training_and_testing:
                        # Determine best trial from multiobjective study
                        weighted_scores = list()
                        for trial in study.best_trials:
                            scores = trial.values
                            ws = np.mean(scores * self.training_testing_weights)
                            weighted_scores.append(ws)
                        best_trial = study.best_trials[np.argmax(weighted_scores)]
                        initial_training_score, initial_testing_score = best_trial.values

                        # Refit estimator with best params
                        best_params = best_trial.params
                        best_estimator.set_params(**best_params)
                        best_estimator.fit(X.loc[:,query_features], y)
                    else:
                        best_trial = study.best_trial
                        initial_training_score = study.best_value

                        # Refit estimator with best params
                        best_params = study.best_params
                        best_estimator.set_params(**best_params)
                        best_estimator.fit(X.loc[:,query_features], y)

                        # Get initial testing score
                        if self.testing_set_provided:
                            initial_testing_score = self.scorer(best_estimator, X_testing.loc[:,query_features], y_testing)
                        
                    # # Feature selection
                    selected_features, initial_feature_weights, feature_weights, feature_selection_performance_drifts, feature_selected_training_cv, feature_selected_testing_score = self._feature_selection(
                        estimator=best_estimator, 
                        X=X.loc[:,query_features], 
                        y=y, 
                        X_testing=X_testing.loc[:,query_features] if self.testing_set_provided else None, 
                        y_testing=y_testing, 
                        # query_features=query_features,
                        study_name=study_name, 
                    )
                    feature_selected_training_score = feature_selected_training_cv.mean()
                
                    self.feature_selection_performance_drifts_[study_name] = feature_selection_performance_drifts
                    self.feature_weights_initial_[study_name] = initial_feature_weights
                    self.feature_weights_[study_name] = feature_weights
                    
                    self.results_[study_name] = {
                        "best_hyperparameters":best_params,
                        "best_estimator":best_estimator,
                        "best_trial":best_trial,
                        "number_of_initial_features":len(query_features),
                        "initial_training_score":initial_training_score,
                        "initial_testing_score":initial_testing_score,
                        "number_of_selected_features":len(selected_features),
                        "feature_selected_training_score":feature_selected_training_score,
                        "feature_selected_testing_score":feature_selected_testing_score,
                        "selected_features":list(selected_features),
                    }
                    # print(study.best_value, model_fs.initial_model_performance_, feature_selected_training_performance)
                    print(f"Synopsis[{study_name}] Input Features: {len(query_features)}, Selected Features: {len(selected_features)}", file=self.log)
                    print(f"Initial Training Score: {initial_training_score}, Feature Selected Training Score: {feature_selected_training_score}", file=self.log)
                    if self.testing_set_provided:
                        print(f"Initial Testing Score: {initial_testing_score}, Feature Selected Testing Score: {feature_selected_testing_score}", file=self.log)
                    print(file=self.log)
                    
                    query_features = selected_features
                    if feature_selected_training_score > best_score:
                        best_score = feature_selected_training_score
                    else:
                        no_progress += 1
                else:
                    warnings.warn(f"Stopping because < 2 features remain {query_features}")
                    if len(query_features) == 0:
                        raise ValueError("0 features selected using current parameters.  Try preprocessing feature matrix, using a different transformation, or a different estimator.")
                    break
        self.is_fitted = True
        return self

    def _get_results(self):
        assert self.is_fitted, "Please `.fit` model before `.get_results()`"
        df = pd.DataFrame(self.results_).T
        df.index.name = "study_name"
        return df
        
    @profile
    def fit(self, X, y, cv=3, X_testing=None, y_testing=None, optimize_with_training_and_testing="auto", **study_kws):
        self._fit(
            X=X,
            y=y,
            cv=cv,
            X_testing=X_testing,
            y_testing=y_testing,
            optimize_with_training_and_testing=optimize_with_training_and_testing,
            **study_kws,
        )
        self.results_ = self._get_results()
        return self

    @profile
    def fit_transform(self, X, y, cv=3, X_testing=None, y_testing=None, optimize_with_training_and_testing="auto", **study_kws):
        self._fit(
            X=X,
            y=y,
            cv=cv,
            X_testing=X_testing,
            y_testing=y_testing,
            optimize_with_training_and_testing=optimize_with_training_and_testing,
            **study_kws,
        )
        self.results_ = self._get_results()
        return self.results_
        

    def to_file(self, filepath):
        self.log = None
        write_object(self, filepath)
                                   

    @classmethod
    def from_file(cls, filepath, log=sys.stdout):
        cls = read_object(filepath)
        cls.log = log
        return cls

    def copy(self):
        return copy.deepcopy(self)


class BayesianClairvoyanceClassification(BayesianClairvoyanceBase):
    def __init__(self, *args, **kwargs):
        if "scorer" not in kwargs:
            kwargs["scorer"] = "accuracy"
        super(BayesianClairvoyanceClassification, self).__init__(*args, **kwargs)

        # Additional initialization for BayesianClairvoyanceClassification if needed

class BayesianClairvoyanceRegression(BayesianClairvoyanceBase):
    def __init__(self, *args, **kwargs):
        if "scorer" not in kwargs:
            kwargs["scorer"] = "neg_root_mean_squared_error"
        super(BayesianClairvoyanceRegression, self).__init__(*args, **kwargs)
      