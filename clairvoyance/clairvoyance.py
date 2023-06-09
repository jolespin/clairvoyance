# -*- coding: utf-8 -*-

# Built-ins
from ast import Or
import os, sys, itertools, argparse, time, datetime, copy, warnings
from collections import OrderedDict
# from multiprocessing import cpu_count

# PyData
import pandas as pd
import numpy as np
import xarray as xr

## Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning
from scipy import stats
from sklearn.metrics import get_scorer, make_scorer
from sklearn.model_selection import cross_val_score, train_test_split, RepeatedStratifiedKFold, StratifiedKFold, RepeatedKFold, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning

# # Parallel
# import ray
# from ray.air.config import ScalingConfig
# from ray.train.sklearn import SklearnTrainer

# Clairvoyance 
try:
    from . import __version__ 
except ImportError:
    __version__ = None
    warnings.warn("Unable to load version information")

# Soothsayer Utils
from soothsayer_utils import (
    assert_acceptable_arguments,
    format_header,
    format_duration,
    # read_dataframe,
    read_object,
    write_object,
    # to_precision,
    pv,
    flatten,
)

# Specify maximum number of threads
# available_threads = cpu_count()

# ==============================================================================
# Functions
# ==============================================================================
# Total sum scaling
def closure(X:pd.DataFrame):
    sum_ = X.sum(axis=1)
    return (X.T/sum_).T

# Center Log-Ratio
def clr(X:pd.DataFrame):
    X_tss = closure(X)
    X_log = np.log(X_tss)
    geometric_mean = X_log.mean(axis=1)
    return (X_log - geometric_mean.values.reshape(-1,1))

# Transformations
def transform(X, method="closure", axis=1, multiplicative_replacement="auto", verbose=0, log=sys.stdout):
#     """
#     Assumes X_data.index = Samples, X_data.columns = features
#     axis = 0 = cols, axis = 1 = rows
#     e.g. axis=1, method=closure: transform for relative abundance so each row sums to 1.
#     "closure" = closure/total-sum-scaling
#     "clr" = center log-ratio
#     None = No transformation
#     """
    # Transpose for axis == 0
    if axis == 0:
        X = X.T

    # Base output
    X_transformed = X.copy()

    # Checks
    if X.ndim != 2:
        raise ValueError("Input matrix must have two dimensions")
    # if np.all(X == 0, axis=1).sum() > 0:
    #     raise ValueError("Input matrix cannot have rows with all zeros")

    # Transformed output
    if method is not None:
        if np.any(X.values < 0):
            raise ValueError("Cannot have negative values")
        if X.shape[1] > 1:
            if method == "closure":
                X_transformed = closure(X)
            if method == "clr":
                if multiplicative_replacement == "auto":
                    if not np.all(X > 0):
                        multiplicative_replacement = 1/X.shape[1]**2
                    else:
                        multiplicative_replacement = None
                if multiplicative_replacement is None:
                    multiplicative_replacement = 0
                assert isinstance(multiplicative_replacement, (float,int,np.floating,np.integer))
                X_transformed = clr(X + multiplicative_replacement)
        else:
            if verbose > 1:
                print("Only 1 feature left.  Ignoring transformation.", file=log)

    # Transpose back
    if axis == 0:
        X_transformed = X_transformed.T

    return X_transformed

def format_stratify(stratify, estimator_type:str, y:pd.Series):
    assert_acceptable_arguments(estimator_type, {"classifier", "regressor"})
    if isinstance(stratify, bool):
        if stratify is True:
            assert estimator_type == "classifier", "If `stratify=True` then the estimator must be a classifier.  Please provide a stratification grouping or select None for regressor."
            stratify = "auto"
        if stratify is False:
            stratify = None
    if isinstance(stratify, str):
        assert stratify == "auto", "`stratify` must be 'auto', a pd.Series, or None"
        if estimator_type == "classifier":
            stratify = y.copy()
        if estimator_type == "regressor":
            stratify = None

    if stratify is not None:
        assert isinstance(stratify, pd.Series)
        assert np.all(stratify.index == y.index)
    return stratify

def format_weights(W):
    # with warnings.catch_warnings():
        # warnings.filterwarnings("ignore", category=ConvergenceWarning)
        # warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        # warnings.filterwarnings("ignore", category=UserWarning)
        
    W = np.abs(W)
    if W.ndim > 1:
        W = np.sum(W, axis=0)
    W = W/W.sum()
    return W

def format_cross_validation(cv, X:pd.DataFrame, y:pd.Series, stratify=True, random_state=0, cv_prefix="cv=", training_column="training_index", testing_column="testing_index", return_type=tuple):
    assert_acceptable_arguments({return_type}, (tuple, pd.DataFrame))
    if return_type == tuple:
        assert np.all(X.index == y.index), "`X.index` and `y.index` must be the same ordering"
        index = X.index
        number_of_observations = len(index)

        # Simple stratified k-fold cross validation
        if isinstance(cv, int):
            splits = list()
            labels = list()
            
            if isinstance(stratify, bool):
                if stratify is True:
                    stratify = y.copy()
                if stratify is False:
                    stratify = None

            if stratify is not None:
                assert isinstance(stratify, pd.Series), "If `stratify` is not None, it must be a pd.Series"
                assert np.all(y.index == stratify.index), "If `stratify` is not None then it must have the same index as `y`"
                assert stratify.dtype != float, "`stratify` cannot be floating point values"
                if y.dtype == float:
                    splitter = StratifiedKFold(n_splits=cv, random_state=random_state, shuffle=False if random_state is None else True).split(X, stratify, groups=stratify)
                else:
                    splitter = StratifiedKFold(n_splits=cv, random_state=random_state, shuffle=False if random_state is None else True).split(X, y, groups=stratify)
            else:
                splitter = KFold(n_splits=cv, random_state=random_state, shuffle=False if random_state is None else True).split(X, y)

            for i, (training_index, testing_index) in enumerate(splitter, start=1):
                id_cv = "{}{}".format(cv_prefix, i)
                splits.append([training_index, testing_index])
                labels.append(id_cv)

            return (splits, labels)

        # Repeated stratified k-fold cross validation
        if isinstance(cv, tuple):
            assert len(cv) == 2, "If tuple is provided, it must be length 2"
            assert map(lambda x: isinstance(x, int), cv), "If tuple is provided, both elements must be integers ≥ 2 for `RepeatedStratifiedKFold`"

            splits = list()
            labels = list()

            if isinstance(stratify, bool):
                if stratify is True:
                    stratify = y.copy()
                if stratify is False:
                    stratify = None

            if stratify is not None:
                assert isinstance(stratify, pd.Series), "If `stratify` is not None, it must be a pd.Series"
                assert np.all(y.index == stratify.index), "If `stratify` is not None then it must have the same index as `y`"
                assert stratify.dtype != float, "`stratify` cannot be floating point values"
                if y.dtype == float:
                    splitter = RepeatedStratifiedKFold(n_splits=cv[0], n_repeats=cv[1], random_state=random_state).split(X, stratify, groups=stratify)
                else:
                    splitter = RepeatedStratifiedKFold(n_splits=cv[0], n_repeats=cv[1], random_state=random_state).split(X, y, groups=stratify)
            else:
                splitter = RepeatedKFold(n_splits=cv[0], n_repeats=cv[1], random_state=random_state).split(X, y)

            for i, (training_index, testing_index) in enumerate(splitter, start=1):
                id_cv = "{}{}".format(cv_prefix, i)
                splits.append([training_index, testing_index])
                labels.append(id_cv)

            return (splits, labels)

        # List
        if isinstance(cv, (list, np.ndarray)):
            assert all(map(lambda query: len(query) == 2, cv)), "If `cv` is provided as a list, each element must be a sub-list of 2 indicies (i.e., training and testing indicies)"
            cv = pd.DataFrame(cv, columns=[training_column, testing_column])
            cv.index = cv.index.map(lambda i: "{}{}".format(cv_prefix, i))

        # DataFrame
        if isinstance(cv, pd.DataFrame):
            cv = cv.copy()
            assert training_column in cv.columns
            assert testing_column in cv.columns
            assert np.all(cv[training_column].map(lambda x: isinstance(x, (list, np.ndarray)))), "`{}` must be either list or np.ndarray of indices".format(training_column)
            assert np.all(cv[testing_column].map(lambda x: isinstance(x, (list, np.ndarray)))), "`{}` must be either list or np.ndarray of indices".format(testing_column)

            index_values = flatten(cv.values, into=list)
            query = index_values[0]
            labels = list(cv.index)
            if isinstance(query, (int, np.integer)):
                assert all(map(lambda x: isinstance(x,(int, np.integer)), index_values)), "If `cv` is provided as a list or pd.DataFrame, all values must either be intger values or keys in the X.index"
                assert np.all(np.unique(index_values) >= 0), "All values in `cv` must be positive integers"
                maxv = max(index_values)
                assert maxv < number_of_observations, "Index values in `cv` cannot exceed ({}) the number of observations ({}).".format(maxv, number_of_observations)
            else:
                missing_keys = set(index_values) - set(index)
                assert len(missing_keys) == 0, "If index values in `cv` are keys and not integers, then all keys must be in `X.index`.  Missing keys: {}".format(missing_keys)
                cv = cv.applymap(lambda observations: list(map(lambda x: X.index.get_loc(x), observations)))
            cv = cv.applymap(np.asarray)
            splits = cv.values.tolist()
            return (splits, labels)
    if return_type == pd.DataFrame:
        splits, labels = format_cross_validation(cv=cv, X=X, y=y, stratify=stratify, random_state=random_state, cv_prefix=cv_prefix, training_column=training_column, testing_column=testing_column, return_type=tuple)
        return pd.DataFrame(splits, index=labels, columns=[training_column, testing_column])

def get_feature_importance_attribute(estimator, importance_getter):
    estimator = clone(estimator)
    _X = np.random.normal(size=(5,2))
    if is_classifier(estimator):
        _y = np.asarray(list("aabba"))
    if is_regressor(estimator):
        _y = np.asarray(np.random.normal(size=5))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        estimator.fit(_X,_y)
    if importance_getter == "auto":
        importance_getter = None
        if hasattr(estimator, "coef_"):
            importance_getter = "coef_"
        if hasattr(estimator, "feature_importances_"):
            importance_getter = "feature_importances_"
        assert importance_getter is not None, "If `importance_getter='auto'`, `estimator` must be either a linear model with `.coef_` or a tree-based model with `.feature_importances_`"
    assert hasattr(estimator, importance_getter), "Fitted estimator does not have feature weight attribute: `{}`".format(importance_getter)
    return importance_getter

def get_balanced_class_subset(y:pd.Series, size:float=0.1, random_state=None):
    n = y.size
    number_of_classes = y.nunique()
    number_of_samples_in_subset = int(n * size)
    subset_size_per_class = int(number_of_samples_in_subset/number_of_classes)
    
    index = list()
    for id_class in y.unique():
        class_samples = y[lambda x: x == id_class].index
        assert len(class_samples) >= subset_size_per_class
        subset = np.random.RandomState(random_state).choice(class_samples, size=subset_size_per_class, replace=False)
        index += subset.tolist()
    return index
def recursive_feature_inclusion(
    estimator, 
    X:pd.DataFrame, 
    y:pd.Series, 
    scorer,
    initial_feature_weights:pd.Series, 
    initial_feature_weights_name:str="initial_feature_weights",
    feature_weight_attribute:str="auto",
    transformation=None,
    multiplicative_replacement="auto",
    metric=np.mean, 
    early_stopping=25, 
    minimum_improvement_in_score=0.0,
    additional_feature_penalty=None,
    maximum_number_of_features="auto",
    target_score=-np.inf, 
    less_features_is_better=True,  
    random_state=0,
    n_jobs=1,
    cv=(5,3), 
    stratify=None, 
    training_column="training_index", 
    testing_column="testing_index", 
    cv_prefix="cv=",
    verbose=0,
    log=sys.stdout,
    progress_message="Recursive feature inclusion",
    remove_zero_weighted_features=True,
    maximum_tries_to_remove_zero_weighted_features=1000,
    X_validation:pd.DataFrame=None,
    y_validation:pd.Series=None,
    # optimize_validation_score = "auto",
    ) -> pd.Series:

    assert len(set(X.columns)) == X.shape[1], "Cannot have duplicate feature names in `X`"
    if additional_feature_penalty is None:
        additional_feature_penalty = lambda number_of_features: 0.0

    # Validation
    X_validation_is_provided = X_validation is not None
    y_validation_is_provided = y_validation is not None

    if X_validation_is_provided is not None:
        assert y_validation_is_provided is not None, "If `X_validation` is provided then `y_validation` must be provided"

    if y_validation_is_provided is not None:
        assert X_validation_is_provided is not None, "If `y_validation` is provided then `X_validation` must be provided"

    validation_set_provided = False
    if all([X_validation_is_provided, y_validation_is_provided]):
        assert np.all(X_validation.index == y_validation.index), "X_validation.index and y_validation.index must have the same ordering"
        assert np.all(X_validation.columns == X.columns), "X_validation.columns and X.columns must have the same ordering"
        validation_set_provided = True
    # if optimize_validation_score == "auto":
    #     if validation_set_provided:
    #         optimize_validation_score = True
    #     else:
    #         optimize_validation_score = False

    # Transformations
    assert_acceptable_arguments(transformation, {None,"clr","closure"})
    if multiplicative_replacement is None:
        multiplicative_replacement = 0.0
    if isinstance(multiplicative_replacement, str):
        assert multiplicative_replacement == "auto", "If `multiplicative_replacement` is a string, it must be `auto`"
    else:
        assert isinstance(multiplicative_replacement, (float, np.floating, int, np.integer)), "If `multiplicative_replacement` is not set to `auto` it must be float or int"

    # Cross-validaiton
    cv_splits, cv_labels = format_cross_validation(cv=cv, X=X, y=y, stratify=stratify, random_state=random_state, cv_prefix=cv_prefix, training_column=training_column, testing_column=testing_column)
    
    # Initial feature weights
    assert set(X.columns) <= set(initial_feature_weights.index), "X.columns must be a subset of feature_weights.index"
    initial_feature_weights = initial_feature_weights.sort_values(ascending=False)
    feature_weight_attribute = get_feature_importance_attribute(estimator, feature_weight_attribute)
    
    # Scorer
    if isinstance(scorer, str):
        scorer = get_scorer(scorer)

    # Maximum number of features
    if maximum_number_of_features == "auto":
        maximum_number_of_features = min(X.shape)
    if maximum_number_of_features == -1:
        maximum_number_of_features = X.shape[1]
    if maximum_number_of_features is None:
        maximum_number_of_features = X.shape[1]
    assert maximum_number_of_features > 0, "maximum_number_of_features must be greater than 0"

    # Best results
    history = OrderedDict()
    validation_scores = OrderedDict()

    best_features = None
    best_score = target_score

    # Feature tracker
    feature_tuples = list()
    unique_feature_sets = list()
    
    # Progress tracker
    no_progress = 0

    if progress_message is None:
        iterable = range(initial_feature_weights.size)
    else:
        iterable = pv(range(initial_feature_weights.size), description=progress_message)

    for i in iterable:
        features = initial_feature_weights.index[:i+1].tolist()
        X_rfi = X.loc[:,features]

        continue_algorithm = True

        # Transform features (if transformation = None, then there is no transformation)
        if X_rfi.shape[1] > 1: 
            X_rfi = transform(X=X_rfi, method=transformation, multiplicative_replacement=multiplicative_replacement, axis=1)
        else:
            if transformation is not None:
                continue_algorithm = False
            if verbose > 2:
                print("Only 1 feature left.  Ignoring transformation.", file=log)

        if continue_algorithm:
            
            # Remove zero-weighted features
            if remove_zero_weighted_features:
                for j in range(maximum_tries_to_remove_zero_weighted_features):
                    X_query = X.loc[:,features]

                    estimator.fit(
                        X=transform(X=X_query, method=transformation, multiplicative_replacement=multiplicative_replacement, axis=1), 
                        y=y,
                    )
                    _W = getattr(estimator, feature_weight_attribute)
                    _w = format_weights(_W)
                    mask_zero_weight_features = format_weights(_W) != 0

                    if np.all(mask_zero_weight_features):
                        X_rfi = transform(X=X_query, method=transformation, multiplicative_replacement=multiplicative_replacement, axis=1)
                        if verbose > 1:
                            if j > 0:
                                print("[Success][Iteration={}, Try={}]: Removed all zero weighted features. The following features remain: {}".format(i, j, list(features)), file=log)
                        break
                    else:
                        if verbose > 2:
                            if j > 0:
                                print("[...][Iteration={}, Try={}]: Removing {} features as they have zero weight in fitted model: {}".format(i, j, len(mask_zero_weight_features) - np.sum(mask_zero_weight_features), X_query.columns[~mask_zero_weight_features].tolist()), file=log)
                        features = X_query.columns[mask_zero_weight_features].tolist()

            feature_set = frozenset(features)
            if feature_set  in unique_feature_sets:
                if verbose > 0:
                    print("Skipping iteration {} because removing zero-weighted features has produced a feature set that has already been evaluated: {}".format(i, set(feature_set)), file=log)
            else:
                feature_tuples.append(tuple(features))
                unique_feature_sets.append(feature_set)

                # Training/Testing Scores
                scores = cross_val_score(estimator=estimator, X=X_rfi, y=y, cv=cv_splits, n_jobs=n_jobs, scoring=scorer)

                average_score = np.mean(scores)
                history[i] = scores #{"average_score":average_score, "sem":sem}

                #! ---
                # Validation Score
                query_score = average_score
                validation_score = np.nan
                if validation_set_provided:
                    estimator.fit(
                        X=X_rfi, 
                        y=y,
                    )
                    
                    # Transform features (if transformation = None, then there is no transformation)
                    X_validation_rfi = transform(X=X_validation.loc[:,features], method=transformation, multiplicative_replacement=multiplicative_replacement, axis=1)
                    validation_score = scorer(estimator=estimator, X=X_validation_rfi, y_true=y_validation)
                    validation_scores[i] = validation_score
                    # if optimize_validation_score:
                    #     query_score = validation_score
                #! ---

                # Add penalties to score target
                penalty_adjusted_score_target = (best_score + minimum_improvement_in_score + additional_feature_penalty(len(features)))
                
                if query_score <= penalty_adjusted_score_target:
                    if verbose > 1:
                        # if optimize_validation_score:
                        #     print("Current iteration {} of N={} features has not improved score: Validation Score[{} ≤ {}]".format(i, len(features), validation_score, best_score), file=log)
                        # else:
                        print("Current iteration {} of N={} features has not improved score: Average Score[{} ≤ {}]".format(i, len(features), average_score, best_score), file=log)

                    no_progress += 1
                else:
                    # if optimize_validation_score:
                    #     if verbose > 0:
                    #         print("Updating best score with N={} features : Validation Score[{} -> {}]".format(len(features), best_score, validation_score), file=log)
                    #     best_score = validation_score
                    # else:
                    if verbose > 0:
                        print("Updating best score with N={} features : Average Score[{} -> {}]".format(len(features), best_score, average_score), file=log)
                    best_score = average_score
                    best_features = features
                    no_progress = 0
                if no_progress >= early_stopping:
                    break
            if len(features) >= maximum_number_of_features:
                if verbose > 0:
                    print("Terminating algorithm after {} iterations with a best score of {} as the maximum number of features has been reached.".format(i+1, best_score), file=log)
                break
    if verbose > 0:
        if best_features is None:
            print("Terminating algorithm after {} iterations with a best score of {} as no feature set improved the score with current parameters".format(i+1, best_score), file=log)
        else:
            print("Terminating algorithm at N={} features after {} iterations with a best score of {}".format(len(best_features), i+1, best_score), file=log)
    
    history = pd.DataFrame(history, index=list(map(lambda x: ("splits", x), cv_labels))).T
    # if validation_set_provided:
    #     history[("validation","score")] = pd.Series(validation_scores)
    history.index = feature_tuples
    history.index.name = "features"
    
    # Summary
    average_scores = history.mean(axis=1)
    sems = history.sem(axis=1)
    if validation_set_provided:
        validation_scores = pd.Series(validation_scores)
        validation_scores.index = feature_tuples
    else:
        validation_scores = pd.Series([np.nan]*len(feature_tuples), index=feature_tuples)
    
    history.insert(loc=0, column=("summary", "number_of_features"),value = history.index.map(len))
    history.insert(loc=1, column=("summary", "average_score"),value = average_scores)
    history.insert(loc=2, column=("summary", "sem"),value = sems)
    history.insert(loc=3, column=("summary", "validation_score"), value = validation_scores)
    history.insert(loc=4, column=("summary", "∆(validation_score-average_score)"), value = average_scores - validation_scores)

    history.columns = pd.MultiIndex.from_tuples(history.columns)

    
    # Highest scoring features (not necessarily the best since there can be many features added with minimal gains)
    # if optimize_validation_score:
    #     highest_score = history[("summary", "validation_score")].max()
    #     highest_scoring_features = list(history.loc[history[("summary", "validation_score")] == highest_score].sort_values(
    #         by=[("summary", "average_score"), ("summary", "number_of_features")], 
    #         ascending=[False, less_features_is_better]).index[0])
    # else:
    highest_score = history[("summary", "average_score")].max()
    highest_scoring_features = list(history.loc[history[("summary", "average_score")] == highest_score, ("summary", "number_of_features")].sort_values(ascending=less_features_is_better).index[0])
    
    # # Best results
    # if optimize_validation_score:
    #     # best_features = list(history.loc[history[("summary", "validation_score")] == best_score].sort_values(
    #     #     by=[("summary", "average_score"), ("summary", "number_of_features")], 
    #     #     ascending=[False, less_features_is_better]).index[0])
    # else:
    #     best_features = list(history.loc[history[("summary", "average_score")] == best_score, ("summary", "number_of_features")].sort_values(ascending=less_features_is_better).index[0])
    best_features = list(history.loc[history[("summary", "average_score")] == best_score].sort_values(
        by=[("summary", "validation_score"), ("summary", "average_score"), ("summary", "number_of_features")], 
        ascending=[False, False, less_features_is_better]).index[0])
    
    best_estimator_sem = history.loc[[tuple(best_features)],("summary","sem")].values[0]
    best_estimator_validation_score = history.loc[[tuple(best_features)],("summary","validation_score")].values[0]

    best_estimator_rci = clone(estimator)
    X_best_features = transform(X=X.loc[:,best_features], method=transformation, multiplicative_replacement=multiplicative_replacement, axis=1)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        best_estimator_rci.fit(X_best_features, y)
    
    # Score statement
    if verbose > 0:
        if highest_score > best_score:
            if additional_feature_penalty is not None:
                print("Highest score was %0.3f with %d features but with `minimum_improvement_in_score=%f` penalty and `additional_feature_penalty` adjustment, the best score was %0.3f with %d features.\n^ Both are stored as .highest_score_ / .highest_scoring_features_ and .best_score_ / .best_feautres_, respectively. ^"%(highest_score, len(highest_scoring_features), minimum_improvement_in_score, best_score, len(best_features)), file=log)
            else:
                print("Highest score was %0.3f with %d features but with `minimum_improvement_in_score=%f` penalty, the best score was %0.3f with %d features.\n^ Both are stored as .highest_score_ / .highest_scoring_features_ and .best_score_ / .best_feautres_, respectively. ^"%(highest_score, len(highest_scoring_features), minimum_improvement_in_score, best_score, len(best_features)), file=log)
    


    # Full training dataset weights
    W = getattr(best_estimator_rci, feature_weight_attribute)
    rci_feature_weights = pd.Series(format_weights(W), index=best_features, name="rci_weights")
    feature_weights = pd.DataFrame([pd.Series(initial_feature_weights,name=initial_feature_weights_name), rci_feature_weights]).T
    feature_weights.columns = feature_weights.columns.map(lambda x: ("full_dataset", x))

    # Cross validation weights
    cv_weights = OrderedDict()
    for id_cv, (training_index, testing_index) in zip(cv_labels, cv_splits):
        X_training = transform(X=X.iloc[training_index].loc[:,best_features], method=transformation, multiplicative_replacement=multiplicative_replacement, axis=1)
        y_training = y.iloc[training_index]
        clf = clone(estimator)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            clf.fit(X_training, y_training)
        cv_weights[id_cv] =  pd.Series(format_weights(getattr(clf, feature_weight_attribute)), index=best_features)
    cv_weights = pd.DataFrame(cv_weights)
    cv_weights.columns = cv_weights.columns.map(lambda x: ("cross_validation", x))
    feature_weights = pd.concat([feature_weights, cv_weights], axis=1)

    return pd.Series(
        dict(
        history=history, 
        best_score=best_score, 
        best_estimator_sem=best_estimator_sem,
        best_features=best_features,
        best_estimator_rci=best_estimator_rci,
        feature_weights=feature_weights,
        highest_score=highest_score,
        highest_scoring_features=highest_scoring_features,
        cv_splits=cv_splits, 
        cv_labels=cv_labels,
        validation_scores=validation_scores,
        best_estimator_validation_score=best_estimator_validation_score,
        ),
        name="recursive_feature_elimination",
    )


# Plotting
def plot_scores_line(
    average_scores:pd.Series, 
    sem:pd.Series, 
    validation_scores:pd.Series=None, 
    horizontal_lines=True,
    vertical_lines="auto",
    title=None,
    figsize=(13,3), 
    linecolor="black",
    errorcolor="gray",
    validation_linecolor="red",
    style="seaborn-white",
    xlim=None,
    ylim=None,
    ax=None, 
    alpha=0.382, 
    xlabel="$N_{Features}$", 
    ylabel="Score", 
    xtick_rotation=0,
    show_xgrid=False,
    show_ygrid=True,
    show_xticks=True, 
    show_legend=True,
    xlabel_kws=dict(), 
    ylabel_kws=dict(), 
    xticklabel_kws=dict(), 
    yticklabel_kws=dict(),
    title_kws=dict(),
    legend_kws=dict(),
    **kwargs,
    ):
    with plt.style.context(style):
        _title_kws = {"fontsize":16, "fontweight":"bold"}; _title_kws.update(title_kws)
        _xlabel_kws = {"fontsize":15}; _xlabel_kws.update(xlabel_kws)
        _ylabel_kws = {"fontsize":15}; _ylabel_kws.update(ylabel_kws)
        _xticklabel_kws = {"fontsize":12, "rotation":xtick_rotation}; _xticklabel_kws.update(xticklabel_kws)
        _yticklabel_kws = {"fontsize":12}; _yticklabel_kws.update(yticklabel_kws)
        _legend_kws = {"fontsize":12}; _legend_kws.update(legend_kws)
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = plt.gcf()
        
        average_scores.plot(ax=ax, color=linecolor, label="Average score", **kwargs)
        x_grid = np.arange(average_scores.size)
        ax.fill_between(x_grid, y1=average_scores-sem, y2=average_scores+sem, alpha=alpha, color=errorcolor, label="SEM")
        if validation_scores is not None:
            if not np.all(validation_scores.isnull()):
                validation_scores.plot(ax=ax, color=validation_linecolor, label="Validation score", **kwargs)

        rci = np.argmax(average_scores.values) 
        if vertical_lines == "auto":
            vertical_lines = rci
        if isinstance(vertical_lines, (float,np.floating, int, np.integer)):
            vertical_lines = [vertical_lines]
        
        for v in vertical_lines:
            h = average_scores.iloc[v]
            ax.axvline(v, color=linecolor, linestyle="--", linewidth=0.75, label="%d features (score = %0.3f)"%(v + 1, h))
            if horizontal_lines:
                ax.plot([0,v], [h,h], color=linecolor, linestyle="--", linewidth=0.75)
            
        ax.set_xticks(x_grid)
        if xlim is None:
            xlim = (1,average_scores.size)
        ax.set_xlim((xlim[0]-1, xlim[1]-1)) # Need to offset
        if ylim:
            ax.set_ylim(ylim)
        if show_xticks:
            ax.set_xticklabels(x_grid + 1, **_xticklabel_kws)
        else:
            ax.set_xticklabels([], fontsize=12)
            
        ax.set_xlabel(xlabel, **_xlabel_kws)
        ax.set_ylabel(ylabel, **_ylabel_kws)
        ax.set_yticklabels(map(lambda x:"%0.2f"%x, ax.get_yticks()), **_yticklabel_kws)
        if title:
            ax.set_title(title, **_title_kws)
        if show_legend:
            ax.legend(**_legend_kws)
        if show_xgrid:
            ax.xaxis.grid(True)
        if show_ygrid:
            ax.yaxis.grid(True)
        return fig, ax
    
def plot_weights_bar(
    feature_weights:pd.Series, 
    title=None,
    figsize=(13,3), 
    color="black",
    style="seaborn-white", 
    ylim=None,
    ax=None, 
    ascending=False, 
    xlabel="Features", 
    ylabel="$W$", 
    xtick_rotation=90,
    show_xgrid=False,
    show_ygrid=True,
    show_xticks=True, 
    show_legend=False,
    xlabel_kws=dict(), 
    ylabel_kws=dict(), 
    xticklabel_kws=dict(), 
    yticklabel_kws=dict(),
    title_kws=dict(),
    legend_kws=dict(),
    **kwargs,
    ):
    with plt.style.context(style):
        _title_kws = {"fontsize":16, "fontweight":"bold"}; _title_kws.update(title_kws)
        _xlabel_kws = {"fontsize":15}; _xlabel_kws.update(xlabel_kws)
        _ylabel_kws = {"fontsize":15}; _ylabel_kws.update(ylabel_kws)
        _xticklabel_kws = {"fontsize":12, "rotation":xtick_rotation}; _xticklabel_kws.update(xticklabel_kws)
        _yticklabel_kws = {"fontsize":12}; _yticklabel_kws.update(yticklabel_kws)
        _legend_kws = {"fontsize":12}; _legend_kws.update(legend_kws)
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = plt.gcf()
        
        feature_weights = feature_weights.dropna()
        if ascending is not None:
            feature_weights = feature_weights.sort_values(ascending=ascending)
        feature_weights.plot(ax=ax, color=color, label=ylabel, kind="bar", **kwargs)
        ax.axhline(0, color="black", linewidth=0.618)
        
        if ylim:
            ax.set_ylim(ylim)
        if show_xticks:
            ax.set_xticklabels(ax.get_xticklabels(), **_xticklabel_kws)
        else:
            ax.set_xticklabels([], fontsize=12)
            
        ax.set_xlabel(xlabel, **_xlabel_kws)
        ax.set_ylabel(ylabel, **_ylabel_kws)
        ax.set_yticklabels(map(lambda x:"%0.2f"%x, ax.get_yticks()), **_yticklabel_kws)
        if title:
            ax.set_title(title, **_title_kws)
        if show_legend:
            ax.legend(**_legend_kws)
        if show_xgrid:
            ax.xaxis.grid(True)
        if show_ygrid:
            ax.yaxis.grid(True)
        return fig, ax

def plot_weights_box(
    feature_weights:pd.DataFrame, 
    title=None,
    figsize=(13,3), 
    color="white",
    linecolor="coral",
    swarmcolor="gray",
    style="seaborn-white", 
    ylim=None,
    ax=None, 
    alpha=0.382,
    ascending=False, 
    xlabel="Features", 
    ylabel="$W$", 
    xtick_rotation=90,
    show_swarm=False,
    show_xgrid=False,
    show_ygrid=True,
    show_xticks=True, 
    show_legend=False,
    xlabel_kws=dict(), 
    ylabel_kws=dict(), 
    xticklabel_kws=dict(), 
    yticklabel_kws=dict(),
    title_kws=dict(),
    legend_kws=dict(),
    **kwargs,
    ):
    with plt.style.context(style):
        _title_kws = {"fontsize":16, "fontweight":"bold"}; _title_kws.update(title_kws)
        _xlabel_kws = {"fontsize":15}; _xlabel_kws.update(xlabel_kws)
        _ylabel_kws = {"fontsize":15}; _ylabel_kws.update(ylabel_kws)
        _xticklabel_kws = {"fontsize":12, "rotation":xtick_rotation}; _xticklabel_kws.update(xticklabel_kws)
        _yticklabel_kws = {"fontsize":12}; _yticklabel_kws.update(yticklabel_kws)
        _legend_kws = {"fontsize":12}; _legend_kws.update(legend_kws)
        _box_kws = dict( linewidth=1.0, boxprops={"facecolor": color}, medianprops={"color": linecolor})
        _box_kws.update(kwargs)
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = plt.gcf()
        
        feature_weights = feature_weights.dropna(how="all", axis=0)
        if ascending is not None:
            index = feature_weights.mean(axis=1).sort_values(ascending=ascending).index
            feature_weights = feature_weights.loc[index]
            
        # feature_weights.T.plot(ax=ax, color=color, label=ylabel, kind="box")
        
        data = pd.melt(feature_weights.T, var_name="Feature", value_name="W")

        sns.boxplot(data=data, x="Feature", y="W", ax=ax, **_box_kws)

        if show_swarm:
            sns.swarmplot(data=data, x="Feature", y="W", ax=ax, color=swarmcolor, alpha=alpha)
        

        if ylim:
            ax.set_ylim(ylim)
        if show_xticks:
            ax.set_xticklabels(ax.get_xticklabels(), **_xticklabel_kws)
        else:
            ax.set_xticklabels([], fontsize=12)
            
        ax.set_xlabel(xlabel, **_xlabel_kws)
        ax.set_ylabel(ylabel, **_ylabel_kws)
        ax.set_yticklabels(map(lambda x:"%0.2f"%x, ax.get_yticks()), **_yticklabel_kws)
        if title:
            ax.set_title(title, **_title_kws)
        if show_legend:
            ax.legend(**_legend_kws)
        if show_xgrid:
            ax.xaxis.grid(True)
        if show_ygrid:
            ax.yaxis.grid(True)
        return fig, ax

def plot_recursive_feature_selection(
    number_of_features:pd.Series, 
    average_scores:pd.Series, 
    # sem:pd.Series=None, 
    # validation_scores:pd.Series=None,
    min_features:int=None,
    max_features:int=None,
    min_score:float=None,
    max_score:float=None,
    ax=None,
    color="darkslategray",
    color_validation="red",
    linewidth=0.618,
    alpha=0.618,
    edgecolor="black", 
    style="seaborn-white",
    figsize=(8,3),
    title=None,
    xlabel="$N_{Features}$",
    ylabel="Score",
    xtick_rotation=0,
    show_xgrid=False,
    show_ygrid=True,
    show_xticks=True, 
    show_legend=False,
    xlabel_kws=dict(), 
    ylabel_kws=dict(), 
    xticklabel_kws=dict(), 
    yticklabel_kws=dict(),
    title_kws=dict(),
    legend_kws=dict(),
    **kwargs,
    ):

    assert isinstance(number_of_features, pd.Series)
    assert isinstance(average_scores, pd.Series)
    assert np.all(number_of_features.index == average_scores.index)
    df = pd.DataFrame([number_of_features, average_scores], index=["number_of_features", "average_scores"]).T
    # if sem is not None:
    #     assert isinstance(sem, pd.Series)
    #     assert np.all(df.index == sem.index)
    #     df["sem"] = sem
    
    if min_features:
        df = df.query("number_of_features >= {}".format(min_features))
    if max_features:
        df = df.query("number_of_features <= {}".format(max_features))
    if min_score:
        df = df.query("average_scores >= {}".format(min_score))
    if max_score:
        df = df.query("average_scores <= {}".format(max_score))
        
    with plt.style.context(style):
        _title_kws = {"fontsize":16, "fontweight":"bold"}; _title_kws.update(title_kws)
        _xlabel_kws = {"fontsize":15}; _xlabel_kws.update(xlabel_kws)
        _ylabel_kws = {"fontsize":15}; _ylabel_kws.update(ylabel_kws)
        _xticklabel_kws = {"fontsize":12, "rotation":xtick_rotation}; _xticklabel_kws.update(xticklabel_kws)
        _yticklabel_kws = {"fontsize":12}; _yticklabel_kws.update(yticklabel_kws)
        _legend_kws = {"fontsize":12}; _legend_kws.update(legend_kws)
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = plt.gcf()

        # if sem is not None:
        #     ax.errorbar(df["number_of_features"].values, df["average_scores"].values, yerr=df["sem"].values, alpha=0.1618, color=color)
        ax.scatter(x=df["number_of_features"].values, y=df["average_scores"].values, edgecolor=edgecolor, alpha=alpha, linewidth=linewidth, color=color, **kwargs)

        ax.set_xlabel(xlabel, **_xlabel_kws)
        ax.set_ylabel(ylabel, **_ylabel_kws)
        # ax.set_yticklabels(map(lambda x:"%0.2f"%x, ax.get_yticks()), **_yticklabel_kws)
        if title:
            ax.set_title(title, **_title_kws)
        if show_legend:
            ax.legend(**_legend_kws)
        if show_xgrid:
            ax.xaxis.grid(True)
        if show_ygrid:
            ax.yaxis.grid(True)
        return fig, ax
    
def plot_scores_comparison(
    number_of_features:pd.Series, 
    average_scores:pd.Series, 
    validation_scores:pd.Series=None,
    min_features:int=None,
    max_features:int=None,
    min_score:float=None,
    max_score:float=None,
    ax=None,
    color="darkslategray",
    linewidth=0.618,
    alpha=0.618,
    edgecolor="black", 
    style="seaborn-white",
    figsize=(8,5),
    title=None,
    xlabel="Average Score",
    ylabel="Validation Score",

    feature_to_size_function = "auto",

    xtick_rotation=0,
    show_xgrid=False,
    show_ygrid=True,
    # show_zgrid=True,

    show_xticks=True, 
    show_legend=True,
    xlabel_kws=dict(), 
    ylabel_kws=dict(), 
    # zlabel_kws=dict(), 

    xticklabel_kws=dict(), 
    yticklabel_kws=dict(),
    # zticklabel_kws=dict(),

    title_kws=dict(),
    legend_kws=dict(),
    **kwargs,
    ):

    assert isinstance(number_of_features, pd.Series)
    assert isinstance(average_scores, pd.Series)
    assert isinstance(validation_scores, pd.Series)

    assert np.all(number_of_features.index == average_scores.index)
    assert np.all(number_of_features.index == validation_scores.index)

    df = pd.DataFrame([number_of_features, average_scores, validation_scores], index=["number_of_features", "average_scores", "validation_scores"]).T

    if min_features:
        df = df.query("number_of_features >= {}".format(min_features))
    if max_features:
        df = df.query("number_of_features <= {}".format(max_features))
    if min_score:
        df = df.query("average_scores >= {}".format(min_score))
    if max_score:
        df = df.query("average_scores <= {}".format(max_score))
    if min_score:
        df = df.query("validation_scores >= {}".format(min_score))
    if max_score:
        df = df.query("validation_scores <= {}".format(max_score))

    if feature_to_size_function == "auto":
        def feature_to_size_function(n):
            return 100/n
    assert hasattr(feature_to_size_function, "__call__")
    s = feature_to_size_function(number_of_features)

    with plt.style.context(style):
        _title_kws = {"fontsize":16, "fontweight":"bold"}; _title_kws.update(title_kws)
        _xlabel_kws = {"fontsize":15}; _xlabel_kws.update(xlabel_kws)
        _ylabel_kws = {"fontsize":15}; _ylabel_kws.update(ylabel_kws)
        # _zlabel_kws = {"fontsize":15}; _zlabel_kws.update(zlabel_kws)

        _xticklabel_kws = {"fontsize":12, "rotation":xtick_rotation}; _xticklabel_kws.update(xticklabel_kws)
        _yticklabel_kws = {"fontsize":12}; _yticklabel_kws.update(yticklabel_kws)
        # _zticklabel_kws = {"fontsize":12}; _zticklabel_kws.update(zticklabel_kws)

        _legend_kws = {"fontsize":12, "frameon":True, "title_fontsize":"x-large"}; _legend_kws.update(legend_kws)
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        else:
            fig = plt.gcf()


        # ax.scatter(xs=df["number_of_features"].values, ys=df["average_scores"].values, zs=df["validation_scores"], edgecolor=edgecolor, alpha=alpha, linewidth=linewidth, color=color, **kwargs)
        ax.scatter(x=df["average_scores"].values, y=df["validation_scores"], s=s, edgecolor=edgecolor, alpha=alpha, linewidth=linewidth, color=color, **kwargs)


        ax.set_xlabel(xlabel, **_xlabel_kws)
        ax.set_ylabel(ylabel, **_ylabel_kws)
        # ax.set_zlabel(zlabel, **_zlabel_kws)

        # ax.set_yticklabels(map(lambda x:"%0.2f"%x, ax.get_yticks()), **_yticklabel_kws)
        if title:
            ax.set_title(title, **_title_kws)
        if show_legend:
            legendary_features = number_of_features.describe()[["25%", "50%", "75%", "max"]].astype(int)
            
            legend_elements = list()
            for n in legendary_features.values:
                marker = plt.Line2D([], [], color=color, marker='o', linestyle='None', markersize=feature_to_size_function(n), label="$N$ = %d"%(n))
                legend_elements.append(marker)

            legend = ax.legend(handles=legend_elements, markerscale=1,title="$N_{Features}$",  **_legend_kws)
            
        if show_xgrid:
            ax.xaxis.grid(True)
        if show_ygrid:
            ax.yaxis.grid(True)
        # if show_zgrid:
        #     ax.zaxis.grid(True)
        return fig, ax

# -------
# Classes
# -------
class ClairvoyanceBase(object):
    def __init__(
        self,
        # Modeling
        estimator,
        param_grid:dict,
        scorer,
        method:str="symmetric",
        importance_getter="auto",
        n_draws=50,
        random_state=0,
        n_jobs=1,

        # Transformation
        transformation=None,
        multiplicative_replacement="auto",

        # Zero weights
        remove_zero_weighted_features=True,
        maximum_tries_to_remove_zero_weighted_features=1000,

        # Labeling
        name=None,
        observation_type=None,
        feature_type=None,
        target_type=None,
        
        verbose=1,
        log=sys.stdout,
         ):         

        # # Tasks
        # if n_jobs == -1:
        #     n_jobs = cpu_count()
        # assert n_jobs > 0
            
        # Method
        assert_acceptable_arguments(method, {"asymmetric", "symmetric"})
        self.method = method
            
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
        self.feature_weight_attribute = get_feature_importance_attribute(estimator, importance_getter)
        assert len(param_grid) > 0, "`param_grid` must have at least 1 key:[value_1, value_2, ..., value_n] pair"
        self.param_grid = param_grid
            
        # Set attributes
        self.name = name
        self.observation_type = observation_type
        self.feature_type = feature_type
        self.target_type = target_type
        self.is_fitted_weights = False
        self.is_fitted_rci = False
        self.validation_set_provided = False
        self.n_draws = n_draws
        self.n_jobs = n_jobs
        self.random_state = random_state
        if isinstance(scorer, str):
            scorer = get_scorer(scorer)
        self.scorer = scorer
        self.scorer_name = scorer._score_func.__name__
        self.verbose = verbose
        self.log = log

        # Transformations
        assert_acceptable_arguments(transformation, {None,"clr","closure"})
        self.transformation = transformation
        if multiplicative_replacement is None:
            multiplicative_replacement = 0.0
        if isinstance(multiplicative_replacement, str):
            assert multiplicative_replacement == "auto", "If `multiplicative_replacement` is a string, it must be `auto`"
        else:
            assert isinstance(multiplicative_replacement, (float, np.floating, int, np.integer)), "If `multiplicative_replacement` is not set to `auto` it must be float or int"
        self.multiplicative_replacement = multiplicative_replacement

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
            pad*" " + "* Parameter Space: {}".format(self.param_grid),
            pad*" " + "* Scorer: {}".format(self.scorer_name),
            pad*" " + "* Method: {}".format(self.method),
            pad*" " + "* Feature Weight Attribute: {}".format(self.feature_weight_attribute),

            pad*" " + "- -- --- ----- -------- -------------",
            
            pad*" " + "* n_draws: {}".format(self.n_draws),
            pad*" " + "* n_jobs: {}".format(self.n_jobs),
            pad*" " + "* random_state: {}".format(self.random_state),

            pad*" " + "- -- --- ----- -------- -------------",
            
            pad*" " + "* Observation Type: {}".format(self.observation_type),
            pad*" " + "* Feature Type: {}".format(self.feature_type),
            pad*" " + "* Target Type: {}".format(self.target_type),
            pad*" " + "* Target Type: {}".format(self.target_type),
            pad*" " + "* Transformation: {}".format(self.transformation),
            pad*" " + "* Multiplicative Replacement: {}".format(self.multiplicative_replacement),
            
            pad*" " + "- -- --- ----- -------- -------------",
            pad*" " + "* Remove Zero Weighted Features: {}".format(self.remove_zero_weighted_features),
            pad*" " + "* Maximum Tries to Remove: {}".format(self.maximum_tries_to_remove_zero_weighted_features),

            pad*" " + "- -- --- ----- -------- -------------",

            pad*" " + "* Fitted(Weights): {}".format(self.is_fitted_weights),
            pad*" " + "* Fitted(RCI): {}".format(self.is_fitted_rci),
            
            
            ]

        return "\n".join(fields)
        
    
    def fit(
        self, 
        X:pd.DataFrame, 
        y:pd.Series, 
        stratify="auto", 
        split_size=0.618033, 
        reset_fitted_estimators=True, 
        sort_hyperparameters_by:list=None, 
        ascending:list=None, 
        progress_message="Permuting samples and fitting models",
    ):
        """
        """

        def _get_estimators():
            # Indexing for hyperparameters and models
            estimators_ = OrderedDict()

            param_grid_expanded = list(map(lambda x:dict(zip(self.param_grid.keys(), x)), itertools.product(*self.param_grid.values())))
            for i, params in enumerate(param_grid_expanded):
                # Construct model
                estimator = clone(self.estimator)
                estimator.set_params(**params)
                estimators_[frozenset(params.items())] = estimator

            return estimators_
        

        # @ray.remote
        def _fit_estimator_symmetric(X_A, X_B, y_A, y_B, estimator):
            """
            Internal: Get coefs
            score_A_B means score of B trained on A
            """

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
                warnings.filterwarnings("ignore", category=UserWarning)

                # Subset A
                estimator.fit(X_A, y_A)
                weight_A = getattr(estimator, self.feature_weight_attribute)
                score_A_B = self.scorer(estimator=estimator, X=X_B, y_true=y_B)

                # Subset B
                estimator.fit(X_B, y_B)
                weight_B = getattr(estimator, self.feature_weight_attribute)
                score_B_A = self.scorer(estimator=estimator, X=X_A, y_true=y_A)
            
            # Avoid instances where all weights are 0
            A_is_all_zeros = np.all(weight_A == 0)
            B_is_all_zeros = np.all(weight_B == 0)
            
            if any([A_is_all_zeros, B_is_all_zeros]):
                weight_nan = np.zeros(weight_A.shape[-1])
                weight_nan[:] = np.nan
                return (weight_nan, np.nan)
            else:
                weight_A = format_weights(weight_A)
                weight_B = format_weights(weight_B)

                # v2: Rationale for taking mean is to minimize overfitting when selecting score thresholds using --minimum_threshold
                return (np.mean([weight_A, weight_B], axis=0), np.mean([score_A_B, score_B_A]))

        def _fit_estimator_asymmetric(X_training, X_testing, y_training, y_testing, estimator):
            """
            Internal: Get coefs
            """

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
                warnings.filterwarnings("ignore", category=UserWarning)

                # Fit
                estimator.fit(X_training, y_training)
                weights = getattr(estimator, self.feature_weight_attribute)
                score = self.scorer(estimator=estimator, X=X_testing, y_true=y_testing)
            
            # Avoid instances where all weights are 0
            weights_are_all_zeros = np.all(weights == 0)
            if weights_are_all_zeros:
                weight_nan = np.zeros(weights.shape[-1])
                weight_nan[:] = np.nan
                return (weight_nan, np.nan)
            else:
                weights = format_weights(weights)
                return (weights, score)


        def _run(X, y, stratify, split_size, method, progress_message):
            """
            Internal: Distribute tasks
            # (n_space_per_iteration, 2_splits, n_classifications, n_features)
            """
            
            # Iterate through n_draws
            feature_weights_collection = list()
            scores_collection = list()
                        
            fit_function = {"symmetric":_fit_estimator_symmetric, "asymmetric":_fit_estimator_asymmetric}[method]
            for i in pv(range(self.random_state, self.random_state + self.n_draws), description=progress_message):
                if method == "symmetric":
                    # Split training data
                    X_A, X_B, y_A, y_B = train_test_split(
                        X,
                        y,
                        test_size=split_size, 
                        stratify=stratify, 
                        random_state=i,
                    )

                    training_data = dict(
                        X_A=X_A,
                        X_B=X_B,
                        y_A=y_A,
                        y_B=y_B,
                    )

                if method == "asymmetric":
                    # Split training data
                    X_training, X_testing, y_training, y_testing = train_test_split(
                        X,
                        y,
                        test_size=split_size, 
                        stratify=stratify, 
                        random_state=i,
                    )

                    training_data = dict(
                        X_training=X_training,
                        X_testing=X_testing,
                        y_training=y_training,
                        y_testing=y_testing,
                    )

                
                # Fit estimators and unpack results
                weights = list()
                scores = list()
                
                for estimator in self.estimators_.values():
                    w, s = fit_function(estimator=estimator, **training_data)
                    weights.append(w)
                    scores.append(s)

                # parallel_results = Parallel(n_jobs=self.n_jobs)(delayed(_fit_logistic_regression)(estimator=estimator, id=id, **training_data) for id, estimator in self.estimators_.items())
                # futures = [_fit_estimators.remote(estimator=estimator, **training_data) id=id, **training_data) for id, estimator in self.estimators_.items()]
                # parallel_results = ray.get(futures)
                
                feature_weights_collection.append(weights)
                scores_collection.append(scores)
            
            # Concatenate and label arrays
            feature_weights_collection = np.stack(feature_weights_collection)
            feature_weights_collection = xr.DataArray(
                data=feature_weights_collection, 
                dims=["iterations", "hyperparameters", "features"], 
                coords=[np.arange(self.n_draws), list(self.estimators_.keys()), self.feature_ids_], 
                name=self.name,
                attrs={"estimator_name":self.estimator_name, "feature_weight_attribute":self.feature_weight_attribute, "scorer_name":self.scorer_name, "random_state":self.random_state, "method":method},
            )
            scores_collection = np.stack(scores_collection)
            scores_collection = pd.DataFrame(scores_collection, index=np.arange(self.n_draws), columns=list(self.estimators_.keys()))
            scores_collection.index.name = "iterations"
            scores_collection.columns.name = "hyperparameters"


            return (feature_weights_collection, scores_collection)
        assert np.all(X.index == y.index), "X.index and y.index must have the same ordering"
        self.X_ = X.copy()
        self.y_ = y.copy()
        if self.estimator_type == "classifier":
            assert y.dtype != float
            self.y_ = y.astype("category")
            self.classes_ = sorted(y.unique())
        self.observation_ids_ = X.index
        self.feature_ids_ = X.columns
        self.number_of_observations_, self.number_of_initial_features_ = X.shape
        self.stratify_ = format_stratify(stratify, estimator_type=self.estimator_type, y=self.y_)

        # Checks
        if sort_hyperparameters_by is not None:
            assert isinstance(sort_hyperparameters_by, list), "`sort_hyperparameters_by` must be a list of size n accompanied by an `ascending` list of n boolean vaues"
            assert isinstance(ascending, list), "`sort_hyperparameters_by` must be a list of size n accompanied by an `ascending` list of n boolean vaues"
            assert len(sort_hyperparameters_by) == len(ascending), "`sort_hyperparameters_by` must be a list of size n accompanied by an `ascending` list of n boolean vaues"
            assert all(map(lambda x: isinstance(x, bool), ascending)), "`sort_hyperparameters_by` must be a list of size n accompanied by an `ascending` list of n boolean vaues"
            assert set(sort_hyperparameters_by) <= set(self.param_grid.keys()), "`sort_hyperparameters_by` must contain a list of keys in `param_grid`"
        else:
            sort_hyperparameters_by = []
            ascending = []
            
        # Fitting 
        self.estimators_ = _get_estimators()
        self.number_of_estimators_ = len(self.estimators_)

        X_query = transform(X=self.X_, method=self.transformation, multiplicative_replacement=self.multiplicative_replacement, axis=1, log=self.log, verbose=self.verbose)

        self.intermediate_weights_, self.intermediate_scores_ = _run(X=X_query, y=self.y_, stratify=self.stratify_, split_size=split_size, method=self.method, progress_message=progress_message)
        # Get best params
        average_scores = self.intermediate_scores_.mean(axis=0).dropna().sort_values()
        df = pd.DataFrame(average_scores.index.map(dict).tolist(), index=average_scores.index)
        df["average_score"] = average_scores
        df = df.sort_values(["average_score"] + sort_hyperparameters_by, ascending=[False] + ascending)
        
        self.best_estimator_ = clone(self.estimator)
        self.best_estimator_.set_params(**dict(df.index[0]))
        self.best_hyperparameters_ = self.best_estimator_.get_params(deep=True)
        self.hyperparameter_average_scores_ = df
        
        if reset_fitted_estimators:
            if self.verbose > 1:
                print("Resetting fitted estimators", file=self.log)
            for params, estimator in self.estimators_.items():
                if self.verbose > 3:
                    print("[Resetting] {}".format(params), file=self.log)
                self.estimators_[params] = clone(estimator)
            
        self.is_fitted_weights = True
                                      
        return self
    
    def get_weights(self, minimum_score=None, metrics=[np.mean, stats.sem], minimim_score_too_high_action="adjust"):
        assert self.is_fitted_weights, "Please `fit` model before proceeding."
        assert_acceptable_arguments(minimim_score_too_high_action, {"adjust", "fatal"})

        minimum_score = -np.inf if minimum_score is None else minimum_score
        
        mask = self.intermediate_scores_ >= minimum_score
        if not np.any(mask):
            maxv = self.intermediate_scores_.values.ravel().max()
            if minimim_score_too_high_action == "adjust":
                minimum_score = maxv
                if self.verbose > 1:
                    print("No scores are available for `minimum_score={}`. Adjusting `minimum_score = {}`".format(minimum_score, maxv), file=self.log)
                mask = self.intermediate_scores_ >= minimum_score
            if minimim_score_too_high_action == "fatal":
                assert np.any(mask), "No scores are available for `minimum_score={}`. Please lower `minimum_score ≤ {}`".format(minimum_score, maxv)
        W = self.intermediate_weights_.values[mask.values]
        
        if callable(metrics):
            metrics = [metrics]
        output = OrderedDict()
        for func in metrics:
            w = func(W, axis=0)
            output[func.__name__] = pd.Series(w, index=self.feature_ids_)
        df = pd.DataFrame(output)
        df.index.name = "features"
        df.columns.name = "metrics"
        return df.squeeze()
    
    
    def recursive_feature_inclusion(
        self, 
        estimator=None, 
        X:pd.DataFrame=None, 
        y:pd.Series=None, 
        X_validation:pd.DataFrame=None,
        y_validation:pd.Series=None,
        cv=(5,3), 
        minimum_score=None, 
        metric=np.mean, 
        early_stopping=25, 
        target_score=-np.inf, 
        minimum_improvement_in_score=0.0, 
        additional_feature_penalty=None,
        maximum_number_of_features=np.inf,
        less_features_is_better=True, 
        training_column="training_index", 
        testing_column="testing_index", 
        cv_prefix="cv=", 
        copy_X_rci=True,
        progress_message="Recursive feature inclusion",
        # optimize_validation_score = "auto",

        ):
        assert self.is_fitted_weights, "Please `fit` model before proceeding."

        if X is None:
            X = self.X_.copy()
        else:
            assert set(X.columns) == set(self.X_.columns), "`X.columns` must be a subset `.feature_ids`"

        if y is None:
            y = self.y_.copy()
        else:
            if self.estimator_type == "classifier":
                assert set(y.unique()) <= set(self.classes_), "`y` must be a subset `.classes_ = {}`".format(self.classes_)

        if estimator is None:
            estimator = self.best_estimator_

        self.clairvoyance_feature_weights_ = self.get_weights(minimum_score=minimum_score, metrics=metric).sort_values(ascending=False)
        self.rci_parameters_ = {"estimator":estimator, "cv":cv, "minimum_score":minimum_score, "metric":metric, "early_stopping":early_stopping, "target_score":target_score, "less_features_is_better":less_features_is_better}
        
        # Validation
        X_validation_is_provided = X_validation is not None
        y_validation_is_provided = y_validation is not None

        if X_validation_is_provided is not None:
            assert y_validation_is_provided is not None, "If `X_validation` is provided then `y_validation` must be provided"

        if y_validation_is_provided is not None:
            assert X_validation_is_provided is not None, "If `y_validation` is provided then `X_validation` must be provided"

        self.validation_set_provided = False
        if all([X_validation_is_provided, y_validation_is_provided]):
            assert np.all(X_validation.index == y_validation.index), "X_validation.index and y_validation.index must have the same ordering"
            assert np.all(X_validation.columns == X.columns), "X_validation.columns and X.columns must have the same ordering"
            self.validation_set_provided = True


        # Recursive feature incusion
        rci_results = recursive_feature_inclusion(
            estimator=estimator, 
            X=X, 
            y=y, 
            scorer=self.scorer,
            initial_feature_weights=self.clairvoyance_feature_weights_, 
            initial_feature_weights_name="clairvoyance_weights",
            feature_weight_attribute=self.feature_weight_attribute,
            transformation=self.transformation,
            multiplicative_replacement=self.multiplicative_replacement,
            metric=metric,
            early_stopping=early_stopping, 
            minimum_improvement_in_score=minimum_improvement_in_score,
            additional_feature_penalty=additional_feature_penalty,
            maximum_number_of_features=maximum_number_of_features,
            target_score=target_score, 
            less_features_is_better=less_features_is_better,  
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            cv=cv, 
            stratify=self.stratify_, 
            training_column=training_column, 
            testing_column=testing_column, 
            cv_prefix=cv_prefix,
            verbose=self.verbose,
            progress_message=progress_message,
            remove_zero_weighted_features=self.remove_zero_weighted_features,
            maximum_tries_to_remove_zero_weighted_features=self.maximum_tries_to_remove_zero_weighted_features,
            X_validation=X_validation,
            y_validation=y_validation,
            # optimize_validation_score=optimize_validation_score,
            )
        
        # Results
        # self.validation_scores_ = rci_results["validation_scores"]
        self.history_ = rci_results["history"]
        if self.remove_zero_weighted_features:
            if self.validation_set_provided:
                self.history_ = self.history_.sort_values([("summary", "validation_score"), ("summary", "average_score"), ("summary", "number_of_features")], ascending=[False, False, less_features_is_better])
            else:
                self.history_ = self.history_.sort_values([("summary", "average_score"), ("summary", "number_of_features")], ascending=[False, less_features_is_better])
        self.highest_score_ = rci_results["highest_score"]
        self.highest_scoring_features_ = rci_results["highest_scoring_features"]
        self.best_score_ = rci_results["best_score"]
        self.best_estimator_sem_ = rci_results["best_estimator_sem"]
        self.best_features_ = rci_results["best_features"]
        self.best_estimator_rci_ = clone(estimator)
        self.best_estimator_validation_score_ = rci_results["best_estimator_validation_score"]

        X_rci = transform(X=X.loc[:,self.best_features_], method=self.transformation, multiplicative_replacement=self.multiplicative_replacement, axis=1)
        with warnings.catch_warnings(): #!
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            self.best_estimator_rci_.fit(X_rci, y)
        self.feature_weights_ =  rci_results["feature_weights"]
        self.rci_feature_weights_ = rci_results["feature_weights"][("full_dataset", "rci_weights")].loc[self.best_features_]
        self.cv_splits_ = rci_results["cv_splits"]
        self.cv_labels_ = rci_results["cv_labels"]
        
        if copy_X_rci:
            self.X_rci_ = X_rci.copy()
            
        self.is_fitted_rci = True
        
        return self.history_
    
    def plot_scores(
        self,
        ylabel="auto",
        **kwargs,
        ):
        if ylabel == "auto":
            ylabel = self.scorer_name
        kwargs["ylabel"] = ylabel
        assert self.is_fitted_rci, "Please run `recursive_feature_inclusion` before proceeding."
        if self.best_score_ < self.highest_score_:
            vertical_lines = [len(self.best_features_)-1, len(self.highest_scoring_features_)-1]
        else:
            vertical_lines = [len(self.best_features_)-1]

        if self.remove_zero_weighted_features:
            return plot_recursive_feature_selection(number_of_features=self.history_[("summary", "number_of_features")], average_scores=self.history_[("summary", "average_score")],  **kwargs)
        else:
            return plot_scores_line(average_scores=self.history_[("summary", "average_score")], sem=self.history_[("summary", "sem")], validation_scores=self.history_[("summary", "validation_score")], vertical_lines=vertical_lines, **kwargs)
        
    def plot_weights(
        self,
        weight_type=("full_dataset","rci_weights"),
        **kwargs,
        ):
        assert self.is_fitted_rci, "Please run `recursive_feature_inclusion` before proceeding."
        assert_acceptable_arguments(weight_type, {("full_dataset","rci_weights"), ("full_dataset","clairvoyance_weights"), "cross_validation"})
        
        if weight_type in {("full_dataset","rci_weights"), ("full_dataset","clairvoyance_weights")}:
            fig, ax = plot_weights_bar(feature_weights=self.feature_weights_[weight_type], **kwargs)
        if weight_type == "cross_validation":
            fig, ax = plot_weights_box(feature_weights=self.feature_weights_[weight_type], **kwargs)
        return fig, ax
    
    def plot_scores_comparison(
        self,
        **kwargs,
        ):
        assert self.is_fitted_rci, "Please run `recursive_feature_inclusion` before proceeding."
        assert self.validation_set_provided, "Please run `recursive_feature_inclusion` with a validation set before proceeding."
        return plot_recursive_feature_selection(number_of_features=self.history_[("summary", "number_of_features")], average_scores=self.history_[("summary", "average_score")],   validation_scores=self.history_[("summary", "validation_score")], **kwargs)

    def copy(self):
        return copy.deepcopy(self)
    
    # def to_file(self, path:str):
    #     write_object(self, path)  
        
    # @classmethod
    # def from_file(cls, path:str):
    #     cls = read_object(path)
    #     return cls
        
        
class ClairvoyanceClassification(ClairvoyanceBase):
    def __init__(
        self,
        # Modeling
        estimator,
        param_grid:dict,
        scorer="accuracy",
        method="asymmetric",
        importance_getter="auto",
        n_draws=10,
        random_state=0,
        n_jobs=1,

        # Transformations
        transformation=None,
        multiplicative_replacement="auto",

        # Zero weights
        remove_zero_weighted_features=True,
        maximum_tries_to_remove_zero_weighted_features=1000,

        # Labeling
        name=None,
        observation_type=None,
        feature_type=None,
        target_type=None,
        
        # Log
        verbose=1,
        log=sys.stdout,
         ): 

        if isinstance(scorer, str):
            assert scorer == "accuracy", "Only `accuracy` is supported when providing `scorer` as a string.  Please use a `scorer` object if any other scoring is implemented.  Many require `pos_label` and `average` arguments." 
            scorer = get_scorer(scorer)
            
        
        super(ClairvoyanceClassification, self).__init__(
            estimator=estimator,
            param_grid=param_grid,
            scorer=scorer,
            method=method,
            importance_getter=importance_getter,
            n_draws=n_draws,
            random_state=random_state,
            n_jobs=n_jobs,

            # Labeling
            name=name,
            observation_type=observation_type,
            feature_type=feature_type,
            target_type=target_type,
            
            # Transformation
            transformation=transformation,
            multiplicative_replacement=multiplicative_replacement,

            # Zero weights
            remove_zero_weighted_features=remove_zero_weighted_features,
            maximum_tries_to_remove_zero_weighted_features=maximum_tries_to_remove_zero_weighted_features,

            # Log
            verbose=verbose,
            log=log,
        )

class ClairvoyanceRegression(ClairvoyanceBase):

    def __init__(
        self,
        # Modeling
        estimator,
        param_grid:dict,
        scorer="neg_root_mean_squared_error",
        method="asymmetric",
        importance_getter="auto",
        n_draws=10,
        random_state=0,
        n_jobs=1,

        # Transformations
        transformation=None,
        multiplicative_replacement="auto",

        # Zero weights
        remove_zero_weighted_features=True,
        maximum_tries_to_remove_zero_weighted_features=1000,

        # Labeling
        name=None,
        observation_type=None,
        feature_type=None,
        target_type=None,
        
        # Log
        verbose=1,
        log=sys.stdout,

         ): 

            
        super(ClairvoyanceRegression, self).__init__(
            estimator=estimator,
            param_grid=param_grid,
            scorer=scorer,
            method=method,
            importance_getter=importance_getter,
            n_draws=n_draws,
            random_state=random_state,
            n_jobs=n_jobs,

            # Labeling
            name=name,
            observation_type=observation_type,
            feature_type=feature_type,
            target_type=target_type,

            # Transformation
            transformation=transformation,
            multiplicative_replacement=multiplicative_replacement,

            # Zero weights
            remove_zero_weighted_features=remove_zero_weighted_features,
            maximum_tries_to_remove_zero_weighted_features=maximum_tries_to_remove_zero_weighted_features,

            # Utility
            verbose=verbose,
            log=log,

        
        )

class ClairvoyanceRecursive(object):
    def __init__(
        self,
        # Modeling
        estimator,
        param_grid:dict,
        scorer,
        method="symmetric",
        importance_getter="auto",
        n_draws=10,
        random_state=0,
        n_jobs=1,
        
        # Recursive feature inclusion
        early_stopping=25, 
        minimum_improvement_in_score=0.0, 
        additional_feature_penalty=None,

        #Iterative
        percentiles=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.925,0.95,0.975,0.99],
        minimum_scores=[-np.inf],
        
        # Compositional data transformations
        transformation=None,
        multiplicative_replacement="auto",
        
        # Zero weights
        remove_zero_weighted_features=True,
        maximum_tries_to_remove_zero_weighted_features=1000,

        # Labeling
        name=None,
        observation_type=None,
        feature_type=None,
        target_type=None,
        
        verbose=1,
        log=sys.stdout,
         ): 
        # # Tasks
        # if n_jobs == -1:
        #     n_jobs = cpu_count()
        #     assert n_jobs > 0
            
        # Method
        assert_acceptable_arguments(method, {"asymmetric", "symmetric"})
        self.method = method
        
        # Estimator
        self.estimator_name = estimator.__class__.__name__
        if is_classifier(estimator):
            self.estimator_type = "classifier"
            self.clairvoyance_class = ClairvoyanceClassification
        if is_regressor(estimator):
            self.estimator_type = "regressor"
            self.clairvoyance_class = ClairvoyanceRegression

        self.estimator = clone(estimator)
        
        if "random_state" in self.estimator.get_params(deep=True):
            query = self.estimator.get_params(deep=True)["random_state"]
            if query is None:
                if verbose > 0:
                    print("Updating `random_state=None` in `estimator` to `random_state={}`".format(random_state), file=log)
                self.estimator.set_params(random_state=random_state)
        self.feature_weight_attribute = get_feature_importance_attribute(estimator, importance_getter)
        assert len(param_grid) > 0, "`param_grid` must have at least 1 key:[value_1, value_2, ..., value_n] pair"
        self.param_grid = param_grid
            
        # Transformations
        assert_acceptable_arguments(transformation, {None,"clr","closure"})
        self.transformation = transformation
        if multiplicative_replacement is None:
            multiplicative_replacement = 0.0
        if isinstance(multiplicative_replacement, str):
            assert multiplicative_replacement == "auto", "If `multiplicative_replacement` is a string, it must be `auto`"
        else:
            assert isinstance(multiplicative_replacement, (float, np.floating, int, np.integer)), "If `multiplicative_replacement` is not set to `auto` it must be float or int"
        self.multiplicative_replacement = multiplicative_replacement
        
        # Set attributes
        self.name = name
        self.observation_type = observation_type
        self.feature_type = feature_type
        self.target_type = target_type
        # self.is_fitted_weights = False
        self.is_fitted_rci = False
        self.validation_set_provided = False
        self.n_draws = n_draws
        self.n_jobs = n_jobs
        self.random_state = random_state
        if isinstance(scorer, str):
            scorer = get_scorer(scorer)
        self.scorer = scorer
        self.scorer_name = scorer._score_func.__name__
        if isinstance(percentiles, (float, np.floating, int, np.integer)):
            percentiles = [percentiles]
        assert all(map(lambda x: 0.0 <= x < 1.0, percentiles)), "All percentiles must be 0.0 ≤ x < 1.0"
        percentiles = sorted(map(float, percentiles))
        if percentiles[0] > 0.0:
            percentiles = [0.0] + percentiles
        self.percentiles = percentiles

        if minimum_scores is None:
            minimum_scores = -np.inf
        if isinstance(minimum_scores, (float, np.floating)):
            minimum_scores = [minimum_scores]
        if minimum_scores[0] != -np.inf:
            minimum_scores = [-np.inf] + minimum_scores
        self.minimum_scores = sorted(minimum_scores)  
        self.early_stopping = early_stopping
        self.minimum_improvement_in_score = minimum_improvement_in_score
        self.additional_feature_penalty = additional_feature_penalty
        self.verbose = verbose
        self.log = log
        self.remove_zero_weighted_features = remove_zero_weighted_features
        self.maximum_tries_to_remove_zero_weighted_features = maximum_tries_to_remove_zero_weighted_features

    def __repr__(self):
        pad = 4
        header = format_header("{}(Name:{})".format(self.__class__.__name__, self.name),line_character="=")
        n = len(header.split("\n")[0])
        fields = [
            header,
            pad*" " + "* Estimator: {}".format(self.estimator_name),
            pad*" " + "* Estimator Type: {}".format(self.estimator_type),
            pad*" " + "* Parameter Space: {}".format(self.param_grid),
            pad*" " + "* Scorer: {}".format(self.scorer_name),
            pad*" " + "* Method: {}".format(self.method),
            pad*" " + "* Feature Weight Attribute: {}".format(self.feature_weight_attribute),

            pad*" " + "- -- --- ----- -------- -------------",
            
            pad*" " + "* n_draws: {}".format(self.n_draws),
            pad*" " + "* n_jobs: {}".format(self.n_jobs),
            pad*" " + "* random_state: {}".format(self.random_state),
            
            pad*" " + "- -- --- ----- -------- -------------",
            
            pad*" " + "* percentiles: {}".format(self.percentiles),
            pad*" " + "* minimum_scores: {}".format(self.minimum_scores),

            pad*" " + "- -- --- ----- -------- -------------",
            
            pad*" " + "* Observation Type: {}".format(self.observation_type),
            pad*" " + "* Feature Type: {}".format(self.feature_type),
            pad*" " + "* Target Type: {}".format(self.target_type),
            pad*" " + "* Transformation: {}".format(self.transformation),
            pad*" " + "* Multiplicative Replacement: {}".format(self.multiplicative_replacement),

            pad*" " + "- -- --- ----- -------- -------------",
            pad*" " + "* Remove Zero Weighted Features: {}".format(self.remove_zero_weighted_features),
            pad*" " + "* Maximum Tries to Remove: {}".format(self.maximum_tries_to_remove_zero_weighted_features),
        
            
            pad*" " + "- -- --- ----- -------- -------------",
            pad*" " + "* Fitted(RCI): {}".format(self.is_fitted_rci),
            
            ]

        return "\n".join(fields)
    
    def fit(
        self, 
        X:pd.DataFrame, 
        y:pd.Series, 
        X_validation:pd.DataFrame=None,
        y_validation:pd.Series=None,
        stratify="auto", 
        split_size=0.618033, 
        cv=(5,3),
        training_column="training_index", 
        testing_column="testing_index", 
        cv_prefix="cv=",
        sort_hyperparameters_by:list=None, 
        ascending:list=None,
        less_features_is_better=True,
        remove_redundancy=True,
        maximum_number_of_features="auto",
        ):

        assert np.all(X.index == y.index), "X.index and y.index must have the same ordering"
        self.X_initial_ = X.copy()
        self.y_ = y.copy()
        if self.estimator_type == "classifier":
            assert y.dtype != float
            self.y_ = y.astype("category")       
            self.classes_ = sorted(y.unique())
        self.observation_ids_ = X.index
        self.feature_ids_initial_ = X.columns
        self.number_of_observations_, self.number_of_initial_features_ = X.shape
        self.stratify_ = format_stratify(stratify, estimator_type=self.estimator_type, y=self.y_)
        self.split_size = split_size
        self.maximum_number_of_features = maximum_number_of_features

        # Validation
        X_validation_is_provided = X_validation is not None
        y_validation_is_provided = y_validation is not None

        if X_validation_is_provided is not None:
            assert y_validation_is_provided is not None, "If `X_validation` is provided then `y_validation` must be provided"

        if y_validation_is_provided is not None:
            assert X_validation_is_provided is not None, "If `y_validation` is provided then `X_validation` must be provided"

        self.validation_set_provided = False
        if all([X_validation_is_provided, y_validation_is_provided]):
            assert np.all(X_validation.index == y_validation.index), "X_validation.index and y_validation.index must have the same ordering"
            assert np.all(X_validation.columns == X.columns), "X_validation.columns and X.columns must have the same ordering"
            self.validation_set_provided = True
            
        # Get cross-validation splits
        self.cv_splits_, self.cv_labels_ = format_cross_validation(cv, X, self.y_, stratify=self.stratify_, random_state=self.random_state, cv_prefix=cv_prefix, training_column=training_column, testing_column=testing_column)

        self.history_ = OrderedDict()
        self.results_ = OrderedDict()
        self.results_baseline_ = OrderedDict()

        with np.errstate(divide='ignore', invalid='ignore'):
            current_features_for_percentile = X.columns
            for i,pctl in enumerate(self.percentiles):
                if len(current_features_for_percentile) > 1:
                    if self.verbose > 2:
                        print("Feature set for percentile={}:".format(pctl), current_features_for_percentile.tolist(), sep="\n", file=self.log)


                    # Initiate model
                    model = self.clairvoyance_class(
                        estimator=self.estimator,
                        param_grid=self.param_grid,
                        scorer=self.scorer,
                        method=self.method,
                        importance_getter=self.feature_weight_attribute,
                        n_draws=self.n_draws,
                        random_state=self.random_state,
                        n_jobs=self.n_jobs,
                        name=(self.name,"percentile={}".format(pctl)),
                        observation_type=self.observation_type,
                        feature_type=self.feature_type,
                        target_type=self.target_type,
                        verbose=self.verbose - 2,
                        log=self.log,
                        transformation=self.transformation,
                        multiplicative_replacement=self.multiplicative_replacement,
                        remove_zero_weighted_features=self.remove_zero_weighted_features,
                        maximum_tries_to_remove_zero_weighted_features=self.maximum_tries_to_remove_zero_weighted_features,
                    )

                    # Fit model
                    model.fit(
                    X=self.X_initial_.loc[:,current_features_for_percentile], 
                    y=self.y_, 
                    stratify=self.stratify_, 
                    split_size=split_size, 
                    reset_fitted_estimators=False, 
                    sort_hyperparameters_by=sort_hyperparameters_by, 
                    ascending=ascending, 
                    progress_message="Permuting samples and fitting models [percentile={}, number_of_features={}]".format(pctl, len(current_features_for_percentile)),
                    )

                    # Check for redundant minimum score thresholds
                    if self.verbose > 1:
                        print("Determining (and removing) minimum score thresholds that yield redundant feature ordering", file=self.log)
                    feature_ordering_from_minimum_scores = dict()
                    for s in self.minimum_scores:
                        w = model.get_weights(s)["mean"].sort_values(ascending=False)
                        feature_order = tuple(w.index.tolist())
                        if feature_order in feature_ordering_from_minimum_scores.values():
                            if self.verbose > 2:
                                print("Removing minimum_score = {} as feature ordering is already accounted for by minimum_score = {}".format(s, {v:k for k,v in feature_ordering_from_minimum_scores.items()}[feature_order]), file=self.log)
                        else:
                            feature_ordering_from_minimum_scores[s] = feature_order  

                    # Baseline Clairvoyance weights to update.  These will be used for selecting the next set of features that are fed back into the algorithm
                    best_clairvoyance_feature_weights_for_percentile = None
                    best_score_for_percentile = -np.inf
                    best_hyperparameters_for_percentile = None
                    best_minimum_score_for_percentile = None

                    X_query = transform(X=self.X_initial_.loc[:,current_features_for_percentile], method=self.transformation, multiplicative_replacement=self.multiplicative_replacement, axis=1)
                    for params, estimator in model.estimators_.items():      

                        # Baseline
                        baseline_scores_for_percentile = cross_val_score(estimator=estimator, X=X_query, y=self.y_, scoring=self.scorer, cv=self.cv_splits_, n_jobs=self.n_jobs)
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", category=ConvergenceWarning)
                            baseline_rci_weights = getattr(estimator.fit(X_query, self.y_), self.feature_weight_attribute)
                        if np.all(baseline_rci_weights == 0):
                            if self.verbose > 2:
                                print("Excluding results from [percentile={}, estimator_params={}] becaue baseline model could not be fit with parameter set".format(pctl, params), file=self.log)
                        else:

                            baseline_validation_score = np.nan
                            if self.validation_set_provided:
                                # Transform features (if transformation = None, then there is no transformation)
                                X_validation_query = transform(X=X_validation.loc[:,current_features_for_percentile], method=self.transformation, multiplicative_replacement=self.multiplicative_replacement, axis=1)
                                baseline_validation_score = self.scorer(estimator=estimator, X=X_validation_query, y_true=y_validation)

                            baseline_rci_weights = format_weights(baseline_rci_weights)
                            self.results_baseline_[(pctl,"baseline", params)] = {
                                "validation_score":baseline_validation_score,
                                "average_score":np.nanmean(baseline_scores_for_percentile), 
                                "sem":stats.sem(baseline_scores_for_percentile),
                                "number_of_features":X_query.shape[1], 
                                "features":X_query.columns.tolist(), 
                                "clairvoyance_weights":np.nan,
                                "rci_weights":baseline_rci_weights, 
                                "estimator":estimator,
                            }

                            # Minimum score thresholds
                            for s in sorted(feature_ordering_from_minimum_scores):
                                progress_message = {True:"Recursive feature inclusion [percentile={}, estimator_params={}, minimum_score={}]".format(pctl, params, s), False:None}[self.verbose > 1]
                                rci_params =  dict(
                                        estimator=estimator, 
                                        X=self.X_initial_.loc[:,current_features_for_percentile], 
                                        y=self.y_, 
                                        cv=self.cv_splits_, 
                                        minimum_score=s, 
                                        metric=np.mean, 
                                        early_stopping=self.early_stopping, 
                                        minimum_improvement_in_score=self.minimum_improvement_in_score, 
                                        additional_feature_penalty=self.additional_feature_penalty,
                                        maximum_number_of_features=self.maximum_number_of_features,
                                        target_score=-np.inf, 
                                        less_features_is_better=less_features_is_better, 
                                        progress_message=progress_message,
                                )
                                if self.validation_set_provided:
                                    rci_params.update(
                                        dict(
                                        X_validation=X_validation.loc[:,current_features_for_percentile],
                                        y_validation=y_validation,
                                        )
                                    )

                                rci_history = model.recursive_feature_inclusion(**rci_params)



                                #!
                                # Update weights if applicable
                                if model.best_score_ > best_score_for_percentile:
                                    best_score_for_percentile = model.best_score_
                                    best_hyperparameters_for_percentile = params
                                    best_minimum_score_for_percentile = s
                                    best_clairvoyance_feature_weights_for_percentile = model.clairvoyance_feature_weights_.copy()

                                # Store results
                                rci_feature_weights = model.rci_feature_weights_[model.best_features_]
                                if not np.any(rci_feature_weights.isnull()) and np.any(rci_feature_weights > 0):
                                    self.results_[(pctl, params, s)] = {
                                        "validation_score":model.best_estimator_validation_score_,
                                        "average_score":model.best_score_, 
                                        "sem":model.best_estimator_sem_,
                                        "number_of_features":len(model.best_features_), 
                                        "features":list(model.best_features_), 
                                        "clairvoyance_weights":model.clairvoyance_feature_weights_[model.best_features_].values.tolist(),
                                        "rci_weights":rci_feature_weights.values.tolist(), 
                                        "estimator":estimator,
                                    }
                                else:
                                    if self.verbose > 2:
                                        print("Excluding results from [percentile={}, estimator_params={}, minimum_score={}] becaue model could not be fit with parameter set".format(pctl, params, s), file=self.log)
                                self.history_[(pctl,params, s)] = rci_history

                                # Reset estimator
                                model.estimators_[params] = clone(estimator)

                    # Get new features
                    if i < len(self.percentiles) - 1:
                        if self.remove_zero_weighted_features: #! Used to be exclude_zero_weighted_features which had a separate functionality from removing zero weighted features in the models.  Keep eye on this
                            nonzero_weights = best_clairvoyance_feature_weights_for_percentile[lambda x: x > 0]
                            current_features_for_percentile = best_clairvoyance_feature_weights_for_percentile[lambda w: w >= np.percentile(nonzero_weights, q=100*self.percentiles[i+1])].sort_values(ascending=False).index
                        else:
                            current_features_for_percentile = best_clairvoyance_feature_weights_for_percentile[lambda w: w >= np.percentile(best_clairvoyance_feature_weights_for_percentile, q=100*self.percentiles[i+1])].sort_values(ascending=False).index
                    else:
                        if self.verbose > 0:
                            print("Terminating algorithm. Last percentile has been processed.", file=self.log)
                else:
                    if self.verbose > 0:
                        print("Terminating algorithm. Only 1 feature remains.", file=self.log)
                    break

            self.results_ = pd.DataFrame(self.results_).T.sort_values(["validation_score", "average_score", "number_of_features", "sem"], ascending=[False, False,less_features_is_better, True])
            self.results_.index.names = ["percentile", "hyperparameters", "minimum_score"]

            self.results_ = self.results_.loc[:,["validation_score", "average_score", "sem", "number_of_features", "features", "clairvoyance_weights", "rci_weights", "estimator"]]

            # Dtypes
            for field in ["validation_score", "average_score", "sem"]:
                self.results_[field] = self.results_[field].astype(float)
            for field in ["number_of_features"]:
                self.results_[field] = self.results_[field].astype(int)

            # Remove redundancy
            if remove_redundancy:
                unique_results = set()
                unique_index = list()
                for idx, row in pv(self.results_.iterrows(), "Removing duplicate results", total=self.results_.shape[0]):
                    id_unique = frozenset([row["average_score"], row["sem"], tuple(row["features"])])
                    if id_unique not in unique_results:
                        unique_results.add(id_unique)
                        unique_index.append(idx)
                    else:
                        if self.verbose > 2:
                            print("Removing duplicate result: {}".format(id_unique), file=self.log)
                self.results_ = self.results_.loc[unique_index]
            self.results_baseline_ = pd.DataFrame(self.results_baseline_).T
            self.results_baseline_.index.names = ["percentile", "hyperparameters", "minimum_score"]

            self.is_fitted_rci = True
            return self

    def get_history(self, sort_values_by=[("summary", "validation_score"), ("summary", "average_score"), ("summary", "number_of_features")], ascending=[False, False, True]):
        assert self.is_fitted_rci, "Please run `fit` before proceeding."

        dataframes = list()
        for params, df in self.history_.items(): # self.history_[(pctl,params, s)] = rci_history
            df.index = df.index.map(lambda x: (*params, x))
            df.index.names = ["percentile", "hyperparameters", "minimum_score", "features"]
            dataframes.append(df)

        df_concatenated = pd.concat(dataframes, axis=0)
        if sort_values_by is not None:
            df_concatenated = df_concatenated.sort_values(by=sort_values_by, ascending=ascending)
            
        return df_concatenated


    def plot_recursive_feature_selection(
        self,
        ylabel="auto",
        comprehensive = "auto",
        include_baseline=False,
        **kwargs,
        ):
        if ylabel == "auto":
            ylabel = self.scorer_name
        kwargs["ylabel"] = ylabel
        assert self.is_fitted_rci, "Please run `fit` before proceeding."
        
        if comprehensive == "auto":
            if self.validation_set_provided:
                comprehensive = True 
            else:
                comprehensive = False
            
        if comprehensive:
            df = self.get_history()
            number_of_features = df[("summary","number_of_features")]
            average_scores = df[("summary", "average_score")]
            # validation_scores = df[("summary", "validation_score")]
        else:
            number_of_features = self.results_["number_of_features"]
            average_scores = self.results_["average_score"]
            validation_scores = self.results_["validation_score"]
            if include_baseline:
                number_of_features = pd.concat([number_of_features, self.results_baseline_["number_of_features"]])
                average_scores = pd.concat([average_scores, self.results_baseline_["average_score"]])
                # validation_scores = pd.concat([validation_scores, self.results_baseline_["validation_score"]])

        return plot_recursive_feature_selection(number_of_features=number_of_features, average_scores=average_scores,  **kwargs)

    def plot_scores_comparison(
        self,
        **kwargs,
        ):
        assert self.is_fitted_rci, "Please run `recursive_feature_inclusion` before proceeding."
        assert self.validation_set_provided, "Please run `recursive_feature_inclusion` with a validation set before proceeding."
        df = self.get_history()
        number_of_features = df[("summary","number_of_features")]
        average_scores = df[("summary", "average_score")]
        validation_scores = df[("summary", "validation_score")]
        return plot_scores_comparison(number_of_features=number_of_features, average_scores=average_scores,   validation_scores=validation_scores, **kwargs)

    def to_file(self, path:str):
        write_object(self, path)  


        
    @classmethod
    def from_file(cls, path:str):
        cls = read_object(path)
        return cls

def main():
    s = """
_______        _______ _____  ______ _    _  _____  __   __ _______ __   _ _______ _______
|       |      |_____|   |   |_____/  \  /  |     |   \_/   |_____| | \  | |       |______
|_____  |_____ |     | __|__ |    \_   \/   |_____|    |    |     | |  \_| |_____  |______
    """
    print(s, file=sys.stderr)
    print("Hello There.\nI live here: https://github.com/jolespin/clairvoyance", file=sys.stderr)
    if len(sys.argv) > 0:
        if sys.argv[1] == "test":
            # Classification
            print("\nRunning test for `ClairvoyanceClassification`", file=sys.stderr)
            import numpy as np
            import pandas as pd
            from sklearn.datasets import load_iris
            from sklearn.linear_model import LogisticRegression

            # Load iris dataset
            X, y = load_iris(return_X_y=True, as_frame=True)
            X.columns = X.columns.map(lambda j: j.split(" (cm")[0].replace(" ","_"))

            # Relabel targets
            target_names = load_iris().target_names
            y = y.map(lambda i: target_names[i])

            # Add 996 noise features (total = 1000 features) in the same range of values as the original features
            number_of_noise_features = 46
            vmin = X.values.ravel().min()
            vmax = X.values.ravel().max()
            X_noise = pd.DataFrame(
                data=np.random.RandomState(0).randint(low=int(vmin*10), high=int(vmax*10), size=(150, number_of_noise_features))/10,
                columns=map(lambda j:"noise_{}".format(j+1), range(number_of_noise_features)),
            )

            X_iris_with_noise = pd.concat([X, X_noise], axis=1)
            X_normalized = X_iris_with_noise - X_iris_with_noise.mean(axis=0).values
            X_normalized = X_normalized/X_normalized.std(axis=0).values

            # Specify model algorithm and parameter grid
            estimator=LogisticRegression(max_iter=1000, solver="liblinear", multi_class="ovr")
            param_grid={
                "C":[1e-10] + (np.arange(1,11)/10).tolist(),
                "penalty":["l1", "l2"],
            }

            # Instantiate model
            clf = ClairvoyanceClassification(
                n_jobs=-1, 
                scorer="accuracy", 
                n_draws=10, 
                estimator=estimator, 
                param_grid=param_grid, 
                verbose=1,
            )
            clf.fit(X_normalized, y)#, sort_hyperparameters_by=["C", "penalty"], ascending=[True, False])
            history = clf.recursive_feature_inclusion(early_stopping=10)
            print(history.head(), file=sys.stdout)

            # Regression
            print("\nRunning test for `ClairvoyanceRegression`", file=sys.stderr)
            # from sklearn.datasets import fetch_openml
            from sklearn.tree import DecisionTreeRegressor
            from sklearn.model_selection import train_test_split

            # Load housing data
            # housing = fetch_openml(name="house_prices", as_frame=True)
            data_url = "http://lib.stat.cmu.edu/datasets/boston"
            raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
            data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
            target = raw_df.values[1::2, 2]
            X = pd.DataFrame(data, columns=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'])
            y = pd.Series(target)

            number_of_noise_features = 50 - X.shape[1]
            X_noise = pd.DataFrame(np.random.RandomState(0).normal(size=(X.shape[0], number_of_noise_features)),  columns=map(lambda j: f"noise_{j}", range(number_of_noise_features)))
            X_housing_with_noise = pd.concat([X, X_noise], axis=1)
            X_normalized = X_housing_with_noise - X_housing_with_noise.mean(axis=0).values
            X_normalized = X_normalized/X_normalized.std(axis=0).values

            # Let's fit the model but leave a held out validation set
            X_training, X_validation, y_training, y_validation = train_test_split(X_normalized, y, random_state=0, test_size=0.1618)

            # Get parameters
            estimator = DecisionTreeRegressor(random_state=0)
            param_grid = {"min_samples_leaf":[1,2,3,5,8],"min_samples_split":{ 0.1618, 0.382, 0.5, 0.618}}

            # Fit model
            reg = ClairvoyanceRegression(name="Housing", n_jobs=-1, n_draws=10, estimator=estimator, param_grid=param_grid, verbose=1)
            reg.fit(X_training, y_training)
            history = reg.recursive_feature_inclusion(early_stopping=10, X=X_training, y=y_training, X_validation=X_validation, y_validation=y_validation)
            print(history.head(), file=sys.stdout)

            # Recursive
            print("\nRunning test for `ClairvoyanceRecursive`", file=sys.stderr)
            from sklearn.tree import DecisionTreeClassifier

            X_normalized = X_iris_with_noise - X_iris_with_noise.mean(axis=0).values
            X_normalized = X_normalized/X_normalized.std(axis=0).values
            target_names = load_iris().target_names
            y = pd.Series(load_iris().target)
            y = y.map(lambda i: target_names[i])

            # Specify model algorithm and parameter grid
            estimator=DecisionTreeClassifier()
            param_grid={
                "criterion":["gini","entropy"],
                "max_features":["log2", "sqrt", None, 0.382, 0.618],
                "min_samples_leaf":[1,2,3,5,8, 13],
            }

            # Instantiate model
            rci = ClairvoyanceRecursive(
                n_jobs=-1, 
                scorer="accuracy", 
                n_draws=10, 
                estimator=estimator, 
                param_grid=param_grid, 
                percentiles=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.925, 0.95, 0.975, 0.99],
                minimum_scores=[-np.inf, 0.382, 0.5],
                verbose=0,
            )
            rci.fit(X_normalized, y)
            print(rci.results_.head(), file=sys.stdout)
        else:
            print("Unrecognized command.  Available commands {test}", file=sys.stderr)
if __name__ == "__main__":
    main()