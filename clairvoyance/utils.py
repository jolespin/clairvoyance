# -*- coding: utf-8 -*-

# Built-ins
from ast import Or
import os, sys, time, warnings, gzip, bz2, operator, pickle
from tqdm import tqdm, tqdm_notebook

# PyData
import pandas as pd
import numpy as np
from scipy.stats import sem

# Machine learning
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import cross_validate, RepeatedStratifiedKFold, StratifiedKFold, RepeatedKFold, KFold

try:
    from feature_engine.selection.base_selection_functions import get_feature_importances
except ImportError:
    # v1.6.2
    from feature_engine.selection.base_selector import get_feature_importances

# from shap import Explainer, TreeExplainer, LinearExplainer
# from shap.maskers import Independent


# ==============================================================================
# Checks from https://github.com/jolespin/soothsayer_utils
# ==============================================================================
def is_file_like(obj):
    return hasattr(obj, "read")

def is_nonstring_iterable(obj):
    condition_1 = hasattr(obj, "__iter__")
    condition_2 =  not type(obj) == str
    return all([condition_1,condition_2])

def assert_acceptable_arguments(query, target, operation="le", message="Invalid option provided.  Please refer to the following for acceptable arguments:"):
    """
    le: operator.le(a, b) : <=
    eq: operator.eq(a, b) : ==
    ge: operator.ge(a, b) : >=
    """
    # If query is not a nonstring iterable or a tuple
    if any([
            not is_nonstring_iterable(query),
            isinstance(query,tuple),
            ]):
        query = [query]
    query = set(query)
    target = set(target)
    func_operation = getattr(operator, operation)
    assert func_operation(query,target), "{}\n{}".format(message, target)
    
def check_testing_set(features, X_testing, y_testing):
    # Testing
    X_testing_is_provided = X_testing is not None
    y_testing_is_provided = y_testing is not None

    if X_testing_is_provided is not None:
        assert y_testing_is_provided is not None, "If `X_testing` is provided then `y_testing` must be provided"

    if y_testing_is_provided is not None:
        assert X_testing_is_provided is not None, "If `y_testing` is provided then `X_testing` must be provided"

    testing_set_provided = False
    if all([X_testing_is_provided, y_testing_is_provided]):
        assert np.all(X_testing.index == y_testing.index), "X_testing.index and y_testing.index must have the same ordering"
        assert np.all(X_testing.columns == pd.Index(features)), "X_testing.columns and X.columns must have the same ordering"
        testing_set_provided = True
    return testing_set_provided
    
# ==============================================================================
# Format
# ==============================================================================

# Format file path
def format_path(path,  into=str, absolute=False):
    assert not is_file_like(path), "`path` cannot be file-like"
    if hasattr(path, "absolute"):
        path = str(path.absolute())
    if hasattr(path, "path"):
        path = str(path.path)
    if absolute:
        path = os.path.abspath(path)
    return into(path)

# Format header for printing
def format_header(text, line_character="=", n=None):
    if n is None:
        n = len(text)
    line = n*line_character
    return "{}\n{}\n{}".format(line, text, line)

# Get duration
def format_duration(t0):
    """
    Adapted from @john-fouhy:
    https://stackoverflow.com/questions/538666/python-format-timedelta-to-string
    """
    duration = time.time() - t0
    hours, remainder = divmod(duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    return "{:02}:{:02}:{:02}".format(int(hours), int(minutes), int(seconds))

# Format stratified data
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


# I/O
# ====
# Reading serial object
def read_object(path, compression="infer", serialization_module=pickle):
    path = format_path(path, str)

    if compression == "infer":
        _ , ext = os.path.splitext(path)
        if (ext == ".pkl") or (ext == ".dill"):
            compression = None
        if ext in {".pgz", ".gz"}:
            compression = "gzip"
        if ext in {".pbz2", ".bz2"}:
            compression = "bz2"
    if compression is not None:
        if compression == "gzip":
            f = gzip.open(path, "rb")
        if compression == "bz2":
            f = bz2.open(path, "rb")
    else:
        f = open(path, "rb")
    obj = serialization_module.load(f)
    f.close()
    return obj

# Writing serial object
def write_object(obj, path, compression="infer", serialization_module=pickle, protocol=None, *args):
    """
    Extensions:
    pickle ==> .pkl
    dill ==> .dill
    gzipped-pickle ==> .pgz
    bzip2-pickle ==> .pbz2
    """
    assert obj is not None, "Warning: `obj` is NoneType"
    path = format_path(path, str)

    # Use infer_compression here
    if compression == "infer":
        _ , ext = os.path.splitext(path)
        if ext in {".pkl", ".dill"}: # if ext in (ext == ".pkl") or (ext == ".dill"):
            compression = None
        if ext in {".pgz", ".gz"}:
            compression = "gzip"
        if ext in {".pbz2", ".bz2"}:
            compression = "bz2"
    if compression is not None:
        if compression == "bz2":
            f = bz2.BZ2File(path, "wb")
        if compression == "gzip":
            f = gzip.GzipFile(path, "wb")
    else:
        f = open(path, "wb")
    serialization_module.dump(obj, f, protocol=protocol, *args)
    f.close()

# Misc
# ====
# Wrapper for tqdm
def pv(iterable, description=None, version=None, total=None, unit='it'):
    """
    Progress viewer
    Wrapper for `tqdm`:
    https://github.com/tqdm/tqdm
    """
    assert_acceptable_arguments([version], {None, "gui",  "notebook"})
    func = tqdm
    if version == "notebook":
        func = tqdm_notebook
    if version == "gui":
        func = tqdm_gui

    return tqdm(
        iterable,
        desc=description,
        total=total,
        unit=unit,
   )
    
# Flatten nested iterables
def flatten(nested_iterable, into=list, unique=False, **kwargs_iterable):
    # Adapted from @wim:
    # https://stackoverflow.com/questions/16312257/flatten-an-iterable-of-iterables
    def _func_recursive(nested_iterable):
        for x in nested_iterable:
            if is_nonstring_iterable(x):
                for element in flatten(x):
                    yield element
            else:
                yield x
    # Unpack data
    data_flattened = list(_func_recursive(nested_iterable))
    if unique:
        data_flattened = sorted(set(data_flattened))
    # Convert type
    return into(data_flattened, **kwargs_iterable)


def get_balanced_class_subset(y:pd.Series, size:float=0.1, random_state=None):
    n = y.size
    number_of_classes = y.nunique()
    if isinstance(size, float):
        number_of_samples_in_subset = int(n * size)
    else:
        number_of_samples_in_subset = n
    subset_size_per_class = int(number_of_samples_in_subset/number_of_classes)
    
    index = list()
    for id_class in y.unique():
        class_samples = y[lambda x: x == id_class].index
        assert len(class_samples) >= subset_size_per_class
        subset = np.random.RandomState(random_state).choice(class_samples, size=subset_size_per_class, replace=False)
        index += subset.tolist()
    return index




# ===================
# Feature importances: 
# ===================
def format_weights(W, scale:bool=True): # Deprecate?
    # with warnings.catch_warnings():
        # warnings.filterwarnings("ignore", category=ConvergenceWarning)
        # warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        # warnings.filterwarnings("ignore", category=UserWarning)
        
    W = np.abs(W)
    if W.ndim > 1:
        W = np.sum(W, axis=0)
    if scale:
        W = W/W.sum()
    return W


# Get feature importance attribute from estimator
def get_feature_importance_attribute(estimator, importance_getter="auto"): # Deprecate?
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

# def shap_importances(estimator, X, explainer_type="auto", masker=None, **explainer_kws):
#     """
#     Modified from Source: 
#     https://github.com/cerlymarco/shap-hypetune
    
#     DOES NOT WORK WITH BASE LEARNERS
    
#     Extract feature importances using Shap

#     explainer_type: 
#         tree, linear
#     Returns
#     -------
#     array of feature importances.
#     """
#     if explainer_type == "auto":
#         importance_getter = get_feature_importance_attribute(estimator, importance_getter="auto")
#         if importance_getter == "feature_importances_":
#             explainer_type = "tree"
#         if importance_getter == "coef_":
#             explainer_type = "linear"

#     if explainer_type == "tree":
#         explainer = TreeExplainer(
#         estimator, 
#         data=X,
#         feature_perturbation="tree_path_dependent",
#         **explainer_kws,
#         )
#     if explainer_type == "linear":
#         masker = Independent(data = X)
#         explainer = LinearExplainer(
#         estimator, 
#         masker=masker,
#         **explainer_kws,
#         # data=X,
#         # feature_perturbation="interventional",
#         )


#     coefs = explainer.shap_values(X)

#     if isinstance(coefs, list):
#         coefs = list(map(lambda x: np.abs(x).mean(0), coefs))
#         # coefs = np.sum(coefs, axis=0)
#     # else:
#         # coefs = np.abs(coefs).mean(0)
#     weights = format_weights(coefs)
#     return weights


# def feature_importances(estimator, importance_getter="auto"):
#     """
#     Extract feature importances from estimator

#     Returns
#     -------
#     array of feature importances.
#     """
#     importance_getter = get_feature_importance_attribute(estimator, importance_getter=importance_getter)
#     if importance_getter == 'coef_':  ## booster='gblinear' (xgb)
#         # coefs = np.square(model.coef_).sum(axis=0)
#         weights = format_weights(estimator.coef_)
#     if importance_getter == 'feature_importances_':  ## booster='gblinear' (xgb)
#         weights = estimator.feature_importances_

#     return weights

# Feature importances from CV
def format_feature_importances_from_cv(cv_results, features):
    # importance for each cross validation fold
    feature_importances_cv = list()
    for estimator in cv_results["estimator"]:
        feature_importances_cv.append(get_feature_importances(estimator))
    feature_importances_mean = pd.Series(np.mean(feature_importances_cv, axis=0), index=features)
    feature_importances_sem = pd.Series(sem(feature_importances_cv, axis=0), index=features)            
        
    return {
        "mean":feature_importances_mean,
        "sem":feature_importances_sem,
    }

# Feature importances from data
def format_feature_importances_from_data(estimator, X, y):
    # importance for each full dataset
    estimator.fit(X,y)
    feature_importances_mean = pd.Series(get_feature_importances(estimator), index=X.columns)
    feature_importances_sem = pd.Series([np.nan]*X.shape[1], index=X.columns)            
        
    return {
        "mean":feature_importances_mean,
        "sem":feature_importances_sem,
    }
