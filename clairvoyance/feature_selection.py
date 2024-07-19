import sys,warnings
from typing import Any, List, Union
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.stats import sem 
from sklearn.model_selection import cross_validate
from sklearn.metrics import get_scorer, make_scorer
from sklearn.base import is_classifier, is_regressor


# Feature Engine
from feature_engine._docstrings.init_parameters.selection import (
    _confirm_variables_docstring,
    _estimator_docstring,

)

from feature_engine.dataframe_checks import check_X_y



from feature_engine.tags import _return_tags


from feature_engine._docstrings.methods import _fit_transform_docstring

from feature_engine._docstrings.fit_attributes import (
    _feature_importances_docstring,
    _feature_names_in_docstring,
    _n_features_in_docstring,
    _performance_drifts_docstring,
)

from feature_engine._docstrings.selection._docstring import (
    _cv_docstring,
    _features_to_drop_docstring,
    _fit_docstring,
    _get_support_docstring,
    _initial_model_performance_docstring,
    _scoring_docstring,
    _threshold_docstring,
    _transform_docstring,
    _variables_attribute_docstring,
    _variables_numerical_docstring,
)
from feature_engine._docstrings.substitute import Substitution

# v1.8.+
Variables = Union[None, int, str, List[Union[str, int]]]
try:
    from feature_engine.selection.base_selection_functions import get_feature_importances
    from feature_engine.selection.base_selector import BaseSelector
    from feature_engine._check_init_parameters.check_variables import (
    _check_variables_input_value,
    )
    from feature_engine.variable_handling import (
        check_numerical_variables,
        find_numerical_variables,
        retain_variables_if_in_df,
    )
# v1.6.2
except ImportError:
    from feature_engine.selection.base_selector import BaseSelector, get_feature_importances
    
    def _check_variables_input_value(variables: Variables) -> Any:
        """
        Checks that the input value for the `variables` parameter located in the init of
        all Feature-engine transformers is of the correct type.
        Allowed  values are None, int, str or list of strings and integers.

        Parameters
        ----------
        variables : string, int, list of strings, list of integers. Default=None

        Returns
        -------
        variables: same as input
        """

        msg = (
            "`variables` should contain a string, an integer or a list of strings or "
            f"integers. Got {variables} instead."
        )
        msg_dupes = "The list entered in `variables` contains duplicated variable names."
        msg_empty = "The list of `variables` is empty."

        if variables is not None:
            if isinstance(variables, list):
                if not all(isinstance(i, (str, int)) for i in variables):
                    raise ValueError(msg)
                if len(variables) == 0:
                    raise ValueError(msg_empty)
                if len(variables) != len(set(variables)):
                    raise ValueError(msg_dupes)
            else:
                if not isinstance(variables, (str, int)):
                    raise ValueError(msg)
        return variables
    
    def check_numerical_variables(
        X: pd.DataFrame, variables: Variables
    ) -> List[Union[str, int]]:
        """
        Checks that the variables in the list are of type numerical.

        More details in the :ref:`User Guide <check_num_vars>`.

        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The dataset.

        variables : List
            The list with the names of the variables to check.

        Returns
        -------
        variables: List
            The names of the numerical variables.

        Examples
        --------
        >>> import pandas as pd
        >>> from feature_engine.variable_handling import check_numerical_variables
        >>> X = pd.DataFrame({
        >>>     "var_num": [1, 2, 3],
        >>>     "var_cat": ["A", "B", "C"],
        >>>     "var_date": pd.date_range("2020-02-24", periods=3, freq="T")
        >>> })
        >>> var_ = check_numerical_variables(X, variables=["var_num"])
        >>> var_
        ['var_num']
        """

        if isinstance(variables, (str, int)):
            variables = [variables]

        if len(X[variables].select_dtypes(exclude="number").columns) > 0:
            raise TypeError(
                "Some of the variables are not numerical. Please cast them as "
                "numerical before using this transformer."
            )

        return variables
    
    def find_numerical_variables(X: pd.DataFrame) -> List[Union[str, int]]:
        """
        Returns a list with the names of all the numerical variables in a dataframe.

        More details in the :ref:`User Guide <find_num_vars>`.

        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The dataset.

        Returns
        -------
        variables: List
            The names of the numerical variables.

        Examples
        --------
        >>> import pandas as pd
        >>> from feature_engine.variable_handling import find_numerical_variables
        >>> X = pd.DataFrame({
        >>>     "var_num": [1, 2, 3],
        >>>     "var_cat": ["A", "B", "C"],
        >>>     "var_date": pd.date_range("2020-02-24", periods=3, freq="T")
        >>> })
        >>> var_ = find_numerical_variables(X)
        >>> var_
        ['var_num']
        """
        variables = list(X.select_dtypes(include="number").columns)
        if len(variables) == 0:
            raise TypeError(
                "No numerical variables found in this dataframe. Please check "
                "variable format with pandas dtypes."
            )
        return variables
    
    def retain_variables_if_in_df(X, variables):
        """Returns the subset of variables in the list that are present in the dataframe.

        More details in the :ref:`User Guide <retain_vars>`.

        Parameters
        ----------
        X:  pandas dataframe of shape = [n_samples, n_features]
            The dataset.

        variables: string, int or list of strings or int.
            The names of the variables to check.

        Returns
        -------
        variables_in_df: List.
            The subset of `variables` that is present `X`.

            Examples
        --------
        >>> import pandas as pd
        >>> from feature_engine.variable_handling import retain_variables_if_in_df
        >>> X = pd.DataFrame({
        >>>     "var_num": [1, 2, 3],
        >>>     "var_cat": ["A", "B", "C"],
        >>>     "var_date": pd.date_range("2020-02-24", periods=3, freq="T")
        >>> })
        >>> vars_in_df = retain_variables_if_in_df(X, ['var_num', 'var_cat', 'var_other'])
        >>> vars_in_df
        ['var_num', 'var_cat']
        """
        if isinstance(variables, (str, int)):
            variables = [variables]

        variables_in_df = [var for var in variables if var in X.columns]

        # Raise an error if no column is left to work with.
        if len(variables_in_df) == 0:
            raise ValueError(
                "None of the variables in the list are present in the dataframe."
            )

        return variables_in_df


# Internals
from .utils import (
    assert_acceptable_arguments, 
    check_testing_set, 
    format_cross_validation,
    format_feature_importances_from_cv, 
    format_feature_importances_from_data,
)
from .transformations import (
    closure, 
    clr, 
    clr_with_multiplicative_replacement,
)




_get_transformation_docstring = """
    transformation: str,callable, default=None
        If provided, must be either 'closure', 'clr', or 'clr_with_multiplicative_replacement'.  
        Can also be callable that transforms a pd.DataFrame.  This is useful
        for feature selection on transformations that are based on the feature set such 
        as center log-ratio or closure transformations in compositional data.
"""

_feature_importances_sem_docstring = """
    feature_importances_sem_:
        Pandas Series with the SEM of feature importance (comes from step 2)
"""

_get_remove_zero_weighted_features_docstring = """
    remove_zero_weighted_features_docstring: bool, default=True
        Remove zero weighted features from feature selection
"""

_threshold_docstring = _threshold_docstring.replace("default = 0.01", "default = 0.0")



# Functions
def remove_zero_weighted_features( # Make this easier to use. Way too specific right now...
    estimator, 
    X, 
    y, 
    selected_features,
    initial_feature_importances, 
    initial_feature_importances_sem,
    feature_selected_training_cv, 
    feature_selected_testing_score=np.nan,
    transformation=None,
    X_testing=None,
    y_testing=None,
    n_jobs=1,
    cv=3, 
    scorer="auto",
    feature_importances_from_cv=True,
    maximum_tries_to_remove_zero_weighted_features=100, 
    verbose=1,
    log=sys.stderr,
    i=0,
    ):
    # Determine scoring
    if scorer == "auto":
        if is_classifier(estimator):
            scorer = "accuracy"
            warnings.warn("Detected estimator as classifier.  Setting scorer to {}".format(scorer))
        if is_regressor(estimator):
            scorer = "neg_root_mean_squared_error"
            warnings.warn("Detected estimator as regressor.  Setting scoring to {}".format(scorer))
        assert scorer != "auto", "`estimator` not recognized as classifier or regressor by sklearn.  Are you using a sklearn-compatible estimator?"
    if isinstance(scorer, str):
        scorer = get_scorer(scorer)
        
    # Testing set
    testing_set_provided = check_testing_set(X.columns, X_testing, y_testing)

    # Initial feature importances
    feature_importances = initial_feature_importances.copy()
    feature_importances_sem = initial_feature_importances_sem.copy()
    feature_selected_training_score = feature_selected_training_cv.mean()
    

    initial_feature_importances_equal_zero = initial_feature_importances == 0

    n_features = len(selected_features)
    if verbose > 0:
        warnings.warn("Detected {}/{} zero-weighted features in final fitted model.  Refitting to remove zero-weighted features".format(initial_feature_importances_equal_zero.sum(), n_features))
    features = list(initial_feature_importances[~initial_feature_importances_equal_zero].index)
    for j in range(1, maximum_tries_to_remove_zero_weighted_features+1):
        X_query = X.loc[:,features]
        if all([
            transformation is not None,
            len(features) > 1,
        ]):
            X_query = transformation(X_query)
        # Cross validate
        try:
            cv_results = cross_validate(
                estimator,
                X_query,
                y,
                scoring=scorer, 
                n_jobs=n_jobs, 
                cv=cv,
                return_estimator=True,
            )
        except ValueError as e:

            # raise Exception("Cross-validation failed.\nNumber of samples: {}, Number of features: {}. Features: ".format(*X_query.shape, list(X_query.columns)))
            if not isinstance(cv, int):
                cv = len(cv)
            return {
                "selected_features":[], 
                "feature_importances":pd.Series([]), 
                "feature_importances_sem":pd.Series([]),
                "feature_selected_training_cv":np.asarray(cv*[np.nan]), 
                "feature_selected_testing_score":np.nan,
            }

        
        # Summarize feature/coeff importance for each cross validation fold
        if feature_importances_from_cv:
            feature_selected_model_cv = cv_results["test_score"]
            feature_importances_results = format_feature_importances_from_cv(
                cv_results=cv_results, 
                features=X_query.columns,
            )
        # Summarize feature/coeff importance for entire dataset
        else:
            feature_importances_results = format_feature_importances_from_data(
                estimator=estimator, 
                X=X_query, 
                y=y,
            )
        feature_selected_importances = feature_importances_results["mean"]
        feature_selected_importances_sem = feature_importances_results["sem"]
        
       
        mask_zero_weight_features = feature_selected_importances != 0

        if np.all(mask_zero_weight_features):
            updated_feature_selected_training_score = cv_results["test_score"].mean()

            # X_fs = transform(X=X_query, method=transformation, multiplicative_replacement=multiplicative_replacement, axis=1)
            if verbose > 0:

                # if j > 0:
                if updated_feature_selected_training_score > feature_selected_training_score:
                    msg = "[Success][Iteration={}, Try={}]: ☺ Removed all zero weighted features and the new training score improved from {} -> {}. The following {}/{} features remain: {}".format(i, j, feature_selected_training_score, updated_feature_selected_training_score, len(features), n_features, list(features))
                else:
                    msg = "[Success][Iteration={}, Try={}]: Removed all zero weighted features but the new training score declined from {} -> {}. The following {}/{} features remain: {}".format(i, j, feature_selected_training_score, updated_feature_selected_training_score, len(features), n_features, list(features))

                print(msg, file=log)
                    
            # Update these and stop:
            # selected_features, feature_importances, performance_drifts, feature_selected_training_score, feature_selected_testing_score
            selected_features = pd.Index(features)
            feature_importances = feature_selected_importances.loc[features]
            feature_importances_sem = feature_selected_importances_sem.loc[features]

            feature_selected_training_score = updated_feature_selected_training_score
            feature_selected_training_cv = cv_results["test_score"]

            # Testing set with updated features
            if testing_set_provided:
                estimator.fit(X_query, y)
                updated_feature_selected_testing_score = scorer(estimator, X_testing.loc[:,features], y_testing)
                if verbose > 0:
                    # if j > 0:
                    if updated_feature_selected_testing_score > feature_selected_testing_score:
                        msg = "[Success][Iteration={}, Try={}]: ☺ New testing score improved from {} -> {}".format(i, j, feature_selected_testing_score, updated_feature_selected_testing_score)
                    else:
                        msg = "[Success][Iteration={}, Try={}]: New testing score declined from {} -> {}".format(i, j, feature_selected_testing_score, updated_feature_selected_testing_score)
                    print(msg, file=log)
                feature_selected_testing_score = updated_feature_selected_testing_score

            break
        else:
            if verbose > 2:
                if j > 0:
                    print("[...][Iteration={}, Try={}]: Removing {} features as they have zero weight in fitted model: {}".format(i, j, len(mask_zero_weight_features) - np.sum(mask_zero_weight_features), X_query.columns[~mask_zero_weight_features].tolist()), file=log)
            features = X_query.columns[mask_zero_weight_features].tolist()

    return {
        "selected_features":selected_features, 
        "feature_importances":feature_importances, 
        "feature_importances_sem":feature_importances_sem,
        "feature_selected_training_cv":feature_selected_training_cv, 
        "feature_selected_testing_score":feature_selected_testing_score,
    }


# Classes
class ClairvoyanceBaseRecursiveSelector(BaseSelector):
    """
    This class has been modified from the BaseRecursiveSelector class from 
    the Feature Engine Python package (https://github.com/feature-engine/feature_engine)
    
    Shared functionality for recursive selectors.

    Parameters
    ----------
    estimator: object
        A Scikit-learn estimator for regression or classification.
        The estimator must have either a `feature_importances` or `coef_` attribute
        after fitting.

    variables: str or list, default=None
        The list of variable to be evaluated. If None, the transformer will evaluate
        all numerical features in the dataset.

    scoring: str, default='roc_auc'
        Desired metric to optimise the performance of the estimator. Comes from
        sklearn.metrics. See the model evaluation documentation for more options:
        https://scikit-learn.org/stable/modules/model_evaluation.html

    threshold: float, int, default = 0.01
        The value that defines if a feature will be kept or removed. Note that for
        metrics like roc-auc, r2_score and accuracy, the thresholds will be floats
        between 0 and 1. For metrics like the mean_square_error and the
        root_mean_square_error the threshold can be a big number.
        The threshold must be defined by the user. Bigger thresholds will select less
        features.

    cv: int, cross-validation generator or an iterable, default=3
        Determines the cross-validation splitting strategy. Possible inputs for cv are:

            - None, to use cross_validate's default 5-fold cross validation

            - int, to specify the number of folds in a (Stratified)KFold,

            - CV splitter
                - (https://scikit-learn.org/stable/glossary.html#term-CV-splitter)

            - An iterable yielding (train, test) splits as arrays of indices.

        For int/None inputs, if the estimator is a classifier and y is either binary or
        multiclass, StratifiedKFold is used. In all other cases, KFold is used. These
        splitters are instantiated with `shuffle=False` so the splits will be the same
        across calls. For more details check Scikit-learn's `cross_validate`'s
        documentation.

    confirm_variables: bool, default=False
        If set to True, variables that are not present in the input dataframe will be
        removed from the list of variables. Only used when passing a variable list to
        the parameter `variables`. See parameter variables for more details.

    {transformation}
    
    {remove_zero_weighted_features}
        
    Attributes
    ----------
    initial_model_performance_:
        Performance of the model trained using the original dataset.

    feature_importances_:
        Pandas Series with the feature importance (comes from step 2)

    feature_importances_sem_:
        Pandas Series with the SEM of feature importance (comes from step 2)

    performance_drifts_:
        Dictionary with the performance drift per examined feature (comes from step 5).

    features_to_drop_:
        List with the features to remove from the dataset.

    variables_:
        The variables that will be considered for the feature selection.

    feature_names_in_:
        List with the names of features seen during `fit`.

    n_features_in_:
        The number of features in the train set used in fit.

    Methods
    -------
    fit:
        Find the important features.
    """.format(
        transformation=_get_transformation_docstring,
        remove_zero_weighted_features=_get_remove_zero_weighted_features_docstring,
        )
    
    def __init__(
        self,
        estimator,
        scoring: str = "auto",
        cv=3,
        n_jobs=1,
        threshold: Union[int, float] = 0.0,
        variables: Variables = None,
        confirm_variables: bool = False,
        feature_importances_from_cv=True,
        transformation = None,
        remove_zero_weighted_features=True,
        maximum_tries_to_remove_zero_weighted_features=100,
        verbose=1,
    ):

        if not isinstance(threshold, (int, float)):
            raise ValueError("threshold can only be integer or float")

        if isinstance(transformation, str):
            assert_acceptable_arguments(transformation, {"closure", "clr", "clr_with_multiplicative_replacement"})
            transformation = globals()[transformation]

        if transformation is not None:
            assert hasattr(transformation, "__call__"), "If `transform` is not None, then it must be a callable function"

        if scoring == "auto":
            if is_classifier(estimator):
                scoring = "accuracy"
                warnings.warn("Detected estimator as classifier.  Setting scoring to {}".format(scoring))
            if is_regressor(estimator):
                scoring = "neg_root_mean_squared_error"
                warnings.warn("Detected estimator as regressor.  Setting scoring to {}".format(scoring))
            assert scoring != "auto", "`estimator` not recognized as classifier or regressor by sklearn.  Are you using a sklearn-compatible estimator?"
            
        super().__init__(confirm_variables)
        self.variables = _check_variables_input_value(variables)
        self.estimator = estimator
        self.scoring = scoring
        self.threshold = threshold
        self.cv = cv
        self.n_jobs = n_jobs
        self.feature_importances_from_cv = feature_importances_from_cv
        self.transformation = transformation
        self.remove_zero_weighted_features=remove_zero_weighted_features
        self.maximum_tries_to_remove_zero_weighted_features = maximum_tries_to_remove_zero_weighted_features
        self.verbose = verbose


    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Find initial model performance. Sort features by importance.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
           The input dataframe

        y: array-like of shape (n_samples)
           Target variable. Required to train the estimator.
        """

        # check input dataframe
        X, y = check_X_y(X, y)

        if self.variables is None:
            self.variables_ = find_numerical_variables(X)
        else:
            if self.confirm_variables is True:
                variables_ = retain_variables_if_in_df(X, self.variables)
                self.variables_ = check_numerical_variables(X, variables_)
            else:
                self.variables_ = check_numerical_variables(X, self.variables)

        # check that there are more than 1 variable to select from
        self._check_variable_number()

        # save input features
        X = X[self.variables_]
        self._get_feature_names_in(X)
            
        # train model with all features and cross-validation
        if X.shape[1] > 1:
            initial_model_cv_results = cross_validate(
                self.estimator,
                X if self.transformation is None else self.transformation(X),
                y,
                cv=self.cv,
                scoring=self.scoring,
                return_estimator=True,
                n_jobs=self.n_jobs,
            )
        else:
            initial_model_cv_results = cross_validate(
                self.estimator,
                X,
                y,
                cv=self.cv,
                scoring=self.scoring,
                return_estimator=True,
                n_jobs=self.n_jobs,
            )

        # store initial model performance
        self.initial_model_cv_ = initial_model_cv_results["test_score"]
        self.initial_model_performance_ = initial_model_cv_results["test_score"].mean()

        # Summarize feature/coeff importance for each cross validation fold
        if self.feature_importances_from_cv:
            feature_importances_results = format_feature_importances_from_cv(
                cv_results=initial_model_cv_results, 
                features=X.columns,
            )
        # Summarize feature/coeff importance for entire dataset
        else:
            feature_importances_results = format_feature_importances_from_data(
                estimator=self.estimator, 
                X=X if self.transformation is None else self.transformation(X), 
                y=y,
            )

        self.initial_feature_importances_ = feature_importances_results["mean"]
        self.initial_feature_importances_sem_ = feature_importances_results["sem"]

        assert self.initial_feature_importances_.abs().max() > 0, "Largest feature importances is zero.  Something went wrong when training model."
        assert not np.all(self.initial_feature_importances_ == 0), "All feature importances weight is zero.  Something went wrong when training model."

        return X, y

    def _more_tags(self):
        tags_dict = _return_tags()
        tags_dict["variables"] = "numerical"
        tags_dict["requires_y"] = True
        # add additional test that fails
        tags_dict["_xfail_checks"][
            "check_parameters_default_constructible"
        ] = "transformer has 1 mandatory parameter"
        tags_dict["_xfail_checks"]["check_estimators_nan_inf"] = "transformer allows NA"

        msg = "transformers need more than 1 feature to work"
        tags_dict["_xfail_checks"]["check_fit2d_1feature"] = msg

        return tags_dict


@Substitution(
    estimator=_estimator_docstring,
    scoring=_scoring_docstring,
    threshold=_threshold_docstring,
    cv=_cv_docstring,
    variables=_variables_numerical_docstring,
    confirm_variables=_confirm_variables_docstring,
    initial_model_performance_=_initial_model_performance_docstring,
    feature_importances_=_feature_importances_docstring,
    feature_importances_sem_=_feature_importances_sem_docstring,
    performance_drifts_=_performance_drifts_docstring,
    features_to_drop_=_features_to_drop_docstring,
    variables_=_variables_attribute_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    fit=_fit_docstring,
    transform=_transform_docstring,
    fit_transform=_fit_transform_docstring,
    get_support=_get_support_docstring,
    transformation=_get_transformation_docstring,
    remove_zero_weighted_features=_get_remove_zero_weighted_features_docstring,
)
class ClairvoyanceRecursiveFeatureAddition(ClairvoyanceBaseRecursiveSelector):
    """
    This class has been modified from the RecursiveFeatureAddition class from 
    the Feature Engine Python package (https://github.com/feature-engine/feature_engine)
    
    ClairvoyanceRecursiveFeatureAddition() selects features following a recursive addition process.

    The process is as follows:

    1. Train an estimator using all the features.

    2. Rank the features according to their importance derived from the estimator.

    3. Train an estimator with the most important feature and determine performance.

    4. Add the second most important feature and train a new estimator.

    5. Calculate the difference in performance between estimators.

    6. If the performance increases beyond the threshold, the feature is kept.

    7. Repeat steps 4-6 until all features have been evaluated.
    
    8. Remove zero weighted features
    
    If transformations are provided, then each time a feature is removed or added the data is transformed.

    Model training and performance calculation can be performed using entire dataset or cross-validation.

    More details in the :ref:`User Guide <recursive_addition>`.

    Parameters
    ----------
    {estimator}

    {variables}

    {scoring}

    {threshold}

    {cv}

    {confirm_variables}

    {transformation}
    
    {remove_zero_weighted_features}

    Attributes
    ----------
    {initial_model_performance_}

    {feature_importances_}

    {feature_importances_sem_}

    {performance_drifts_}

    {features_to_drop_}

    {variables_}

    {feature_names_in_}

    {n_features_in_}


    Methods
    -------
    {fit}

    {fit_transform}

    {get_support}

    {transform}

    Examples
    --------

    >>> import pandas as pd
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from feature_engine.selection import RecursiveFeatureAddition
    >>> X = pd.DataFrame(dict(x1 = [1000,2000,1000,1000,2000,3000],
    >>>                     x2 = [2,4,3,1,2,2],
    >>>                     x3 = [1,1,1,0,0,0],
    >>>                     x4 = [1,2,1,1,0,1],
    >>>                     x5 = [1,1,1,1,1,1]))
    >>> y = pd.Series([1,0,0,1,1,0])
    >>> rfa = RecursiveFeatureAddition(RandomForestClassifier(random_state=42), cv=2)
    >>> rfa.fit_transform(X, y)
       x2  x4
    0   2   1
    1   4   2
    2   3   1
    3   1   1
    4   2   0
    5   2   1
    """

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Find the important features. Note that the selector trains various models at
        each round of selection, so it might take a while.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
           The input dataframe

        y: array-like of shape (n_samples)
           Target variable. Required to train the estimator.
        """

        X, y = super().fit(X, y)

        # Sort the feature importance values decreasingly
        self.initial_feature_importances_.sort_values(ascending=False, inplace=True)
        feature_importances_tmp = self.initial_feature_importances_.copy()
        if self.remove_zero_weighted_features:
            feature_importances_tmp = feature_importances_tmp[lambda x: abs(x) > 0]
            n_features_after_zero_removal = feature_importances_tmp.size
            n_features_initial = self.initial_feature_importances_.size
            if n_features_initial > n_features_after_zero_removal:
                warnings.warn("remove_zero_weighted_features=True and removed {}/{} features".format((n_features_initial - n_features_after_zero_removal), n_features_initial))
                
        # Extract most important feature from the ordered list of features
        first_most_important_feature = list(feature_importances_tmp.index)[0]

        # Run baseline model using only the most important feature
        X_1 = X[first_most_important_feature].to_frame()
        baseline_model_cv_results = cross_validate(
            self.estimator,
            X_1,
            y,
            cv=self.cv,
            scoring=self.scoring,
            return_estimator=True,
            n_jobs=self.n_jobs,
        )

        # Save baseline model performance
        baseline_model_performance = baseline_model_cv_results["test_score"].mean()
        feature_selected_model_cv = baseline_model_cv_results["test_score"].copy()
        
        # Summarize feature/coeff importance for each cross validation fold
        if self.feature_importances_from_cv:
            feature_importances_results = format_feature_importances_from_cv(
                cv_results=baseline_model_cv_results, 
                features=X_1.columns,
            )
        # Summarize feature/coeff importance for entire dataset
        else:
            feature_importances_results = format_feature_importances_from_data(
                estimator=self.estimator, 
                X=X_1, # This isn't needed because it's always going to be a single feature :if self.transformation is None else self.transformation(X_1), 
                y=y,
            )

        feature_selected_importances = feature_importances_results["mean"]
        feature_selected_importances_sem = feature_importances_results["sem"]

        # list to collect selected features
        # It is initialized with the most important feature
        _selected_features = [first_most_important_feature]

        # dict to collect features and their performance_drift
        # It is initialized with the performance drift of
        # the most important feature
        self.performance_drifts_ = {first_most_important_feature: 0}

        # loop over the ordered list of features by feature importance starting
        # from the second element in the list.
        for feature in tqdm(list(feature_importances_tmp.index)[1:], desc="Recursive feature addition"):
            X_tmp = X[_selected_features + [feature]]

            # Add feature and train new model
            query_model_cv_results = cross_validate(
                self.estimator,
                X_tmp if self.transformation is None else self.transformation(X_tmp),
                y,
                cv=self.cv,
                scoring=self.scoring,
                return_estimator=True,
            )

            # assign new model performance
            query_model_performance = query_model_cv_results["test_score"].mean()

            # Calculate performance drift
            performance_drift = query_model_performance - baseline_model_performance

            # Save feature and performance drift
            self.performance_drifts_[feature] = performance_drift

            # Update selected features if performance improves
            if performance_drift > self.threshold:
                # add feature to the list of selected features
                _selected_features.append(feature)

                # Update new baseline model performance
                baseline_model_performance = query_model_performance
                feature_selected_model_cv = query_model_cv_results["test_score"]
                
                # Summarize feature/coeff importance for each cross validation fold
                if self.feature_importances_from_cv:
                    feature_importances_results = format_feature_importances_from_cv(
                        cv_results=query_model_cv_results, 
                        features=X_tmp.columns,
                    )
                # Summarize feature/coeff importance for entire dataset
                else:
                    feature_importances_results = format_feature_importances_from_data(
                        estimator=self.estimator, 
                        X=X_tmp if self.transformation is None else self.transformation(X_tmp), 
                        y=y,
                    )
                feature_selected_importances = feature_importances_results["mean"]
                feature_selected_importances_sem = feature_importances_results["sem"]

        # Remove zero-weighted features
        if self.remove_zero_weighted_features:
            if np.any(feature_selected_importances == 0):
                cleaned_results = remove_zero_weighted_features(
                    estimator=self.estimator, 
                    X=X_tmp.loc[:,_selected_features], 
                    y=y, 
                    selected_features=_selected_features,
                    initial_feature_importances=feature_selected_importances.loc[_selected_features], 
                    initial_feature_importances_sem=feature_selected_importances_sem.loc[_selected_features], 
                    feature_selected_training_cv=feature_selected_model_cv, 
                    feature_selected_testing_score=np.nan,
                    transformation=self.transformation,
                    X_testing=None,
                    y_testing=None,
                    n_jobs=self.n_jobs,
                    cv=self.cv, 
                    scorer=self.scoring,
                    feature_importances_from_cv=self.feature_importances_from_cv,
                    maximum_tries_to_remove_zero_weighted_features=self.maximum_tries_to_remove_zero_weighted_features, 
                    verbose=self.verbose,
                    log=sys.stderr,
                    )
                # ['selected_features', 'feature_importances', 'feature_importances_sem', 'feature_selected_training_cv', 'feature_selected_testing_performance']
                if len(cleaned_results["selected_features"]) >= 1:
                    _selected_features = cleaned_results["selected_features"]
                    feature_selected_importances = cleaned_results["feature_importances"]
                    feature_selected_importances_sem = cleaned_results["feature_importances_sem"]
                    feature_selected_model_cv = cleaned_results["feature_selected_training_cv"]
                
        self.features_to_drop_ = [
            f for f in self.variables_ if f not in _selected_features
        ]
        
        
        a = feature_selected_importances.size
        b = len(_selected_features)
        assert a == b, "Number of selected feature importances (N={}) is different than number of selected features (N={})".format(a, b)
        if a == 0:
            raise ValueError("0 features selected using current parameters.  Try preprocessing feature matrix, using a different transformation, or a different estimator.")
        self.selected_features_ = list(_selected_features)
        self.feature_selected_importances_ = feature_selected_importances.loc[_selected_features]
        self.feature_selected_importances_sem_ = feature_selected_importances_sem.loc[_selected_features]
        self.feature_selected_model_cv_ = feature_selected_model_cv
        self.feature_selected_model_performance_ = feature_selected_model_cv.mean()

        return self

@Substitution(
    estimator=_estimator_docstring,
    scoring=_scoring_docstring,
    threshold=_threshold_docstring,
    cv=_cv_docstring,
    variables=_variables_numerical_docstring,
    confirm_variables=_confirm_variables_docstring,
    initial_model_performance_=_initial_model_performance_docstring,
    feature_importances_=_feature_importances_docstring,
    performance_drifts_=_performance_drifts_docstring,
    features_to_drop_=_features_to_drop_docstring,
    variables_=_variables_attribute_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    fit=_fit_docstring,
    transform=_transform_docstring,
    fit_transform=_fit_transform_docstring,
    get_support=_get_support_docstring,
    transformation=_get_transformation_docstring,
    remove_zero_weighted_features=_get_remove_zero_weighted_features_docstring,
)
class ClairvoyanceRecursiveFeatureElimination(ClairvoyanceBaseRecursiveSelector):
    """
    This class has been modified from the RecursiveFeatureElimination class from 
    the Feature Engine Python package (https://github.com/feature-engine/feature_engine)
    
    ClairvoyanceRecursiveFeatureElimination() selects features following a recursive elimination
    process.

    The process is as follows:

    1. Train an estimator using all the features.

    2. Rank the features according to their importance derived from the estimator.

    3. Remove the least important feature and fit a new estimator.

    4. Calculate the performance of the new estimator.

    5. Calculate the performance difference between the new and original estimator.

    6. If the performance drop is below the threshold the feature is removed.

    7. Repeat steps 3-6 until all features have been evaluated.
    
    8. Remove zero weighted features
    
    If transformations are provided, then each time a feature is removed or added the data is transformed.

    Model training and performance calculation can be performed using entire dataset or cross-validation.

    More details in the :ref:`User Guide <recursive_elimination>`.

    Parameters
    ----------
    {estimator}

    {variables}

    {scoring}

    {threshold}

    {cv}

    {confirm_variables}

    {transformation}
    
    {remove_zero_weighted_features}

    Attributes
    ----------
    {initial_model_performance_}

    {feature_importances_}

    {performance_drifts_}

    {features_to_drop_}

    {variables_}

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    {fit}

    {fit_transform}

    {get_support}

    {transform}

    Examples
    --------

    >>> import pandas as pd
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from feature_engine.selection import RecursiveFeatureElimination
    >>> X = pd.DataFrame(dict(x1 = [1000,2000,1000,1000,2000,3000],
    >>>                     x2 = [2,4,3,1,2,2],
    >>>                     x3 = [1,1,1,0,0,0],
    >>>                     x4 = [1,2,1,1,0,1],
    >>>                     x5 = [1,1,1,1,1,1]))
    >>> y = pd.Series([1,0,0,1,1,0])
    >>> rfe = RecursiveFeatureElimination(RandomForestClassifier(random_state=2), cv=2)
    >>> rfe.fit_transform(X, y)
       x2
    0   2
    1   4
    2   3
    3   1
    4   2
    5   2
    """

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Find the important features. Note that the selector trains various models at
        each round of selection, so it might take a while.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
           The input dataframe
        y: array-like of shape (n_samples)
           Target variable. Required to train the estimator.
        """

        X, y = super().fit(X, y)

        # Sort the feature importance values increasingly
        self.initial_feature_importances_.sort_values(ascending=True, inplace=True)
        feature_importances_tmp = self.initial_feature_importances_.copy()
        if self.remove_zero_weighted_features:
            feature_importances_tmp = feature_importances_tmp[lambda x: abs(x) > 0]
            n_features_after_zero_removal = feature_importances_tmp.size
            n_features_initial = self.initial_feature_importances_.size
            if n_features_initial > n_features_after_zero_removal:
                warnings.warn("remove_zero_weighted_features=True and removed {}/{} features".format((n_features_initial - n_features_after_zero_removal), n_features_initial))
            
        # to collect selected features
        _selected_features = []

        # temporary copy where we will remove features recursively
        variables = self.variables_
        if self.remove_zero_weighted_features:
            variables = list(set(feature_importances_tmp.index) & set(variables))
        X_tmp = X[variables].copy()

        # we need to update the performance as we remove features
        baseline_model_performance = self.initial_model_performance_
        feature_selected_model_cv = self.initial_model_cv_.copy()
        feature_selected_importances = self.initial_feature_importances_.copy()
        feature_selected_importances_sem = self.initial_feature_importances_sem_.copy()

        # dict to collect features and their performance_drift after shuffling
        self.performance_drifts_ = {}

        # evaluate every feature, starting from the least important
        # remember that feature_importances_ is ordered already
        for feature in tqdm(list(feature_importances_tmp.index), desc="Recursive feature elimination"):

            # if there is only 1 feature left
            if X_tmp.shape[1] == 1:
                self.performance_drifts_[feature] = 0
                _selected_features.append(feature)
                break

            # remove feature and train new model
                
            X_query = X_tmp.drop(columns=feature)
            query_model_cv_results = cross_validate(
                self.estimator,
                X_query if self.transformation is None else self.transformation(X_query),
                y,
                cv=self.cv,
                scoring=self.scoring,
                return_estimator=False,
                n_jobs=self.n_jobs,
            )

            # assign new model performance
            query_model_performance = query_model_cv_results["test_score"].mean()

            # Calculate performance drift
            performance_drift = baseline_model_performance - query_model_performance

            # Save feature and performance drift
            self.performance_drifts_[feature] = performance_drift

            # Update selected features if performance improves
            if performance_drift > self.threshold:
                _selected_features.append(feature)

            else:
                # remove feature and adjust initial performance
                X_tmp = X_tmp.drop(columns=feature)

                baseline_model_cv_results = cross_validate(
                    self.estimator,
                    X_tmp if self.transformation is None else self.transformation(X_tmp),
                    y,
                    cv=self.cv,
                    return_estimator=False,
                    scoring=self.scoring,
                    n_jobs=self.n_jobs,
                )

                # store initial model performance
                baseline_model_performance = baseline_model_cv_results["test_score"].mean()

        
        # Fit final model
        if len(_selected_features) < X.shape[1]:
            X_query = X_tmp.loc[:,_selected_features]
            query_model_cv_results = cross_validate(
                self.estimator,
                X_query if self.transformation is None else self.transformation(X_query),
                y,
                cv=self.cv,
                scoring=self.scoring,
                return_estimator=True,
                n_jobs=self.n_jobs,
            )
            feature_selected_model_cv = query_model_cv_results["test_score"]
            
            # Summarize feature/coeff importance for each cross validation fold
            if self.feature_importances_from_cv:
                feature_importances_results = format_feature_importances_from_cv(
                    cv_results=query_model_cv_results, 
                    features=X_query.columns,
                )
            # Summarize feature/coeff importance for entire dataset
            else:
                feature_importances_results = format_feature_importances_from_data(
                    estimator=self.estimator, 
                    X=X_query if self.transformation is None else self.transformation(X_query), 
                    y=y,
                )
            feature_selected_importances = feature_importances_results["mean"]
            feature_selected_importances_sem = feature_importances_results["sem"]

        # Remove zero-weighted features
        if self.remove_zero_weighted_features:
            if np.any(feature_selected_importances == 0):
                cleaned_results = remove_zero_weighted_features(
                    estimator=self.estimator, 
                    X=X_tmp.loc[:,_selected_features], 
                    y=y, 
                    selected_features=_selected_features,
                    initial_feature_importances=feature_selected_importances.loc[_selected_features], 
                    initial_feature_importances_sem=feature_selected_importances_sem.loc[_selected_features], 
                    feature_selected_training_cv=feature_selected_model_cv, 
                    feature_selected_testing_score=np.nan,
                    transformation=self.transformation,
                    X_testing=None,
                    y_testing=None,
                    n_jobs=self.n_jobs,
                    cv=self.cv, 
                    scorer=self.scoring,
                    feature_importances_from_cv=self.feature_importances_from_cv,
                    maximum_tries_to_remove_zero_weighted_features=self.maximum_tries_to_remove_zero_weighted_features, 
                    verbose=self.verbose,
                    log=sys.stderr,
                    )
                # ['selected_features', 'feature_importances', 'feature_importances_sem', 'feature_selected_training_cv', 'feature_selected_testing_performance']
            
                _selected_features = cleaned_results["selected_features"]
                feature_selected_importances = cleaned_results["feature_importances"]
                feature_selected_importances_sem = cleaned_results["feature_importances_sem"]
                feature_selected_model_cv = cleaned_results["feature_selected_training_cv"]


        # Features to drop
        self.features_to_drop_ = [
            f for f in self.variables_ if f not in _selected_features
        ]
            
        a = feature_selected_importances.size
        b = len(_selected_features)
        assert a == b, "Number of selected feature importances (N={}) is different than number of selected features (N={})".format(a, b)
        self.selected_features_ = list(_selected_features)
        self.feature_selected_importances_ = feature_selected_importances.loc[_selected_features]
        self.feature_selected_importances_sem_ = feature_selected_importances_sem.loc[_selected_features]
        self.feature_selected_model_cv_ = feature_selected_model_cv
        self.feature_selected_model_performance_ = feature_selected_model_cv.mean()

        return self

