# -*- coding: utf-8 -*-

# Built-ins
from ast import Or
import os, sys, itertools, argparse, time, copy, warnings
from collections import OrderedDict
# from multiprocessing import cpu_count

# PyData
import pandas as pd
import numpy as np
import xarray as xr

# Machine learning
from scipy import stats
from sklearn.metrics import get_scorer, make_scorer
from sklearn.model_selection import cross_val_score, train_test_split, RepeatedStratifiedKFold, StratifiedKFold, RepeatedKFold, KFold
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning

# Clairvoyance
from .utils import *
from .transformations import legacy_transform as transform

# Plotting
def plot_scores_line(
    average_scores:pd.Series, 
    sem:pd.Series, 
    testing_scores:pd.Series=None, 
    horizontal_lines=True,
    vertical_lines="auto",
    title=None,
    figsize=(13,3), 
    linecolor="black",
    errorcolor="gray",
    testing_linecolor="red",
    style="seaborn-v0_8-white",
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
        if testing_scores is not None:
            if not np.all(testing_scores.isnull()):
                testing_scores.plot(ax=ax, color=testing_linecolor, label="Testing score", **kwargs)

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
    style="seaborn-v0_8-white", 
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
    style="seaborn-v0_8-white", 
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
    # Testing_scores:pd.Series=None,
    min_features:int=None,
    max_features:int=None,
    min_score:float=None,
    max_score:float=None,
    ax=None,
    color="darkslategray",
    color_testing="red",
    linewidth=0.618,
    alpha=0.618,
    edgecolor="black", 
    style="seaborn-v0_8-white",
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
    testing_scores:pd.Series=None,
    min_features:int=None,
    max_features:int=None,
    min_score:float=None,
    max_score:float=None,
    ax=None,
    color="darkslategray",
    linewidth=0.618,
    alpha=0.618,
    edgecolor="black", 
    style="seaborn-v0_8-white",
    figsize=(8,5),
    title=None,
    xlabel="Average Score",
    ylabel="Testing Score",

    feature_to_size_function = "auto",

    xtick_rotation=0,
    show_xgrid=True,
    show_ygrid=True,
    # show_zgrid=True,

    show_xticks=True, 
    show_legend=True,
    legend_markers=["min", "25%", "50%", "75%", "max"],
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
    assert isinstance(testing_scores, pd.Series)
    assert set(legend_markers) <= set(["min", "25%", "50%", "75%", "max"]), 'legend_markers must be a subset of ["min", "25%", "50%", "75%", "max"]'
    legend_markers = sorted(legend_markers, key=lambda x:["min", "25%", "50%", "75%", "max"].index(x))

    assert np.all(number_of_features.index == average_scores.index)
    assert np.all(number_of_features.index == testing_scores.index)

    df = pd.DataFrame([number_of_features, average_scores, testing_scores], index=["number_of_features", "average_scores", "testing_scores"]).T

    if min_features:
        df = df.query("number_of_features >= {}".format(min_features))
    if max_features:
        df = df.query("number_of_features <= {}".format(max_features))
    if min_score:
        df = df.query("average_scores >= {}".format(min_score))
    if max_score:
        df = df.query("average_scores <= {}".format(max_score))
    if min_score:
        df = df.query("testing_scores >= {}".format(min_score))
    if max_score:
        df = df.query("testing_scores <= {}".format(max_score))

    if feature_to_size_function == "auto":
        def feature_to_size_function(n):
            return 100/n
    assert hasattr(feature_to_size_function, "__call__")
    marker_sizes = feature_to_size_function(number_of_features)

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


        # ax.scatter(xs=df["number_of_features"].values, ys=df["average_scores"].values, zs=df["testing_scores"], edgecolor=edgecolor, alpha=alpha, linewidth=linewidth, color=color, **kwargs)
        ax.scatter(x=df["average_scores"].values, y=df["testing_scores"], s=marker_sizes, edgecolor=edgecolor, alpha=alpha, linewidth=linewidth, color=color, **kwargs)


        ax.set_xlabel(xlabel, **_xlabel_kws)
        ax.set_ylabel(ylabel, **_ylabel_kws)
        # ax.set_zlabel(zlabel, **_zlabel_kws)

        # ax.set_yticklabels(map(lambda x:"%0.2f"%x, ax.get_yticks()), **_yticklabel_kws)
        if title:
            ax.set_title(title, **_title_kws)
        if show_legend:
            legendary_features = number_of_features.describe()[legend_markers].astype(int)
            
            legend_elements = list()
            for i,n in legendary_features.items():
                # marker = plt.Line2D([], [], color=color, marker='o', linestyle='None', markersize=feature_to_size_function(n), label="$N$ = %d"%(n))
                # legend_elements.append(marker)
                i = i.replace("%","\%")
                legend_marker = ax.scatter([], [], s=feature_to_size_function(n), label="$N_{%s}$ = %d"%(i,n), color=color, marker='o')
                legend_elements.append(legend_marker)


            legend = ax.legend(handles=legend_elements,  title="$N_{Features}$",  markerscale=1, **_legend_kws)
            
        if show_xgrid:
            ax.xaxis.grid(True)
        if show_ygrid:
            ax.yaxis.grid(True)
        # if show_zgrid:
        #     ax.zaxis.grid(True)
        return fig, ax


def recursive_feature_addition(
    estimator, 
    X:pd.DataFrame, 
    y:pd.Series, 
    scorer,
    initial_feature_weights:pd.Series, 
    initial_feature_weights_name:str="initial_feature_weights",
    feature_weight_attribute:str="auto",
    transformation=None,
    multiplicative_replacement="auto",
    metric=np.nanmean, 
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
    progress_message="Recursive feature addition",
    remove_zero_weighted_features=True,
    maximum_tries_to_remove_zero_weighted_features=1000,
    X_testing:pd.DataFrame=None,
    y_testing:pd.Series=None,
    # optimize_testing_score = "auto",
    ) -> pd.Series:

    assert len(set(X.columns)) == X.shape[1], "Cannot have duplicate feature names in `X`"
    if additional_feature_penalty is None:
        additional_feature_penalty = lambda number_of_features: 0.0

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
        assert np.all(X_testing.columns == X.columns), "X_testing.columns and X.columns must have the same ordering"
        testing_set_provided = True
    # if optimize_testing_score == "auto":
    #     if testing_set_provided:
    #         optimize_testing_score = True
    #     else:
    #         optimize_testing_score = False

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
    testing_scores = OrderedDict()

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
        X_rfa = X.loc[:,features]

        continue_algorithm = True

        # Transform features (if transformation = None, then there is no transformation)
        if X_rfa.shape[1] > 1: 
            X_rfa = transform(X=X_rfa, method=transformation, multiplicative_replacement=multiplicative_replacement, axis=1)
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
                    mask_zero_weight_features = _w != 0

                    if np.all(mask_zero_weight_features):
                        X_rfa = transform(X=X_query, method=transformation, multiplicative_replacement=multiplicative_replacement, axis=1)
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
                scores = cross_val_score(estimator=estimator, X=X_rfa, y=y, cv=cv_splits, n_jobs=n_jobs, scoring=scorer)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    average_score = np.nanmean(scores)
                history[i] = scores #{"average_score":average_score, "sem":sem}

                #! ---
                # Testing Score
                query_score = average_score
                testing_score = np.nan
                if testing_set_provided:
                    estimator.fit(
                        X=X_rfa, 
                        y=y,
                    )
                    
                    # Transform features (if transformation = None, then there is no transformation)
                    X_testing_rfa = transform(X=X_testing.loc[:,features], method=transformation, multiplicative_replacement=multiplicative_replacement, axis=1)
                    testing_score = scorer(estimator=estimator, X=X_testing_rfa, y_true=y_testing)
                    testing_scores[i] = testing_score
                    # if optimize_testing_score:
                    #     query_score = testing_score
                #! ---

                # Add penalties to score target
                penalty_adjusted_score_target = (best_score + minimum_improvement_in_score + additional_feature_penalty(len(features)))
                
                if query_score <= penalty_adjusted_score_target:
                    if verbose > 1:
                        # if optimize_testing_score:
                        #     print("Current iteration {} of N={} features has not improved score: Testing Score[{} ≤ {}]".format(i, len(features), testing_score, best_score), file=log)
                        # else:
                        print("Current iteration {} of N={} features has not improved score: Average Score[{} ≤ {}]".format(i, len(features), average_score, best_score), file=log)

                    no_progress += 1
                else:
                    # if optimize_testing_score:
                    #     if verbose > 0:
                    #         print("Updating best score with N={} features : Testing Score[{} -> {}]".format(len(features), best_score, testing_score), file=log)
                    #     best_score = testing_score
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
    # if testing_set_provided:
    #     history[("testing","score")] = pd.Series(testing_scores)
    history.index = feature_tuples
    history.index.name = "features"
    
    # Summary
    average_scores = history.mean(axis=1)
    sems = history.sem(axis=1)
    if testing_set_provided:
        testing_scores = pd.Series(testing_scores)
        testing_scores.index = feature_tuples
    else:
        testing_scores = pd.Series([np.nan]*len(feature_tuples), index=feature_tuples)
    
    history.insert(loc=0, column=("summary", "number_of_features"),value = history.index.map(len))
    history.insert(loc=1, column=("summary", "average_score"),value = average_scores)
    history.insert(loc=2, column=("summary", "sem"),value = sems)
    history.insert(loc=3, column=("summary", "testing_score"), value = testing_scores)
    history.insert(loc=4, column=("summary", "∆(testing_score-average_score)"), value = average_scores - testing_scores)

    history.columns = pd.MultiIndex.from_tuples(history.columns)

    
    # Highest scoring features (not necessarily the best since there can be many features added with minimal gains)
    # if optimize_testing_score:
    #     highest_score = history[("summary", "testing_score")].max()
    #     highest_scoring_features = list(history.loc[history[("summary", "testing_score")] == highest_score].sort_values(
    #         by=[("summary", "average_score"), ("summary", "number_of_features")], 
    #         ascending=[False, less_features_is_better]).index[0])
    # else:
    highest_score = history[("summary", "average_score")].max()
    try:
        highest_scoring_features = list(history.loc[history[("summary", "average_score")] == highest_score, ("summary", "number_of_features")].sort_values(ascending=less_features_is_better).index[0])
        
    
        # # Best results
        # if optimize_testing_score:
        #     # best_features = list(history.loc[history[("summary", "testing_score")] == best_score].sort_values(
        #     #     by=[("summary", "average_score"), ("summary", "number_of_features")], 
        #     #     ascending=[False, less_features_is_better]).index[0])
        # else:
        #     best_features = list(history.loc[history[("summary", "average_score")] == best_score, ("summary", "number_of_features")].sort_values(ascending=less_features_is_better).index[0])

        if testing_set_provided:
            best_features = list(history.sort_values(
                by=[("summary", "testing_score"), ("summary", "average_score"), ("summary", "number_of_features")], 
                ascending=[False, False, less_features_is_better]).index[0])

        else:
            best_features = list(history.loc[history[("summary", "average_score")] == best_score].sort_values(
                by=[("summary", "testing_score"), ("summary", "average_score"), ("summary", "number_of_features")], 
                ascending=[False, False, less_features_is_better]).index[0])
        
        best_estimator_sem = history.loc[[tuple(best_features)],("summary","sem")].values[0]
        best_estimator_testing_score = history.loc[[tuple(best_features)],("summary","testing_score")].values[0]

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
            testing_scores=testing_scores,
            best_estimator_testing_score=best_estimator_testing_score,
            ),
            name="recursive_feature_elimination",
        )
    except IndexError:
        return pd.Series(
            dict(
            history=history, 
            best_score=np.nan, 
            best_estimator_sem=np.nan,
            best_features=np.nan,
            best_estimator_rci=np.nan,
            feature_weights=np.nan,
            highest_score=highest_score,
            highest_scoring_features=np.nan,
            cv_splits=cv_splits, 
            cv_labels=cv_labels,
            testing_scores=testing_scores,
            best_estimator_testing_score=np.nan,
            ),
            name="recursive_feature_elimination",
        )



# -------
# Classes
# -------
class LegacyClairvoyanceBase(object):
    def __init__(
        self,
        # Modeling
        estimator,
        param_space:dict,
        scorer,
        method:str="asymmetric", #imbalanced?
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
        assert len(param_space) > 0, "`param_space` must have at least 1 key:[value_1, value_2, ..., value_n] pair"
        self.param_space = param_space
            
        # Set attributes
        self.name = name
        self.observation_type = observation_type
        self.feature_type = feature_type
        self.target_type = target_type
        self.is_fitted_weights = False
        self.is_fitted_rci = False
        self.testing_set_provided = False
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
            pad*" " + "* Parameter Space: {}".format(self.param_space),
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

            param_space_expanded = list(map(lambda x:dict(zip(self.param_space.keys(), x)), itertools.product(*self.param_space.values())))
            for i, params in enumerate(param_space_expanded):
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
                return (np.nanmean([weight_A, weight_B], axis=0), np.nanmean([score_A_B, score_B_A]))

        def _fit_estimator_asymmetric(X_training, X_validation, y_training, y_validation, estimator):
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
                score = self.scorer(estimator=estimator, X=X_validation, y_true=y_validation)
            
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
                    X_training, X_validation, y_training, y_validation = train_test_split(
                        X,
                        y,
                        test_size=split_size, 
                        stratify=stratify, 
                        random_state=i,
                    )

                    training_data = dict(
                        X_training=X_training,
                        X_validation=X_validation,
                        y_training=y_training,
                        y_validation=y_validation,
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
            assert set(sort_hyperparameters_by) <= set(self.param_space.keys()), "`sort_hyperparameters_by` must contain a list of keys in `param_space`"
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

        if not df.empty:
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
    
    def get_weights(self, minimum_score=None, metrics=[np.nanmean, stats.sem], minimim_score_too_high_action="adjust"):
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
    
    
    def recursive_feature_addition(
        self, 
        estimator=None, 
        X:pd.DataFrame=None, 
        y:pd.Series=None, 
        X_testing:pd.DataFrame=None,
        y_testing:pd.Series=None,
        cv=(5,3), 
        minimum_score=None, 
        metric=np.nanmean, 
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
        # optimize_testing_score = "auto",

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
        
        # Testing
        X_testing_is_provided = X_testing is not None
        y_testing_is_provided = y_testing is not None

        if X_testing_is_provided is not None:
            assert y_testing_is_provided is not None, "If `X_testing` is provided then `y_testing` must be provided"

        if y_testing_is_provided is not None:
            assert X_testing_is_provided is not None, "If `y_testing` is provided then `X_testing` must be provided"

        self.testing_set_provided = False
        if all([X_testing_is_provided, y_testing_is_provided]):
            assert np.all(X_testing.index == y_testing.index), "X_testing.index and y_testing.index must have the same ordering"
            assert np.all(X_testing.columns == X.columns), "X_testing.columns and X.columns must have the same ordering"
            self.testing_set_provided = True


        # Recursive feature incusion
        rci_results = recursive_feature_addition(
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
            X_testing=X_testing,
            y_testing=y_testing,
            # optimize_testing_score=optimize_testing_score,
            )
        
        # Results
        # self.testing_scores_ = rci_results["testing_scores"]
        self.history_ = rci_results["history"]
        if self.remove_zero_weighted_features:
            if self.testing_set_provided:
                self.history_ = self.history_.sort_values([("summary", "testing_score"), ("summary", "average_score"), ("summary", "number_of_features")], ascending=[False, False, less_features_is_better])
            else:
                self.history_ = self.history_.sort_values([("summary", "average_score"), ("summary", "number_of_features")], ascending=[False, less_features_is_better])
        self.highest_score_ = rci_results["highest_score"]
        self.highest_scoring_features_ = rci_results["highest_scoring_features"]
        self.best_score_ = rci_results["best_score"]
        self.best_estimator_sem_ = rci_results["best_estimator_sem"]
        self.best_features_ = rci_results["best_features"]
        self.best_estimator_rci_ = clone(estimator)
        self.best_estimator_testing_score_ = rci_results["best_estimator_testing_score"]

        self.status_ok_ = False
        if isinstance(self.best_features_, list):
            if np.all(pd.notnull(self.best_features_)):
                self.status_ok_ = True

        if self.status_ok_:
            X_rci = transform(X=X.loc[:,self.best_features_], method=self.transformation, multiplicative_replacement=self.multiplicative_replacement, axis=1)
            with warnings.catch_warnings(): #!
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                self.best_estimator_rci_.fit(X_rci, y)
            self.rci_feature_weights_ = rci_results["feature_weights"][("full_dataset", "rci_weights")].loc[self.best_features_]
        else:
            self.rci_feature_weights_ = np.nan
        self.feature_weights_ =  rci_results["feature_weights"]
        self.cv_splits_ = rci_results["cv_splits"]
        self.cv_labels_ = rci_results["cv_labels"]
        
        if copy_X_rci:
            if self.status_ok_:
                self.X_rci_ = X_rci.copy()
            else:
                self.X_rci_ = np.nan
            
        self.is_fitted_rci = True
        
        return self.history_

    def get_history(self, sort_values_by=[("summary", "testing_score"), ("summary", "average_score"), ("summary", "number_of_features")], ascending=[False, False, True], summary=True):
        assert self.is_fitted_rci, "Please run `fit` before proceeding."

        df_history = self.history_.copy()

        if sort_values_by is not None:
            df_history = df_history.sort_values(by=sort_values_by, ascending=ascending)
        if summary:
            df_history = df_history["summary"]
            
        return df_history
    
    def plot_scores(
        self,
        ylabel="auto",
        **kwargs,
        ):
        if ylabel == "auto":
            ylabel = self.scorer_name
        kwargs["ylabel"] = ylabel
        assert self.is_fitted_rci, "Please run `recursive_feature_addition` before proceeding."
        if self.best_score_ < self.highest_score_:
            vertical_lines = [len(self.best_features_)-1, len(self.highest_scoring_features_)-1]
        else:
            vertical_lines = [len(self.best_features_)-1]

        if self.remove_zero_weighted_features:
            return plot_recursive_feature_selection(number_of_features=self.history_[("summary", "number_of_features")], average_scores=self.history_[("summary", "average_score")],  **kwargs)
        else:
            return plot_scores_line(average_scores=self.history_[("summary", "average_score")], sem=self.history_[("summary", "sem")], testing_scores=self.history_[("summary", "testing_score")], vertical_lines=vertical_lines, **kwargs)
        
    def plot_weights(
        self,
        weight_type=("full_dataset","rci_weights"),
        **kwargs,
        ):
        assert self.is_fitted_rci, "Please run `recursive_feature_addition` before proceeding."
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
        assert self.is_fitted_rci, "Please run `recursive_feature_addition` before proceeding."
        assert self.testing_set_provided, "Please run `recursive_feature_addition` with a testing set before proceeding."
        return plot_scores_comparison(number_of_features=self.history_[("summary", "number_of_features")], average_scores=self.history_[("summary", "average_score")],   testing_scores=self.history_[("summary", "testing_score")], **kwargs)

    def copy(self):
        return copy.deepcopy(self)
    
    # def to_file(self, path:str):
    #     write_object(self, path)  
        
    # @classmethod
    # def from_file(cls, path:str):
    #     cls = read_object(path)
    #     return cls
        
        
class LegacyClairvoyanceClassification(LegacyClairvoyanceBase):
    def __init__(
        self,
        # Modeling
        estimator,
        param_space:dict,
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
            
        
        super(LegacyClairvoyanceClassification, self).__init__(
            estimator=estimator,
            param_space=param_space,
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

class LegacyClairvoyanceRegression(LegacyClairvoyanceBase):

    def __init__(
        self,
        # Modeling
        estimator,
        param_space:dict,
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

            
        super(LegacyClairvoyanceRegression, self).__init__(
            estimator=estimator,
            param_space=param_space,
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

class LegacyClairvoyanceRecursive(object):
    def __init__(
        self,
        # Modeling
        estimator,
        param_space:dict,
        scorer,
        method="asymmetric",
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
            self.clairvoyance_class = LegacyClairvoyanceClassification
        if is_regressor(estimator):
            self.estimator_type = "regressor"
            self.clairvoyance_class = LegacyClairvoyanceRegression

        self.estimator = clone(estimator)
        
        if "random_state" in self.estimator.get_params(deep=True):
            query = self.estimator.get_params(deep=True)["random_state"]
            if query is None:
                if verbose > 0:
                    print("Updating `random_state=None` in `estimator` to `random_state={}`".format(random_state), file=log)
                self.estimator.set_params(random_state=random_state)
        self.feature_weight_attribute = get_feature_importance_attribute(estimator, importance_getter)
        assert len(param_space) > 0, "`param_space` must have at least 1 key:[value_1, value_2, ..., value_n] pair"
        self.param_space = param_space
            
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
        self.testing_set_provided = False
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
            pad*" " + "* Parameter Space: {}".format(self.param_space),
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
        X_testing:pd.DataFrame=None,
        y_testing:pd.Series=None,
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

        # Testing
        X_testing_is_provided = X_testing is not None
        y_testing_is_provided = y_testing is not None

        if X_testing_is_provided is not None:
            assert y_testing_is_provided is not None, "If `X_testing` is provided then `y_testing` must be provided"

        if y_testing_is_provided is not None:
            assert X_testing_is_provided is not None, "If `y_testing` is provided then `X_testing` must be provided"

        self.testing_set_provided = False
        if all([X_testing_is_provided, y_testing_is_provided]):
            assert np.all(X_testing.index == y_testing.index), "X_testing.index and y_testing.index must have the same ordering"
            assert np.all(X_testing.columns == X.columns), "X_testing.columns and X.columns must have the same ordering"
            self.testing_set_provided = True
            
        # Get cross-validation splits
        self.cv_splits_, self.cv_labels_ = format_cross_validation(cv, X, self.y_, stratify=self.stratify_, random_state=self.random_state, cv_prefix=cv_prefix, training_column=training_column, testing_column=testing_column)

        self.history_ = OrderedDict()
        self.results_ = OrderedDict()
        self.results_baseline_ = OrderedDict()
        self.status_ok_ = True

        with np.errstate(divide='ignore', invalid='ignore'):
            current_features_for_percentile = X.columns
            for i,pctl in enumerate(self.percentiles):
                if len(current_features_for_percentile) > 1:
                    if self.verbose > 2:
                        print("Feature set for percentile={}:".format(pctl), current_features_for_percentile.tolist(), sep="\n", file=self.log)


                    # Initiate model
                    model = self.clairvoyance_class(
                        estimator=self.estimator,
                        param_space=self.param_space,
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
                        w = model.get_weights(s)["nanmean"].sort_values(ascending=False)
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


                        if self.verbose > 2:
                            print("[Start] recursive feature inclusion [percentile={}, estimator_params={}]".format(pctl, params), file=self.log)

                        if not np.all(X_query == 0):
                            # Baseline
                            self._debug = dict(X=X_query, y=self.y_, scoring=self.scorer, cv=self.cv_splits_, n_jobs=self.n_jobs) #?

                            baseline_scores_for_percentile = cross_val_score(estimator=estimator, X=X_query, y=self.y_, scoring=self.scorer, cv=self.cv_splits_, n_jobs=self.n_jobs)

                            # break #?
                            if self.verbose > 3:
                                print("[Completed] Baseline cross-validation for training set [percentile={}, estimator_params={}]".format(pctl, params), file=self.log)
                            with warnings.catch_warnings():
                                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                                baseline_rci_weights = getattr(estimator.fit(X_query, self.y_), self.feature_weight_attribute)
                            if np.all(baseline_rci_weights == 0):
                                if self.verbose > 2:
                                    print("Excluding results from [percentile={}, estimator_params={}] becaue baseline model could not be fit with parameter set".format(pctl, params), file=self.log)
                            else:

                                baseline_testing_score = np.nan
                                if self.testing_set_provided:
                                    # Transform features (if transformation = None, then there is no transformation)
                                    X_testing_query = transform(X=X_testing.loc[:,current_features_for_percentile], method=self.transformation, multiplicative_replacement=self.multiplicative_replacement, axis=1)
                                    baseline_testing_score = self.scorer(estimator=estimator, X=X_testing_query, y_true=y_testing)
                                    if self.verbose > 3:
                                        print("[Completed] Baseline cross-validation for testing set [percentile={}, estimator_params={}]".format(pctl, params), file=self.log)
                                baseline_rci_weights = format_weights(baseline_rci_weights)
                                self.results_baseline_[(pctl,"baseline", params)] = {
                                    "testing_score":baseline_testing_score,
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
                                            metric=np.nanmean, 
                                            early_stopping=self.early_stopping, 
                                            minimum_improvement_in_score=self.minimum_improvement_in_score, 
                                            additional_feature_penalty=self.additional_feature_penalty,
                                            maximum_number_of_features=self.maximum_number_of_features,
                                            target_score=-np.inf, 
                                            less_features_is_better=less_features_is_better, 
                                            progress_message=progress_message,
                                    )
                                    if self.testing_set_provided:
                                        rci_params.update(
                                            dict(
                                            X_testing=X_testing.loc[:,current_features_for_percentile],
                                            y_testing=y_testing,
                                            )
                                        )

                                    rci_history = model.recursive_feature_addition(**rci_params)


                                     
                                    # Update feature weights
                                    # if not np.all(pd.isnull(model.rci_feature_weights_)):
                                    feature_weights_ok = model.status_ok_
                                    #!
                                    # Update weights if applicable
                                    # if pd.notnull(model.best_score_):
                                    model_score_ok = pd.notnull(model.best_score_)

                                    if all([feature_weights_ok, model_score_ok]):
                                        rci_feature_weights = model.rci_feature_weights_[model.best_features_]

                                        if model.best_score_ > best_score_for_percentile:
                                            best_score_for_percentile = model.best_score_
                                            best_hyperparameters_for_percentile = params
                                            best_minimum_score_for_percentile = s
                                            best_clairvoyance_feature_weights_for_percentile = model.clairvoyance_feature_weights_.copy()



                                        if np.any(rci_feature_weights > 0):
                                            self.results_[(pctl, params, s)] = {
                                                "testing_score":model.best_estimator_testing_score_,
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
                                                print("Excluding results from [percentile={}, estimator_params={}, minimum_score={}] because model could not be fit with parameter set".format(pctl, params, s), file=self.log)
                                        self.history_[(pctl,params, s)] = rci_history

                                        if self.verbose > 2:
                                            print("[End] recursive feature inclusion [percentile={}, estimator_params={}]".format(pctl, params), file=self.log)

                                        # Reset estimator
                                        model.estimators_[params] = clone(estimator)
                                    else:
                                        if self.verbose > 1:
                                            print("Excluding results from [percentile={}, estimator_params={}, minimum_score={}] failed the following checks:\n * feature_weights_ok = {}\n * model_score_ok = {}".format(pctl, params, s, feature_weights_ok, model_score_ok), file=self.log)


                    # Get new features
                    if i < len(self.percentiles) - 1:
                        if np.all(pd.isnull(best_clairvoyance_feature_weights_for_percentile)):
                            if self.verbose > 0:
                                print("Terminating algorithm. All weights returned as NaN.", file=self.log)
                            self.status_ok_ = False
                            break
                        else:
                            if self.remove_zero_weighted_features: #! Used to be exclude_zero_weighted_features which had a separate functionality from removing zero weighted features in the models.  Keep eye on this
                                nonzero_weights = best_clairvoyance_feature_weights_for_percentile[lambda x: x > 0]
                                self._debug2 = {
                                    "model":model,
                                    "best_score_for_percentile":best_score_for_percentile,
                                    "best_hyperparameters_for_percentile":best_hyperparameters_for_percentile,
                                    "best_minimum_score_for_percentile":best_minimum_score_for_percentile,
                                    "best_clairvoyance_feature_weights_for_percentile":best_clairvoyance_feature_weights_for_percentile,
                                }
                                
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

            self.results_ = pd.DataFrame(self.results_).T

            if self.status_ok_:
                self.results_ = self.results_.sort_values(["testing_score", "average_score", "number_of_features", "sem"], ascending=[False, False,less_features_is_better, True])
                self.results_.index.names = ["percentile", "hyperparameters", "minimum_score"]

                self.results_ = self.results_.loc[:,["testing_score", "average_score", "sem", "number_of_features", "features", "clairvoyance_weights", "rci_weights", "estimator"]]

                # Dtypes
                for field in ["testing_score", "average_score", "sem"]:
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
            else:
                if self.verbose > 0:
                    print("All models failed for parameter set and data input", file=self.log)

            self.is_fitted_rci = True
            return self

    def get_history(self, sort_values_by=[("summary", "testing_score"), ("summary", "average_score"), ("summary", "number_of_features")], ascending=[False, False, True], summary=True):
        assert self.is_fitted_rci, "Please run `fit` before proceeding."

        dataframes = list()
        for params, df in self.history_.items(): # self.history_[(pctl,params, s)] = rci_history
            df = df.copy()
            df.index = df.index.map(lambda x: (*params, x))
            df.index.names = ["percentile", "hyperparameters", "minimum_score", "features"]
            dataframes.append(df)

        df_concatenated = pd.concat(dataframes, axis=0)
        if sort_values_by is not None:
            df_concatenated = df_concatenated.sort_values(by=sort_values_by, ascending=ascending)
        if summary:
            df_concatenated = df_concatenated["summary"]
            
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
            if self.testing_set_provided:
                comprehensive = True 
            else:
                comprehensive = False
            
        if comprehensive:
            df = self.get_history(summary=True)
            number_of_features = df["number_of_features"]
            average_scores = df["average_score"]
            # Testing_scores = df[("summary", "testing_score")]
        else:
            number_of_features = self.results_["number_of_features"]
            average_scores = self.results_["average_score"]
            testing_scores = self.results_["testing_score"]
            if include_baseline:
                number_of_features = pd.concat([number_of_features, self.results_baseline_["number_of_features"]])
                average_scores = pd.concat([average_scores, self.results_baseline_["average_score"]])
                # Testing_scores = pd.concat([testing_scores, self.results_baseline_["testing_score"]])

        return plot_recursive_feature_selection(number_of_features=number_of_features, average_scores=average_scores,  **kwargs)

    def plot_scores_comparison(
        self,
        **kwargs,
        ):
        assert self.is_fitted_rci, "Please run `recursive_feature_addition` before proceeding."
        assert self.testing_set_provided, "Please run `recursive_feature_addition` with a testing set before proceeding."
        df = self.get_history(summary=True)
        number_of_features = df["number_of_features"]
        average_scores = df["average_score"]
        testing_scores = df["testing_score"]
        return plot_scores_comparison(number_of_features=number_of_features, average_scores=average_scores,   testing_scores=testing_scores, **kwargs)

    def to_file(self, path:str):
        write_object(self, path)  


        
    @classmethod
    def from_file(cls, path:str):
        cls = read_object(path)
        return cls

# BayesianClairvoyanceBase
# BayesianClairvoyanceClassification
# BayesianClairvoyanceRegression
# BayesianClairvoyanceRecursive
# Shap?
# feature addition or elimination
# * Algo is going to be hyperparameter tuning then feature selection, then hyperparameter tuning then feature selection, for n_iter times. Should be good after like 10. 
