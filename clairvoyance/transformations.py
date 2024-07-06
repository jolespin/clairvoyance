# -*- coding: utf-8 -*-

# Built-ins
from ast import Or
import os, sys, time, warnings

# PyData
import pandas as pd
import numpy as np

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

def clr_with_multiplicative_replacement(X:pd.DataFrame, pseudocount="auto"):
    if pseudocount == "auto":
        if np.any(X == 0):
            pseudocount = 1/X.shape[1]**2
        else:
            pseudocount = None
    if pseudocount is None:
        pseudocount = 0.0
    X = X + pseudocount
    return clr(X)

# Transformations
def legacy_transform(X, method="closure", axis=1, multiplicative_replacement="auto", verbose=0, log=sys.stdout):
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
