from sklearn.utils.validation import check_X_y
from sklearn.utils.multiclass import type_of_target
import numpy as np
from sklearn.utils import check_consistent_length
from scipy.optimize import Bounds, LinearConstraint

def _check_X_y(X, y):
    if type_of_target(y) != "binary":
        raise ValueError("This solver needs a binary target.")

    classes = np.unique(y)
    if len(classes) < 2:
        raise ValueError("This solver needs samples of 2 classes"
                         " in the data, but the data contains only one"
                         " class: {}.".format(classes[0]))
    # change all the y that equals zero to -1
    y[y == 0] = -1
    X, y = check_X_y(X, y, accept_sparse='csr', order="C")
    return X, y, classes

def _check_constraints(constraints, n, fit_intercept):
    if not isinstance(constraints, LinearConstraint):
        raise TypeError("Constraints is not of type "
                        "scipy.optimize.LinearConstraint.")

    A = constraints.A
    lb = constraints.lb
    ub = constraints.ub
    check_consistent_length(lb, ub)

    if n + int(fit_intercept) != A.shape[1]:
        raise ValueError("Number of columns of matrix A is incorrect; got {} "
                         "and must be {}.".format(A.shape[1], n + int(fit_intercept)))

    check_consistent_length(A, lb)

def _check_bounds(bounds, n, fit_intercept):
    if not isinstance(bounds, Bounds):
        raise TypeError("Bounds is not of type "
                        "scipy.optimize.Bounds.")
    lb = bounds.lb
    ub = bounds.ub
    check_consistent_length(lb, ub)
    