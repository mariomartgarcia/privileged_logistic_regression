import cvxpy as cp
import numpy as np
import warnings
# To define the Privliged Logistic Regression Class
from sklearn.base import BaseEstimator
from sklearn.linear_model._base import LinearClassifierMixin
from sklearn.linear_model._base import SparseCoefMixin
# To compute class weight and sample weight
from sklearn.utils import compute_class_weight
from sklearn.utils.validation import _check_sample_weight
# To check the input dimension
from privileged_util import _check_X_y, _check_constraints, _check_bounds
# To check if the class is fitted
from sklearn.utils.validation import check_is_fitted
from cvxpy.expressions import cvxtypes
from scipy.special import expit
from sklearn.preprocessing import LabelEncoder

class PrivilegedLogisticRegression(BaseEstimator, LinearClassifierMixin,
                         SparseCoefMixin):
    """
    Constrained Logistic Regression (aka logit, MaxEnt) classifier.
    This class implements regularized logistic regression supported bound
    and linear constraints using the 'ecos' solver.
    All solvers support only L1, L2 and Elastic-Net regularization or no
    regularization. The 'lbfgs' solver supports bound constraints for L2
    regularization. The 'ecos' solver supports bound constraints and linear
    constraints for all regularizations.
    Parameters
    ----------
    penalty : {'l1', 'l2', 'elasticnet', 'none'}, default='l2'
        Used to specify the norm used in the penalization. The 'lbfgs',
        solver supports only 'l2' penalties if bounds are provided.
        If 'none', no regularization is applied.
    tol : float, default=1e-4
        Tolerance for stopping criteria.
    C : float, default=1.0
        Inverse of regularization strength; must be a positive float.
        Like in support vector machines, smaller values specify stronger
        regularization.
    fit_intercept : bool, default=True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the decision function.
    class_weight : dict or 'balanced', default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one.
        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``.
        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.
    solver : {'ecos', 'lbfgs'}, default='lbfgs'
        Algorithm/solver to use in the optimization problem.
        - Unconstrained 'lbfgs' handles all regularizations.
        - Bound constrainted 'lbfgs' handles L2 or no penalty.
        - For other cases, use 'ecos'.
        Note that 'ecos' uses the general-purpose solver ECOS via CVXPY.
    max_iter : int, default=100
        Maximum number of iterations taken for the solvers to converge.
    l1_ratio : float, default=None
        The Elastic-Net mixing parameter, with ``0 <= l1_ratio <= 1``. Only
        used if ``penalty='elasticnet'`. Setting ``l1_ratio=0`` is equivalent
        to using ``penalty='l2'``, while setting ``l1_ratio=1`` is equivalent
        to using ``penalty='l1'``. For ``0 < l1_ratio <1``, the penalty is a
        combination of L1 and L2.
    warm_start : bool, default=False
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
    verbose : bool, default=False
        Enable verbose output.
    Attributes
    ----------
    classes_ : ndarray of shape (n_classes, )
        A list of class labels known to the classifier.
    coef_ : ndarray of shape (1, n_features)
        Coefficient of the features in the decision function.
    intercept_ : ndarray of shape (1,)
        Intercept (a.k.a. bias) added to the decision function.
        If `fit_intercept` is set to False, the intercept is set to zero.
    """

    def __init__(self, penalty="l2", lambda_base=1.0, lambda_star=1.0, alpha=1.0,
        xi_link=1, tol=1e-4, fit_intercept=True, class_weight=None, class_weight_star=None,
        max_iter=100, verbose=False):

        if penalty == 'None' or penalty == 'none':
            penalty = None
        self.penalty = penalty
        
        self.lambda_base = lambda_base
        self.lambda_star = lambda_star
        self.alpha = alpha
        self.xi_link = xi_link

        self.tol = tol
        self.fit_intercept = fit_intercept
        self.class_weight = class_weight
        self.class_weight_star = class_weight_star
        self.max_iter = max_iter
        self.verbose = verbose

    def fit(self, X, y, X_star=None, y_star=None, sample_weight=None, sample_weight_star=None,
        bounds=None, bounds_star=None, constraints=None, constraints_star=None):
        """
        Fit the model according to the given training data.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like of shape (n_samples,)
            Target vector relative to X.
        sample_weight : array-like of shape (n_samples,) default=None
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.
        bounds & bounds_star: scipy.optimize.Bounds or None, default None
            Bounds on the coefficients and intercept.
        constraints & constraints_star: scipy.optimize.LinearConstraint or None, default None
            Linear constraints on the coefficients and intercept.
        Returns
        -------
        self
            Fitted estimator.
        """

        X, y, _ = _check_X_y(X, y)

        if y_star is None:
            y_star = y

        X_star, y_star, _ = _check_X_y(X_star, y_star)

        # Obtain Class Attributes
        self.classes_ = np.unique(y)

        n_samples, n_features = X.shape
        n_samples_star, n_features_star = X_star.shape

        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
        sample_weight_star = _check_sample_weight(sample_weight_star, X_star, dtype=X_star.dtype)


        if self.class_weight is not None:
            le = LabelEncoder()
            class_weight_ = compute_class_weight(
                class_weight=self.class_weight, classes=self.classes_, y=y)
            sample_weight *= class_weight_[le.fit_transform(y)]
        if self.class_weight_star is not None:
            le = LabelEncoder()
            class_weight_ = compute_class_weight(
                class_weight=self.class_weight, classes=np.unique(y_star), y=y_star)
            sample_weight_star *= class_weight_[le.fit_transform(y_star)]
   
        coef_, intercept_, coef_star_= self._fit_ecos(X, X_star, y, y_star, 
            sample_weight, sample_weight_star)

        if coef_ is not None:
            self.coef_ = np.asarray([coef_])
            self.intercept_ = np.asarray([intercept_])
            self.coef_star_ = np.asarray([coef_star_])
        else:
            # print('The problem is infeasible or unbounded.')
            raise ValueError('The problem is infeasible or unbounded.')

        return self

    def save_coef(self, save_dir, selection_metric):
        with open(save_dir / f'coef_{self.penalty}_{selection_metric}.npy', 'wb') as f:
            np.save(f, self.coef_)
            np.save(f, self.intercept_)
            np.save(f, self.coef_star_)


    def predict_proba(self, X):
        """
        Probability estimates.
        The returned estimates for all classes are ordered by the
        label of classes.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vector to be scored, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        Returns
        -------
        T : array-like of shape (n_samples, n_classes)
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in ``self.classes_``.
        """
        check_is_fitted(self)

        proba = np.empty((X.shape[0], 2))
        p0 = expit(-self.decision_function(X))
        proba[:, 0] = p0
        proba[:, 1] = 1 - p0

        return proba

    def predict_log_proba(self, X):
        """
        Predict logarithm of probability estimates.
        The returned estimates for all classes are ordered by the
        label of classes.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vector to be scored, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        Returns
        -------
        T : array-like of shape (n_samples, n_classes)
            Returns the log-probability of the sample for each class in the
            model, where classes are ordered as they are in ``self.classes_``.
        """
        return np.log(self.predict_proba(X))


    def _fit_ecos(self, X, X_star, y, y_star, sample_weight, sample_weight_star):

        '''
        Solve Logistic Regression with the ECOS solver

        Input:
            X: (m,n) data Information from base domain

        Output:

        '''
        # On Base Domain
        m, n = X.shape
        if self.fit_intercept:
            nn = n + 1
            X = np.c_[X, np.ones(m)]
        else:
            nn = n

        beta = cp.Variable(nn)
        Xbeta = X @ beta

        # On Privliged Domain
        m_star, n_star = X_star.shape
        beta_star = cp.Variable(n_star)
        Xbeta_star = X_star @ beta_star

        # Regularization
        w = beta
        if self.fit_intercept:
            w = beta[:-1]    
        w_star = beta_star

        if self.penalty == "l1":
            penalty = cp.norm(w, 1)
            penalty_star = cp.norm(w_star, 1)
        elif self.penalty == "l2":
            penalty = 0.5 * cp.sum_squares(w)
            penalty_star = 0.5 * cp.sum_squares(w_star)
        else:
            penalty = 0
            penalty_star = 0

        # Constraints
        cons = []
        cons.append(penalty <= self.lambda_base)
        cons.append(penalty_star <= self.lambda_star)
            
        # Objective Function
        objective_base = sample_weight @ (cp.logistic(-cp.multiply(y, Xbeta)))
        objective_star = self.alpha * (sample_weight_star @ cp.logistic(-cp.multiply(y_star, Xbeta_star))) 
        objective_link = self.xi_link * cp.sum_squares(cp.multiply(y, Xbeta)[:m_star] - cp.multiply(y_star, Xbeta_star))
        
        # Define and Solve the Optimization Problem in CVXPY
        obj = cp.Minimize(objective_base+objective_star+objective_link)

        problem = cp.Problem(obj, cons)
        try:
            problem.solve(max_iter=self.max_iter, verbose=self.verbose) #abstol=self.tol)
        except Exception as e:
            warnings.warn('Encountered an error while solving the problem. Hyperparameters may be invalid.' +
                          f'lambda_base: {self.lambda_base}, lambda_star: {self.lambda_star}, '+ 
                          f'alpha: {self.alpha}, xi_link: {self.xi_link}')
            # Sometimes would raise "cvxpy.error.SolverError: Solver 'ECOS' failed. 
            # Try another solver, or solve with verbose=True for more information."
            raise ValueError(e)

        # Return intercept and coefficient
        if self.fit_intercept:
            intercept_ = beta.value[-1]
            coef_ = beta.value[:-1]
        else:
            intercept_ = 0
            coef_ = beta.value
        coef_star_ = beta_star.value

        return coef_, intercept_, coef_star_


