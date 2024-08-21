from .common import loss_quadratic, loss_logistic

from typing import List, Tuple

import numpy as np
from scipy.spatial import distance

import jaxlib
import jax
from jax import random, jit
import jax.numpy as jnp

from jaxopt import GradientDescent, LBFGS, ProximalGradient
from jaxopt.prox import prox_lasso


def score_to_labels(score):
    return np.where(score > 0, 1, -1)


@jit
def softplus(x, beta=1):
    # calculate (1/beta) * np.log(1 + np.exp(beta * x))
    return (1 / beta) * jnp.logaddexp(beta * x, 0)


def predict1(params, x, scale=1.0):
    """Forward pass of one weak learner"""
    a, b, w = params
    return a + b * softplus(x=jnp.dot(x, w.T), beta=scale)


def add_intercept(x):
    """Add intercept term in dataset array as feature 0"""
    x = np.atleast_2d(x)
    return np.hstack((x, np.ones((x.shape[0], 1))))


# params for lasso proximal
def init_lasso_params(p, penalty_weight):
    # First layer weights and biases
    w = penalty_weight * jnp.ones(p)
    a = jnp.array([0])
    return (a, w)


def init_network_params1(p, b, key):
    """Initialize parameters for one weak learner

    Args:
        p (int): Number of data dimensions.
        b (float): b parameter value (bias).
        key (jax.PRNGKey): jax pseudo random number key.

    Returns:
        _type_: _description_
    """
    a_key, w_key = random.split(key, 2)
    # projection weights
    w = random.normal(w_key, (p,))
    # scaling term
    a = random.normal(a_key)
    # bias term
    b = jnp.array(b)
    return (a, b, w)


def learn1(
    x,
    y,
    y0,
    params0,
    optimizer_params,
    loss=loss_quadratic,
    ridge=1e-3,
    softplus_scale=1,
    optimizer=LBFGS,
):
    """Optimize parameters for one GBMAP iteration (one feature) with ridge penalty.

    Args:
        x (array-like): Covariates.
        y (array-like): Target.
        y0 (array-like): Initial prediction.
        params0 (_type_): Initial GBMAP params.
        optimizer_params (dict): Jaxopt optimizer parameters.
        loss (func, optional): Loss fuction. Defaults to loss_quadratic.
        ridge (float, optional): Ridge penalty. Defaults to 1e-3.
        softplus_scale (int, optional): Softplus scaling term. Defaults to 1.
        optimizer (jaxopt-optimizer, optional): Optimizer. Defaults to LBFGS.

    Returns:
        tuple: Optimized GBMAP params for one iteration.
    """

    # repackage a,w to avoid changing b
    a, b, w = params0
    par0 = a, w

    def objective_fn(par):
        a, w = par
        params = a, b, w
        return loss(
            y, y0 + predict1(params=params, x=x, scale=softplus_scale)
        ) + ridge * jnp.average(w**2)

    solver = optimizer(fun=objective_fn, **optimizer_params)
    opt_result = solver.run(init_params=par0)
    # Extract optimized parameters
    a, w = opt_result.params
    return (a, b, w)


def learn1_prox(
    x,
    y,
    y0,
    params0,
    proximal_params,
    optimizer_params,
    loss=loss_quadratic,
    ridge=1e-3,
    softplus_scale=1,
):
    """Optimize one GBMAP iteration with proximal lasso"""

    a, b, w = params0
    par0 = a, w

    def objective_fn(par):
        a, w = par
        params = a, b, w
        return loss(
            y, y0 + predict1(params=params, x=x, scale=softplus_scale)
        ) + ridge * jnp.average(w**2)

    solver = ProximalGradient(fun=objective_fn, prox=prox_lasso, **optimizer_params)
    opt_result = solver.run(
        par0,
        hyperparams_prox=proximal_params,
    )
    # Extract optimized parameters
    a, w = opt_result.params
    return (a, b, w)


def learn(
    m,
    x,
    y,
    key,
    optimizer_params,
    n_fit=1,
    lambda_l2=1e-3,
    lambda_l1=1e-3,
    regularization="l2",
    y0=None,
    loss=loss_quadratic,
    softplus_scale=1,
    optimizer=LBFGS,
    width_refit=False,
):
    """Optimize GBMAP m iterations greedily.

    Args:
        m (int): Number of new features (dimensionality of embedding).
        x (array-like): _description_
        y (array-like): _description_
        key (jax.PRNGKey): jax pseudo random number key.
        optimizer_params (dict): Jaxopt optimizer parameters.
        n_fit (int, optional): Number of times to optimize GBMAP with different initial values. Defaults to 1.
        lambda_l2 (float, optional): Ridge penalty. Defaults to 1e-3.
        lambda_l1 (float, optional): Lasso penalty. Defaults to 1e-3.
        regularization (str, optional): Which regularization to use, one of ['l1', 'l2']. Defaults to "l2".
        y0 (array-like, optional): Initial predictions. Defaults to None.
        loss (func, optional): Loss fuction. Defaults to loss_quadratic.
        softplus_scale (int, optional): Softplus scaling term. Defaults to 1.
        optimizer (jaxopt-optimizer, optional): Optimizer. Defaults to LBFGS.
        width_refit (bool, optional): Perform the refitting depth-wise, i.e., refit each GBMAP iteration
        n_fit times and pick the best. Defaults to False.

    Raises:
        TypeError: Raised if regularization param is not 'l1' or 'l2'.

    Returns:
        tuple: GBMAP parameters and losses
    """

    if y0 is None:
        y0 = jnp.zeros(y.shape)

    def learnf_l2(y0, b, key):
        params0 = init_network_params1(x.shape[1], b, key)

        params = learn1(
            x=x,
            y=y,
            y0=y0,
            params0=params0,
            optimizer_params=optimizer_params,
            loss=loss,
            ridge=lambda_l2,
            softplus_scale=softplus_scale,
            optimizer=optimizer,
        )
        predict = predict1(params=params, x=x, scale=softplus_scale)
        lossv = loss(y, y0 + predict)
        return params, lossv

    def learnf_l1(y0, b, key):
        params0 = init_network_params1(x.shape[1], b, key)
        # lasso params for proximal gradient
        l1_hyperparams = init_lasso_params(x.shape[1], lambda_l1)

        params = learn1_prox(
            x=x,
            y=y,
            y0=y0,
            params0=params0,
            proximal_params=l1_hyperparams,
            optimizer_params=optimizer_params,
            loss=loss,
            ridge=lambda_l2,
            softplus_scale=softplus_scale,
        )
        predict = predict1(params=params, x=x, scale=softplus_scale)
        lossv = loss(y, y0 + predict)
        return params, lossv

    # set up optimization for one weak learner
    if regularization == "l2":
        # use l2 penalty
        learnf_jit = jit(learnf_l2)
    elif regularization == "l1":
        # l1 penalty
        learnf_jit = jit(learnf_l1)
    else:
        raise TypeError("Invalid regularization '{}'.".format(regularization))

    ##############
    # width refit
    ##############
    if width_refit:
        # fits jth params n_fit times and picks the best
        params = [None] * m
        losses = np.array([None] * m)
        # pred from the current ensemble
        yhat = y0.copy()

        # fit m weak learners
        for j in range(m):

            # fit one weak learner n_fit times with a different seed
            lossesi = [None] * n_fit
            paramsi = np.array([None] * n_fit)
            for i in range(n_fit):
                key, subkey = random.split(key)
                # fit weak learner with b = +1/-1
                params_plus, loss_plus = learnf_jit(yhat, 1.0, subkey)
                params_minus, loss_minus = learnf_jit(yhat, -1.0, subkey)
                paramsi[i] = params_minus if loss_minus < loss_plus else params_plus
                lossesi[i] = (
                    loss_minus if loss_minus < loss_plus else loss_plus
                ).item()

            # pick the best fit (min of losses)
            min_idx = np.argmin(lossesi)
            losses[j] = lossesi[min_idx]
            params[j] = paramsi[min_idx]
            # update ensemble pred
            yhat = yhat + predict1(params[j], x, scale=softplus_scale)
        return params, losses
    ##############
    # depth refit
    ##############
    else:
        # fits all m params n_fit times and picks the best
        params_losses = []
        for i in range(n_fit):
            paramsi = [None] * m
            lossesi = np.array([None] * m)
            # pred from current ensemble (need to reset for learning another gbmap)
            yhat = y0.copy()

            # fit m weak learners
            for j in range(m):
                key, subkey = random.split(key)
                params_plus, loss_plus = learnf_jit(yhat, 1.0, subkey)
                params_minus, loss_minus = learnf_jit(yhat, -1.0, subkey)
                paramsi[j] = params_minus if loss_minus < loss_plus else params_plus
                lossesi[j] = (
                    loss_minus if loss_minus < loss_plus else loss_plus
                ).item()
                # update ensemble pred
                yhat = yhat + predict1(paramsi[j], x, scale=softplus_scale)
            params_losses.append((paramsi, lossesi))

        # pick the best fit (min of final losses)
        params, losses = min(params_losses, key=lambda pl: pl[1][-1])
        return params, losses


def predict(params, x, y0=0.0, softplus_scale=1):
    """Complete GBMAP prediction (all iterations)"""
    z = jnp.array([predict1(parami, x, softplus_scale) for parami in params]).T
    return y0 + jnp.sum(z, axis=1)


class GBMAP:
    """Gradient Boosting MAP class"""

    def __init__(
        self,
        n_boosts=10,  # boosting iterations (m)
        optim_maxiter=100,  # max iterations for a single optimization iteration
        optim_tol=1e-3,  # optimizer tolerance stopping criterion
        penalty_ridge=1e-3,  # regularization strength for l2
        penalty_lasso=1e-3,  # regularization strength for l1
        regularization="l2",  # regularization is ridge by default (set to 'l1' for lasso)
        softplus_scale=1,  # softplus scaling
        optimizer="lbfgs",  # optimizer (set to 'gd' for vanilla gradient descent)
        n_fit=1,  # number of refits used in optimizing gbmap
        width_refit=False,  # width-wise refitting: optimize each iteration n_fit times before moving on
        is_classifier=False,  # is classifier or not
        random_state=None,  # random seed, accepts jax pseudo random number or int
    ):
        self.n_boosts = n_boosts
        self.optim_maxiter = optim_maxiter
        self.optim_tol = optim_tol
        # regularization strength
        self.penalty_ridge = penalty_ridge
        self.penalty_lasso = penalty_lasso
        self.softplus_scale = softplus_scale
        self.regularization = regularization
        if regularization == "l1":
            # l1 regularization, use proximal optimizer
            self.optimizer = ProximalGradient
        elif regularization == "l2":
            # l2, use gradient descent or LBFGS
            if optimizer == "gd":
                self.optimizer = GradientDescent
            if optimizer == "lbfgs":
                self.optimizer = LBFGS
        else:
            raise TypeError(
                "'{}' is not a valid regularization, should be 'l1' or 'l2'.".format(
                    regularization
                )
            )
        self.n_fit = n_fit
        if is_classifier:
            self.loss = loss_logistic
        else:
            self.loss = loss_quadratic

        self.is_classifier = is_classifier

        self.width_refit = width_refit

        # random seed for reproducibility
        self.random_state = random_state

        # train losses for each boosting iteration
        self.losses = None
        # model parameters
        self.params = None

    def fit(self, x, y, y0=None, n_fit=None):
        """Optimize GBMAP"""

        n_fit = self.n_fit if n_fit is None else n_fit
        x_with_intercept = add_intercept(x)

        if type(self.random_state) is jaxlib.xla_extension.ArrayImpl:
            key = self.random_state
        elif type(self.random_state) is int:
            key = jax.random.PRNGKey(self.random_state)
        else:
            # random seed not set, get a random seed (seems hacky, but jax requires a seed to be set)
            key = jax.random.PRNGKey(np.random.randint(low=0, high=1e16))

        params, losses = learn(
            m=self.n_boosts,
            x=x_with_intercept,
            y=y,
            key=key,
            optimizer_params={"maxiter": self.optim_maxiter, "tol": self.optim_tol},
            n_fit=n_fit,
            lambda_l2=self.penalty_ridge,
            lambda_l1=self.penalty_lasso,
            regularization=self.regularization,
            y0=y0,
            loss=self.loss,
            softplus_scale=self.softplus_scale,
            optimizer=self.optimizer,
            width_refit=self.width_refit,
        )
        self.params, self.losses = params, losses
        return self

    def predict(self, x, y0=0.0, n_boosts=None, get_score=False):
        """GBAMP prediction"""

        if n_boosts is None:
            n_boosts = self.n_boosts

        x_with_intercept = add_intercept(x)

        z = jnp.array(
            [
                predict1(parami, x_with_intercept, self.softplus_scale)
                for parami in self.params[:n_boosts]
            ]
        ).T

        score = np.array(y0 + jnp.sum(z, axis=1))

        if self.is_classifier:
            if get_score:
                return score
            else:
                # transform scores to labels
                return score_to_labels(score)
        else:
            return score

    def transform(self, x):
        """Embed data"""

        return np.hstack(
            [
                predict1(
                    params=self.params[i], x=add_intercept(x), scale=self.softplus_scale
                )[:, None]
                for i in range(self.n_boosts)
            ]
        )

    def get_coordinate_distance(
        self, reference_points, sample_points, metric="euclidean", stat="min"
    ):
        """Compute the distance between each sample point to the set of reference points in the embedding coordinates

        Args:
            reference_points (array like): a set of reference points
            sample_points (array like): a set of sample points we want to compute the distance to reference_points

        Returns:
            array like: distance in the embedding coordinates
        """
        coor_ref = np.array(self.transform(reference_points))
        coor_sam = np.array(self.transform(sample_points))
        C = distance.cdist(coor_sam, coor_ref, metric=metric)
        if stat == "min":
            dis = np.min(C, axis=1)
        elif stat == "mean":
            dis = np.mean(C, axis=1)
        else:
            raise ValueError("stat not implemented.")

        return dis

    def get_activated_hyperplane(self, X):
        """Get activated hyperplanes

        A weak learner with parameter w is activated for a point x if w^T x > 0.
        This function returns the indices of activated weak learners for every point in X.

        Args:
            X (array like): data

        Returns:
            list of tuples: activated regions for every point (row) in X
        """
        X = np.atleast_2d(X)
        n = X.shape[0]
        X = np.hstack((X, np.ones((n, 1))))  # (n, p+1)
        m = len(self.params)
        W = np.array([w for (_, _, w) in self.params])  # (m, p+1)
        A = X @ W.T > 0  # (n, m)
        return [tuple(np.where(A[i, :])[0]) for i in range(n)]

    def get_explanation(self, X=None, regions=None, total_intercept=False):
        """Get linear model coefficients

        This function assumes self is piecewise linear (i.e., infinite self.softplus_scale).
        self can then be written as:
        $f(x) = \sum_{k=1}^m f_k(x) = \sum_k a_k + b_k * w_k^T * x * I(w_k^T x > 0)$,
        where I(.) = 1 if . is true and zero otherwise.

        The coefficients in a region R are the sum of coefficients over the weak
        learners activated in R (sum_{k active in R} w_k * b_k).

        The coefficients for a point are the coefficients of the region it belongs to.

        Args:
            X (array like): data points as (n, p) array
            regions (list of tuples): k regions as tuples of size <=m, with values <m,
            where m = self.n_boosts. Ignored if X is not None.
            total_intercept (bool): If False (default), the intercept term for the k:th learner
            is $b_k * w_{k,p+1}$, else it is $a_k + b_k * w_{k,p+1}$.

        Returns:
            One of:
            (n, p+1) array: p+1 coefficients for every point in X, if X is not None.
            (k, p+1) array: p+1 coefficients for every region in regions, if regions is not None.
            (m, p+1) array: p+1 coefficients for every learner in self, if both X and regions are None.
        """
        Wb = np.array([b * w for (_, b, w) in self.params])  # (m, p+1)
        if total_intercept:
            j_intercept = Wb.shape[1] - 1
            A = np.array([a for a, _, _ in self.params]).ravel()
            Wb[:, j_intercept] = Wb[:, j_intercept] + A
            W0 = np.repeat([0, A.sum()], [Wb.shape[1] - 1, 1])
        if X is None and regions is None:
            return Wb
        if X is not None:
            # Ignore regions.
            regions = self.get_activated_hyperplane(X)

        return np.array(
            [
                (np.sum(Wb[r, :], axis=0) if not (total_intercept and r == ()) else W0)
                for r in regions
            ]
        )

    def get_params(self, i):
        a, b, w = self.params[i]
        return {"a": a, "b": b, "w": w}

    def print_params(self, i=None):
        if i is None:
            for j, param in enumerate(self.params):
                a, b, w = param
                print("Params for weak learner j={}".format(j))
                print("a (linear constant): {}".format(a))
                print("b (linear scale): {}".format(b))
                print("w (feature weights): {}".format(w[:-1]))
                print("w (intercept): {}".format(w[-1]))
                print("-----------------")

    @classmethod
    def from_params(cls, params: List[Tuple], input_params: dict = dict()):
        """Construct GBMAP from a list of parameters

        Allows skipping fitting.

        Args:
            params (list): Parameters for weak learners as a list of m tuples (a, b, w).
            input_params (dict): Input parameters for GBMAP constructor. n_boosts is ignored and set to len(params).

        Returns:
            GBMAP
        """
        input_params = input_params | {"n_boosts": len(params)}
        gbmap = cls(**input_params)
        gbmap.params = params
        return gbmap

    def __repr__(self):
        return f"GBMAP(n_boosts={self.n_boosts}, softplus_scale={self.softplus_scale})"
