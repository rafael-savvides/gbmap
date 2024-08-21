from common import results_in_subdir_path

from experiment_configs import SCALING_N, SCALING_P

from gbmap.gbmap import (
    loss_quadratic,
    loss_logistic,
    init_network_params1,
    predict1,
    learn1,
)

import argparse
import time
import os

import numpy as np
import pandas as pd

from jax import random, jit
from jax.nn import sigmoid
import jax.numpy as jnp

from sklearn.decomposition import PCA
from lol import LOL
from ivis import Ivis


# Construct a random unit vector of dimensionality p
def random_unit_vector(p, key):
    while True:
        u = random.normal(key, (p,))
        s = jnp.sqrt(jnp.sum(u**2))
        if s > 0.0:
            return u / s
        key = random.split(key, 1)


# Make synthetic data as described in the manuscript
def make_synth(
    n,
    p,
    ntrain=None,
    prel=None,
    nonlin=lambda x: 5.0 * jnp.cos(x),
    classification=False,
    seed=42,
):
    key = random.PRNGKey(seed)
    if ntrain is None:
        ntrain = n
    if prel is None:
        prel = p
    keyX, key_irr, keyu, keyU = random.split(key, 4)
    X = random.normal(keyX, (n, prel))
    u = random_unit_vector(prel, keyu)

    y = nonlin(X) @ u

    if prel < p:
        X = jnp.hstack((X, random.normal(key_irr, (n, p - prel))))
    X = X @ random.orthogonal(keyU, p)

    y = y - jnp.average(y[:ntrain])

    if classification:
        y = jnp.select(
            [random.uniform(key, y.shape) <= sigmoid(y)], [1.0], default=-1.0
        )

    return np.array(X), np.array(y)


def add_intercept(x):
    if len(x.shape) == 1:
        x = x[None, :]
    return np.hstack((x, np.ones((x.shape[0], 1))))


def learn(
    m,
    x,
    y,
    key,
    y0=None,
    loss=loss_quadratic,
    ridge=1e-3,
    precompile=False,
):
    if y0 is None:
        y0 = jnp.zeros(y.shape)

    keys = random.split(key, m)

    @jit
    def learnf(y0, b, key):
        params0 = init_network_params1(x.shape[1], b, key)

        params = learn1(x, y, y0, params0, {"maxiter": 100, "tol": 1e-3}, loss, ridge)
        predict = predict1(params, x)
        lossv = loss(y, y0 + predict)
        return params, lossv

    if precompile:
        learnf(y0, 1.0, keys[0])

    params = [None] * m
    losses = [None] * m

    start_time = time.time()
    for j in range(m):
        params_plus, loss_plus = learnf(y0, 1.0, keys[j])
        params_minus, loss_minus = learnf(y0, -1.0, keys[j])
        params[j] = params_minus if loss_minus < loss_plus else params_plus
        losses[j] = (loss_minus if loss_minus < loss_plus else loss_plus).item()
        y0 = y0 + predict1(params[j], x)
    time_duration = time.time() - start_time

    return params, jnp.array(losses), time_duration


def dummy_regression(ytrain, ytest):
    return jnp.full(ytest.shape, jnp.average(ytrain))


def logit(p):
    return jnp.log(p / (1.0 - p))


def dummy_classification(ytrain, ytest):
    return jnp.full(ytest.shape, logit(jnp.average(0.5 + 0.5 * jnp.sign(ytrain))))


def scaling_experiment(method, n_size, p_size, repeat_idx, save_path):
    """Runs a scaling experiment

    Args:
        method (str): Dimensionality reduction method.
        n_size (_type_): Number of datapoints.
        p_size (_type_): Number of data dimensions.
        repeat_idx (int): Number of repeat (for parallel processing).
        save_path (str): Write path for results.

    Raises:
        ValueError: Raised if given method was unknown.
    """

    print("Scaling {}: n={}, p={}, r={}".format(method, n_size, p_size, repeat_idx))

    random_key = 42 + n_size + p_size + repeat_idx

    X, y = make_synth(n=n_size, p=p_size, classification=True, seed=random_key)

    loss = None
    runtime = None
    dummy = loss_logistic(y, dummy_regression(y, y))
    m_comp = 2  # 10

    if method == "GBMAP":
        X = add_intercept(X)
        key = random.PRNGKey(random_key)
        _, losses, runtime = learn(
            m_comp, X, y, key, loss=loss_logistic, precompile=True
        )
        loss = losses[-1]
    elif method == "PCA":
        start = time.time()
        pca = PCA(n_components=m_comp, svd_solver="full")
        _ = pca.fit_transform(X)
        runtime = time.time() - start
    elif method == "LOL":
        start = time.time()
        # LOL return always n_components-1 embedding
        loll = LOL(
            n_components=m_comp + 1,
            svd_solver="full",
        )
        _ = loll.fit_transform(X, y)
        runtime = time.time() - start
    elif method == "IVIS":
        y_ivis = (y + 1) / 2
        start = time.time()
        # LOL return always n_components-1 embedding
        ivis = Ivis(
            embedding_dims=m_comp,
            n_epochs_without_progress=5,
            verbose=0,
        )
        _ = ivis.fit_transform(X, y_ivis)
        runtime = time.time() - start
    else:
        raise ValueError("Unknown method ({}).".format(method))

    res = {
        "method": method,
        "n": n_size,
        "p": p_size,
        "repeat": repeat_idx,
        "runtime": runtime,
        "loss": loss,
        "dummyloss": dummy,
    }

    df = pd.DataFrame([res])  # dataframe with one row
    # append results to a file, creates a new file if one does not exists
    df.to_csv(
        save_path,
        mode="a",
        header=not os.path.exists(save_path),
        index=False,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--method",
        type=str,
        default="GBMAP",
        help="Dimensionality reduction method",
    )
    parser.add_argument(
        "-p",
        "--features",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Scaling datapoints (default) or features (using -p).",
    )
    parser.add_argument("-r", "--repeat", type=int, default=0, help="Repeat index")

    args = parser.parse_args()

    suffix = "p" if args.features else "n"

    # get a path for results.csv file
    save_path = results_in_subdir_path(
        "scaling",
        csv_filename="scaling_{}-{}_{}.csv".format(args.method, suffix, args.repeat),
    )

    if args.features:
        n_sizes = SCALING_P["n_sizes"]
        p_sizes = SCALING_P["p_sizes"]
    else:
        n_sizes = SCALING_N["n_sizes"]
        p_sizes = SCALING_N["p_sizes"]

    for n in n_sizes:
        for p in p_sizes:
            # data points scaling
            scaling_experiment(
                method=args.method,
                n_size=n,
                p_size=p,
                repeat_idx=args.repeat,
                save_path=save_path,
            )
