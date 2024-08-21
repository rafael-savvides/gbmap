import jax.numpy as jnp
from jax.nn import softplus


# def mse(y, y_pred):
#    return jnp.mean((y - y_pred) ** 2)


def loss_quadratic(y, yp):
    """Quadratic loss"""
    return jnp.average((y - yp) ** 2)


def mse(y, yp):
    # alias for loss_quadratic
    return loss_quadratic(y, yp)


def loss_logistic(y, yp):
    """Logistic loss"""
    return jnp.average(softplus(-y * yp))


# Logit
def logit(p):
    return jnp.log(p / (1.0 - p))


# average function used, e.g., by knnpred for classification problems where y is a vector of target labels -1 and +1.
def caverage(y, pseudocount=1.0):
    return logit(
        (jnp.sum(0.5 + 0.5 * y) + 0.5 * pseudocount) / (y.shape[0] + pseudocount)
    )


def accuracy(y, yp):
    return jnp.average(y * yp >= 0.0)
