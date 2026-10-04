from cost_function import compute_cost
import numpy as np


def compute_gradient(X, y, w, b):
    """
    Calcule les gradients par rapport aux poids w et au biais b.
    """

    m = X.shape[0]

    predictions = X @ w + b

    errors = predictions - y

    dj_dw = (1 / m) * (X.T @ errors)

    dj_db = (1 / m) * np.sum(errors)

    return dj_dw, dj_db


def gradient_descent(X, y, w, b, alpha, num_iters):
    """
    Entraîne une régression linéaire avec Gradient Descent.
    """

    for i in range(num_iters):

        dj_dw, dj_db = compute_gradient(
            X, y, w, b
        )

        w = w - alpha * dj_dw

        b = b - alpha * dj_db

    return w, b