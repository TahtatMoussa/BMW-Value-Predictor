import numpy as np


def compute_cost(X, y, w, b):
    """
    Calcule le coût d'une régression linéaire.

    Parameters
    ----------
    X : numpy.ndarray
        Matrice des features.
    y : numpy.ndarray
        Valeurs réelles.
    w : numpy.ndarray
        Poids du modèle.
    b : float
        Biais du modèle.

    Returns
    -------
    float
        Valeur du coût.
    """

    m = X.shape[0]

    predictions = X @ w + b

    errors = predictions - y

    squared_errors = errors ** 2

    cost = (1 / (2 * m)) * np.sum(squared_errors)

    return cost