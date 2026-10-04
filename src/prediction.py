def predict(X, w, b):
    """
    Calcule les prédictions du modèle.

    Parameters
    ----------
    X : numpy.ndarray
        Données d'entrée.
    w : numpy.ndarray
        Poids entraînés.
    b : float
        Biais entraîné.

    Returns
    -------
    numpy.ndarray
        Prix prédits.
    """

    predictions = X @ w + b

    return predictions