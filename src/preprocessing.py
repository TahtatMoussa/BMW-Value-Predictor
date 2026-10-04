import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def preprocess_data(csv_path, test_size=0.2, random_state=42):

    # Charger le dataset
    df = pd.read_csv(csv_path)

    # Nettoyer les espaces dans les noms de modèles
    df["model"] = df["model"].str.strip()

    # Copie pour le nettoyage
    df_clean = df.copy()

    # Supprimer les doublons
    df_clean = df_clean.drop_duplicates()

    # Supprimer le prix considéré comme anomalie
    df_clean = df_clean[df_clean["price"] != 123456]

    # Supprimer les engineSize = 0 sauf pour les i3
    df_clean = df_clean[
        (df_clean["model"] == "i3") |
        (df_clean["engineSize"] != 0)
    ]

    # Séparer les features et la cible
    y = df_clean["price"]
    X = df_clean.drop("price", axis=1)

    # Variables catégorielles
    categorical_features = [
        "model",
        "transmission",
        "fuelType"
    ]

    # Variables numériques
    numerical_features = [
        "year",
        "mileage",
        "tax",
        "mpg",
        "engineSize"
    ]

    # One-Hot Encoding
    encoder = OneHotEncoder(
        drop="first",
        handle_unknown="ignore",
        sparse_output=False
    )

    X_categorical = encoder.fit_transform(
        X[categorical_features]
    )

    # Sélection des variables numériques
    X_numerical = X[numerical_features]

    # Combiner numérique + catégoriel
    X_final = np.hstack([
        X_numerical.to_numpy(),
        X_categorical
    ])

    # Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_final,
        y,
        test_size=test_size,
        random_state=random_state
    )

    # Standardisation des variables numériques
    scaler = StandardScaler()

    X_train_num = X_train[:, :5]
    X_test_num = X_test[:, :5]

    scaler.fit(X_train_num)

    X_train_num_scaled = scaler.transform(X_train_num)
    X_test_num_scaled = scaler.transform(X_test_num)

    # Variables catégorielles
    X_train_cat = X_train[:, 5:]
    X_test_cat = X_test[:, 5:]

    # Recombiner les features
    X_train_scaled = np.hstack([
        X_train_num_scaled,
        X_train_cat
    ])

    X_test_scaled = np.hstack([
        X_test_num_scaled,
        X_test_cat
    ])

    return (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler,
        encoder
    )



def preprocess_new_car(
    model,
    year,
    mileage,
    transmission,
    fuel_type,
    tax,
    mpg,
    engine_size,
    encoder,
    scaler
):
    """
    Prépare les caractéristiques d'une nouvelle BMW
    pour le modèle de Machine Learning.
    """

    # Créer un DataFrame contenant la nouvelle BMW
    new_car = pd.DataFrame([{
        "model": model,
        "year": year,
        "mileage": mileage,
        "transmission": transmission,
        "fuelType": fuel_type,
        "tax": tax,
        "mpg": mpg,
        "engineSize": engine_size
    }])

    # Variables catégorielles
    categorical_features = [
        "model",
        "transmission",
        "fuelType"
    ]

    # Variables numériques
    numerical_features = [
        "year",
        "mileage",
        "tax",
        "mpg",
        "engineSize"
    ]

    # Appliquer le même One-Hot Encoder
    X_categorical = encoder.transform(
        new_car[categorical_features]
    )

    # Sélectionner les variables numériques
    X_numerical = new_car[numerical_features]

    # Standardiser avec le même scaler
    X_numerical_scaled = scaler.transform(
    X_numerical.to_numpy()
    )

    # Combiner les variables numériques et catégorielles
    X_final = np.hstack([
        X_numerical_scaled,
        X_categorical
    ])

    return X_final