import numpy as np
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from preprocessing import preprocess_data


# 1. Préparer les données
X_train, X_test, y_train, y_test, scaler, encoder = preprocess_data(
    "data/bmw.csv"
)


# 2. Transformer les prix avec le logarithme
y_train_log = np.log(y_train)


# 3. Créer le modèle
model = LinearRegression()


# 4. Entraîner le modèle
model.fit(X_train, y_train_log)


# 5. Faire les prédictions
predictions_log = model.predict(X_test)


# 6. Revenir au prix réel en euros
predictions = np.exp(predictions_log)


# 7. Évaluer le modèle
mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)


print("===== ÉVALUATION DU MODÈLE =====")
print(f"MAE  : {mae:.2f} €")
print(f"RMSE : {rmse:.2f} €")


# 8. Regrouper le modèle et les outils de preprocessing
model_data = {
    "model": model,
    "scaler": scaler,
    "encoder": encoder
}


# 9. Sauvegarder tout dans un fichier
joblib.dump(
    model_data,
    "models/bmw_model.pkl"
)


print()
print("Modèle sauvegardé dans : models/bmw_model.pkl")