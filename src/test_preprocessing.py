from preprocessing import preprocess_data, preprocess_new_car


# Préparer les données d'entraînement
(
    X_train,
    X_test,
    y_train,
    y_test,
    scaler,
    encoder
) = preprocess_data("data/bmw.csv")

# Préparer une nouvelle BMW
X_new = preprocess_new_car(
    model="3 Series",
    year=2018,
    mileage=50000,
    transmission="Automatic",
    fuel_type="Diesel",
    tax=150,
    mpg=50.0,
    engine_size=2.0,
    encoder=encoder,
    scaler=scaler
)


print("Shape :", X_new.shape)
print("Données :", X_new)