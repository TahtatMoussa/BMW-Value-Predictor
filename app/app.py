import sys
from pathlib import Path

import streamlit as st
import numpy as np
import joblib



PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.preprocessing import preprocess_new_car

model_data = joblib.load("models/bmw_model.pkl")

ml_model = model_data["model"]
scaler = model_data["scaler"]
encoder = model_data["encoder"]


# Configuration de la page
st.set_page_config(
    page_title="BMW Value Predictor",
    page_icon="🔴🔵",
    layout="centered"
)


# Logo BMW
logo_path = Path(__file__).parent / "assets" / "bmw_logo.png"

logo_col = st.columns([1, 2, 1])[1]

with logo_col:
    st.image(
        str(logo_path),
        width=300
    )

# Titre
st.title("🔴🔵BMW Value Predictor")

st.write(
    "Estimez le prix d'une BMW d'occasion "
    "à partir de ses caractéristiques."
)


# Section des caractéristiques
st.subheader("Caractéristiques de la BMW")


# Première ligne
col1, col2 = st.columns(2)

with col1:
    model = st.selectbox(
        "Modèle",
        [
            "1 Series",
            "2 Series",
            "3 Series",
            "4 Series",
            "5 Series",
            "6 Series",
            "7 Series",
            "8 Series",
            "M2",
            "M3",
            "M4",
            "M5",
            "M6",
            "X1",
            "X2",
            "X3",
            "X4",
            "X5",
            "X6",
            "X7",
            "Z3",
            "Z4",
            "i3",
            "i8"
        ]
    )

with col2:
    year = st.number_input(
        "Année",
        min_value=1996,
        max_value=2020,
        value=2018,
        step=1
    )


# Deuxième ligne
col1, col2 = st.columns(2)

with col1:
    mileage = st.number_input(
        "Kilométrage",
        min_value=0,
        max_value=300000,
        value=50000,
        step=1000
    )

with col2:
    transmission = st.selectbox(
        "Transmission",
        [
            "Automatic",
            "Manual",
            "Semi-Auto"
        ]
    )


# Troisième ligne
col1, col2 = st.columns(2)

with col1:
    fuel_type = st.selectbox(
        "Type de carburant",
        [
            "Diesel",
            "Petrol",
            "Hybrid",
            "Other",
            "Electric"
        ]
    )

with col2:
    tax = st.number_input(
        "Taxe",
        min_value=0,
        max_value=1000,
        value=150,
        step=1
    )


# Quatrième ligne
col1, col2 = st.columns(2)

with col1:
    mpg = st.number_input(
        "Consommation (MPG)",
        min_value=0.0,
        max_value=500.0,
        value=50.0,
        step=0.1
    )

with col2:
    engine_size = st.number_input(
        "Taille du moteur",
        min_value=0.0,
        max_value=7.0,
        value=2.0,
        step=0.1
    )


# Bouton de prédiction
st.divider()

if st.button("💰 Estimer le prix"):

    X_new = preprocess_new_car(
        model=model,
        year=year,
        mileage=mileage,
        transmission=transmission,
        fuel_type=fuel_type,
        tax=tax,
        mpg=mpg,
        engine_size=engine_size,
        encoder=encoder,
        scaler=scaler
    )

    prediction_log = ml_model.predict(X_new)

    prediction_price = np.exp(prediction_log[0])

    st.divider()

    st.subheader("💰 Estimation")

    st.metric(
        label="Prix estimé",
        value=f"{prediction_price:,.0f} €"
    )

    st.info(
        "Cette valeur est une estimation produite par "
        "le modèle de Machine Learning à partir des caractéristiques saisies."
    )