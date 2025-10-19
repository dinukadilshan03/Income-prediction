"""Prediction helpers converted from the inference notebook."""
from typing import Dict
from pathlib import Path

import pandas as pd
import numpy as np
import joblib

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
TOOLS_DIR = BASE_DIR / "tools"


RACE_MAPPING = {
    "White": "White",
    "Black": "Black",
    "Asian": "Asian-Pac-Islander",
    "Hispanic": "Amer-Indian-Eskimo",
    "Other": "Other",
}


def load_model_and_tools():
    model = joblib.load(MODELS_DIR / "best_xgboost_model.pkl")
    hours_per_week_scaler = None
    try:
        hours_per_week_scaler = joblib.load(TOOLS_DIR / "hours_scaler.pkl")
    except FileNotFoundError:
        hours_per_week_scaler = None
    education_encoder = joblib.load(TOOLS_DIR / "education_encoder.pkl")
    return model, education_encoder, hours_per_week_scaler


def preprocess_user_input(user_input: pd.DataFrame, reference_df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess a single user input row into the model feature space.

    reference_df should be the cleaned.csv used during training so we can align columns.
    """
    data = reference_df.copy()
    required_columns = [col for col in data.columns if col != "income"]

    # Encode education
    encoder = joblib.load(TOOLS_DIR / "education_encoder.pkl")
    user_input["education"] = encoder.transform(user_input["education"])

    # Map race
    user_input["race"] = user_input["race"].map(RACE_MAPPING).fillna("Other")

    # One-hot encode categorical variables
    user_input = pd.get_dummies(
        user_input,
        columns=["gender", "workclass", "marital-status", "occupation", "relationship", "race", "native-country"],
    )

    # Scale hours-per-week if scaler available
    try:
        hours_scaler = joblib.load(TOOLS_DIR / "hours_scaler.pkl")
        user_input[["hours-per-week"]] = hours_scaler.transform(user_input[["hours-per-week"]])
    except FileNotFoundError:
        pass

    # Log scaling for capital gains/losses
    user_input["capital-gain"] = np.log1p(user_input["capital-gain"].astype(float))
    user_input["capital-loss"] = np.log1p(user_input["capital-loss"].astype(float))

    missing_cols = set(required_columns) - set(user_input.columns)
    for col in missing_cols:
        user_input[col] = 0

    user_input = user_input[required_columns]
    return user_input


def predict_from_user_input(user_input: pd.DataFrame, model) -> Dict[str, object]:
    """Return a prediction and optional probability for a preprocessed user_input."""
    pred = model.predict(user_input)[0]
    proba = None
    if hasattr(model, "predict_proba"):
        proba_values = model.predict_proba(user_input)[0]
        proba = float(proba_values[int(pred)])
    label = ">=50K" if pred == 1 else "<=50K"
    return {"prediction": int(pred), "label": label, "probability": proba}
