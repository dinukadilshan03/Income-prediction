"""Utility helpers extracted from notebooks for data loading and evaluation."""
from pathlib import Path
from typing import Tuple, List, Dict

import pandas as pd
# Import joblib inside functions to avoid import-time failures if the package
# isn't available in the running environment.
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
TOOLS_DIR = BASE_DIR / "tools"


def load_clean_data() -> pd.DataFrame:
    """Load the cleaned dataset used during model training."""
    return pd.read_csv(DATA_DIR / "cleaned.csv")


def load_raw_data() -> pd.DataFrame:
    """Load the original dataset to source categorical value ranges."""
    df = pd.read_csv(DATA_DIR / "income.csv")
    return df.replace("?", "Unknown")


def load_artifacts() -> Tuple[object, object, object]:
    """Load persisted artifacts: model, education encoder and hours scaler if present.

    Returns (model, education_encoder, hours_scaler_or_None)
    """
    import joblib

    model = joblib.load(MODELS_DIR / "best_xgboost_model.pkl")
    education_encoder = joblib.load(TOOLS_DIR / "education_encoder.pkl")
    try:
        hours_scaler = joblib.load(TOOLS_DIR / "hours_scaler.pkl")
    except FileNotFoundError:
        hours_scaler = None
    return model, education_encoder, hours_scaler


def evaluate_model(clean_df: pd.DataFrame, model) -> Dict[str, float]:
    """Evaluate a model on an 80/20 split and return weighted metrics."""
    X = clean_df.drop(columns=["income"])
    y = clean_df["income"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    y_pred = model.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }
    return metrics
