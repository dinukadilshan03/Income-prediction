"""Preprocessing utilities converted from the preprocessing notebook."""
from typing import List
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
TOOLS_DIR = BASE_DIR / "tools"


def build_category_options(raw_df: pd.DataFrame):
    categorical_cols = [
        "workclass",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "native-country",
        "gender",
    ]
    options = {}
    for column in categorical_cols:
        values = raw_df[column].dropna().unique().tolist()
        options[column] = sorted(values)
    return options


def preprocess_and_save_cleaned():
    """Run the preprocessing pipeline from the notebook and save cleaned.csv and encoders.

    This is a convenience function that replicates the preprocessing notebook steps.
    """
    data = pd.read_csv(DATA_DIR / "income.csv", na_values="?")

    # Drop redundant columns
    selected_features = [c for c in data.columns if c not in ["fnlwgt", "educational-num"]]
    data = data[selected_features]

    # Remove duplicates
    data = data.drop_duplicates()

    # Replace placeholder values and fill missing categoricals
    data = data.replace("?", np.nan)
    categorical_cols = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "gender",
        "race",
        "native-country",
    ]
    for col in categorical_cols:
        data[col] = data[col].fillna("Unknown")

    # Convert target to binary
    data["income"] = data["income"].astype(str).str.strip()
    data["income"] = data["income"].apply(lambda x: 1 if x == ">50K" else 0)

    # Outlier handling for numeric columns (IQR)
    numeric_cols_iqr = ["age", "hours-per-week"]

    def remove_outliers_iqr(df, col):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        return df[(df[col] >= lower) & (df[col] <= upper)]

    for col in numeric_cols_iqr:
        data = remove_outliers_iqr(data, col)

    # Log transform skewed monetary features
    for col in ["capital-gain", "capital-loss"]:
        data[col] = np.log1p(data[col].astype(float))

    # Label encode education
    encoder = LabelEncoder()
    data["education"] = encoder.fit_transform(data["education"])
    # Save encoder
    joblib.dump(encoder, TOOLS_DIR / "education_encoder.pkl")

    # One-hot encode remaining nominal columns
    columns_to_encode = [
        "workclass",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "native-country",
        "gender",
    ]
    data_encoded = pd.get_dummies(data, columns=columns_to_encode)

    # Save cleaned dataset
    data_encoded.to_csv(DATA_DIR / "cleaned.csv", index=False)
    return data_encoded


def preprocess_input(
    user_input: pd.DataFrame,
    feature_columns: List[str],
    education_encoder,
    hours_scaler,
) -> pd.DataFrame:
    """Mirror the notebook preprocessing pipeline for fresh user inputs."""

    prepared = user_input.copy()

    # Encode education levels
    prepared["education"] = education_encoder.transform(prepared["education"])

    # Apply the same scaling and transformations as training
    if hours_scaler is not None:
        prepared[["hours-per-week"]] = hours_scaler.transform(prepared[["hours-per-week"]])

    for column in ["capital-gain", "capital-loss"]:
        prepared[column] = np.log1p(prepared[column].astype(float))

    prepared = pd.get_dummies(
        prepared,
        columns=[
            "workclass",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "native-country",
            "gender",
        ],
        drop_first=False,
    )

    # Align with the training feature space
    missing_cols = set(feature_columns) - set(prepared.columns)
    for column in missing_cols:
        prepared[column] = 0

    prepared = prepared[feature_columns]
    return prepared
