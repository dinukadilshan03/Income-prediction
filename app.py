"""Interactive Streamlit frontend for the Income Prediction project."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from src import utils, preprocess, predict

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
TOOLS_DIR = BASE_DIR / "tools"

st.set_page_config(
    page_title="Income Prediction Dashboard",
    page_icon="💰",
    layout="wide",
)

DEFAULT_RACE_CATEGORY = "Other"

RACE_DISPLAY_OPTIONS: Tuple[str, ...] = (
    "African American",
    "Hispanic",
    "Caucasian",
    "Asian",
    "Other",
)

RACE_LABEL_ALIASES: Dict[str, str] = {
    "african american": "Black",
    "african-american": "Black",
    "black": "Black",
    "caucasian": "White",
    "caucassian": "White",
    "white": "White",
    "asian": "Asian-Pac-Islander",
    "asian pac": "Asian-Pac-Islander",
    "asian pacific": "Asian-Pac-Islander",
    "asian pacific islander": "Asian-Pac-Islander",
    "asian-pac-islander": "Asian-Pac-Islander",
    "pacific islander": "Asian-Pac-Islander",
    "amer indian eskimo": "Amer-Indian-Eskimo",
    "american indian": "Amer-Indian-Eskimo",
    "native american": "Amer-Indian-Eskimo",
    "native": "Amer-Indian-Eskimo",
    "amerindian": "Amer-Indian-Eskimo",
    "amer-indian-eskimo": "Amer-Indian-Eskimo",
    "hispanic": "Other",
    "latino": "Other",
    "latina": "Other",
    "latinx": "Other",
    "spanish": "Other",
    "other": "Other",
}


# Workclass display options and mapping to training labels
WORKCLASS_DISPLAY_TO_TRAINING: Dict[str, str] = {
    "Private Sector": "Private",
    "Local Government": "Local-gov",
    "Unknown / Not Provided": "Unknown",
    "Self-Employed (Non-incorporated)": "Self-emp-not-inc",
    "Federal Government": "Federal-gov",
    "State Government": "State-gov",
    "Self-Employed (Incorporated)": "Self-emp-inc",
    "Unpaid": "Without-pay",
    "Never Employed": "Never-worked",
}

WORKCLASS_DISPLAY_OPTIONS = list(WORKCLASS_DISPLAY_TO_TRAINING.keys())


def canonicalize_workclass_value(value: str) -> str:
    """Map a user-friendly workclass label back to the training label.

    If the provided value is already one of the training labels, return it.
    Missing or empty values are mapped to 'Unknown'.
    """
    if pd.isna(value):
        return "Unknown"

    normalized = str(value).strip()
    if not normalized:
        return "Unknown"

    # If user passed a display label, map to training label
    if normalized in WORKCLASS_DISPLAY_TO_TRAINING:
        return WORKCLASS_DISPLAY_TO_TRAINING[normalized]

    # Otherwise, try a relaxed match (case-insensitive)
    lower = normalized.lower()
    for disp, train in WORKCLASS_DISPLAY_TO_TRAINING.items():
        if disp.lower() == lower:
            return train

    # If it's already a training-style label, return as-is
    return normalized


# Occupation display options and mapping to training labels
OCCUPATION_DISPLAY_TO_TRAINING: Dict[str, str] = {
    "Machine Operator / Inspector": "Machine-op-inspct",
    "Agriculture / Farming & Fishing": "Farming-fishing",
    "Protective Services (Police/Fire)": "Protective-serv",
    "Unknown / Not Provided": "Unknown",
    "Other Services": "Other-service",
    "Professional Specialties (e.g., Lawyers, Doctors)": "Prof-specialty",
    "Crafts & Repair": "Craft-repair",
    "Administrative & Clerical": "Adm-clerical",
    "Executive & Managerial": "Exec-managerial",
    "Technical Support": "Tech-support",
    "Sales & Marketing": "Sales",
    "Private Household Services (e.g., Nannies)": "Priv-house-serv",
    "Transportation & Moving": "Transport-moving",
    "Labor & Cleaning Services": "Handlers-cleaners",
    "Military / Armed Forces": "Armed-Forces",
}

OCCUPATION_DISPLAY_OPTIONS = list(OCCUPATION_DISPLAY_TO_TRAINING.keys())


def canonicalize_occupation_value(value: str) -> str:
    """Map a user-friendly occupation label back to the training label.

    Missing or empty values map to 'Unknown'. If the value already looks
    like a training label, return as-is.
    """
    if pd.isna(value):
        return "Unknown"

    normalized = str(value).strip()
    if not normalized:
        return "Unknown"

    if normalized in OCCUPATION_DISPLAY_TO_TRAINING:
        return OCCUPATION_DISPLAY_TO_TRAINING[normalized]

    lower = normalized.lower()
    for disp, train in OCCUPATION_DISPLAY_TO_TRAINING.items():
        if disp.lower() == lower:
            return train

    return normalized


# Marital-status display mapping
MARITAL_DISPLAY_TO_TRAINING: Dict[str, str] = {
    "Never Married": "Never-married",
    "Married (Civilian Spouse)": "Married-civ-spouse",
    "Widowed": "Widowed",
    "Divorced": "Divorced",
    "Separated": "Separated",
    "Married (Spouse Absent)": "Married-spouse-absent",
    "Married (Armed Forces Spouse)": "Married-AF-spouse",
}

MARITAL_DISPLAY_OPTIONS = list(MARITAL_DISPLAY_TO_TRAINING.keys())


def canonicalize_marital_value(value: str) -> str:
    if pd.isna(value):
        return "Unknown"
    normalized = str(value).strip()
    if not normalized:
        return "Unknown"
    if normalized in MARITAL_DISPLAY_TO_TRAINING:
        return MARITAL_DISPLAY_TO_TRAINING[normalized]
    lower = normalized.lower()
    for disp, train in MARITAL_DISPLAY_TO_TRAINING.items():
        if disp.lower() == lower:
            return train
    return normalized


def canonicalize_race_value(value) -> str:
    """Map user-provided race labels to the categories seen during training."""

    if pd.isna(value):
        return DEFAULT_RACE_CATEGORY

    normalized = str(value).strip()
    if not normalized:
        return DEFAULT_RACE_CATEGORY

    simplified = (
        normalized.lower()
        .replace("-", " ")
        .replace("_", " ")
        .replace("/", " ")
    )
    simplified = " ".join(simplified.split())

    return RACE_LABEL_ALIASES.get(simplified, normalized)


@st.cache_data(show_spinner=False)
def load_clean_data() -> pd.DataFrame:
    """Wrapper to `src.utils.load_clean_data` so Streamlit caching works."""
    return utils.load_clean_data()


@st.cache_data(show_spinner=False)
def load_raw_data() -> pd.DataFrame:
    """Wrapper to `src.utils.load_raw_data` so Streamlit caching works."""
    return utils.load_raw_data()


@st.cache_resource(show_spinner=False)
def load_artifacts() -> Tuple[object, object, object | None]:
    """Wrapper to `src.utils.load_artifacts` so Streamlit caching works."""
    return utils.load_artifacts()


@st.cache_resource(show_spinner=False)
def get_feature_columns() -> List[str]:
    """Return the feature ordering used during training."""
    clean_df = load_clean_data()
    return [column for column in clean_df.columns if column != "income"]


def build_category_options(raw_df: pd.DataFrame) -> Dict[str, List[str]]:
    """Collect sorted categorical values to power Streamlit widgets."""
    categorical_cols = [
        "workclass",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "native-country",
        "gender",
    ]
    options: Dict[str, List[str]] = {}
    for column in categorical_cols:
        values = raw_df[column].dropna().unique().tolist()
        options[column] = sorted(values)
    # For race, prefer a curated display list
    options["race"] = list(RACE_DISPLAY_OPTIONS)

    # For workclass, present user-friendly display labels (defined above)
    options["workclass"] = WORKCLASS_DISPLAY_OPTIONS

    # For occupation, present friendly display labels
    options["occupation"] = OCCUPATION_DISPLAY_OPTIONS

    # For marital-status, present friendly display labels
    options["marital-status"] = MARITAL_DISPLAY_OPTIONS
    return options


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

    if "race" in prepared.columns:
        prepared["race"] = prepared["race"].apply(canonicalize_race_value)

    # Map friendly workclass labels back to the training labels
    if "workclass" in prepared.columns:
        prepared["workclass"] = prepared["workclass"].apply(canonicalize_workclass_value)

    # Map friendly occupation labels back to training labels
    if "occupation" in prepared.columns:
        prepared["occupation"] = prepared["occupation"].apply(canonicalize_occupation_value)

    # Map marital-status friendly labels back to training labels
    if "marital-status" in prepared.columns:
        prepared["marital-status"] = prepared["marital-status"].apply(canonicalize_marital_value)

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


def evaluate_model(clean_df: pd.DataFrame, model) -> Dict[str, float]:
    """Recreate the notebook evaluation: 80/20 split with weighted metrics."""
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


def main() -> None:
    clean_df = load_clean_data()
    raw_df = load_raw_data()
    model, education_encoder, hours_scaler = load_artifacts()
    feature_columns = get_feature_columns()
    category_options = build_category_options(raw_df)

    st.title("Income Prediction Assistant")
    st.caption(
        "Interact with the trained XGBoost model to estimate whether an individual's income exceeds $50K."
    )

    

    st.subheader("Provide applicant details")
    with st.form("prediction_form"):
        profile_col, background_col = st.columns((1, 1))

        with profile_col:
            st.markdown("#### Personal & financial profile")
            age = st.number_input("Age", min_value=17, max_value=90, value=35)
            hours_per_week = st.slider("Hours per week", min_value=1, max_value=99, value=40)
            capital_gain = st.number_input("Capital gain", min_value=0, value=0, step=100)
            capital_loss = st.number_input("Capital loss", min_value=0, value=0, step=100)

        with background_col:
            st.markdown("#### Background & demographics")
            edu_col, work_col = st.columns((1, 1))
            education = edu_col.selectbox("Education", options=list(education_encoder.classes_), index=0)
            workclass = work_col.selectbox("Workclass", options=category_options["workclass"], index=0)

            household_col, role_col = st.columns((1, 1))
            marital_status = household_col.selectbox("Marital status", options=category_options["marital-status"], index=0)
            occupation = role_col.selectbox("Occupation", options=category_options["occupation"], index=0)

            culture_col, identity_col = st.columns((1, 1))
            relationship = culture_col.selectbox("Relationship", options=category_options["relationship"], index=0)
            race = identity_col.selectbox("Race", options=category_options["race"], index=0)

            locale_col, gender_col = st.columns((1, 1))
            native_country = locale_col.selectbox(
                "Native country", options=category_options["native-country"], index=0
            )
            gender = gender_col.selectbox("Gender", options=category_options["gender"], index=0)

        submitted = st.form_submit_button("Predict income")

    if submitted:
        user_input = pd.DataFrame(
            [
                {
                    "age": age,
                    "education": education,
                    "capital-gain": capital_gain,
                    "capital-loss": capital_loss,
                    "hours-per-week": hours_per_week,
                    "workclass": workclass,
                    "marital-status": marital_status,
                    "occupation": occupation,
                    "relationship": relationship,
                    "race": race,
                    "native-country": native_country,
                    "gender": gender,
                }
            ]
        )

        try:
            prepared = preprocess_input(user_input, feature_columns, education_encoder, hours_scaler)
            prediction = model.predict(prepared)[0]
            proba = None
            if hasattr(model, "predict_proba"):
                proba_values = model.predict_proba(prepared)[0]
                proba = float(proba_values[int(prediction)])

            label = ">=50K" if prediction == 1 else "<=50K"


            prediction_color = "#2E7D32" if prediction == 1 else "#1E88E5"
            outcome_sentence = (
                "The model expects this profile to earn at least $50K per year."
                if prediction == 1
                else "The model expects this profile to earn $50K or less per year."
            )

            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, {prediction_color}33, rgba(30, 30, 35, 0.1));
                    border: 1px solid {prediction_color}66;
                    padding: 1.5rem;
                    border-radius: 1rem;
                    margin-bottom: 1rem;
                    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.18);
                    backdrop-filter: blur(8px);
                ">
                    <p style="font-size: 0.9rem; margin: 0; color: rgba(255, 255, 255, 0.75);">Predicted income class</p>
                    <h3 style="margin: 0.3rem 0 0.6rem 0; color: #FFFFFF; font-size: 1.6rem;">{label}</h3>
                    <p style="margin: 0; color: rgba(255, 255, 255, 0.7);">{outcome_sentence}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Only display the predicted class and a short outcome sentence.
            # Probability estimates and the model-ready row are intentionally omitted
            # for the commercial product release.
        except Exception as exc:  # pragma: no cover - surfaced directly to the UI
            st.error("An error occurred while preparing the input. Please review your entries.")
            st.exception(exc)


if __name__ == "__main__":
    main()
