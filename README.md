# Income Prediction

[![python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/) [![streamlit](https://img.shields.io/badge/streamlit-demo-orange)](#)

A small, reproducible project that predicts whether an individual's annual income exceeds a threshold (for example >50K) using tabular census-style data. It includes preprocessing pipelines, trained model artifacts, and a minimal Streamlit app for quick, single-record predictions.

## Table of contents

- [Highlights](#highlights)
- [Repository layout](#repository-layout)
- [Data](#data)
- [Models & tools](#models--tools)
- [Quick start (local)](#quick-start-local)
- [Example usage (script)](#example-usage-script)
- [Troubleshooting](#troubleshooting)
- [Next steps / contributions](#next-steps--contributions)

## Highlights

- End-to-end: raw + cleaned data, preprocessing artifacts, trained model(s), and a demo UI.
- Reusable: notebooks were converted into modular `src/` packages for reuse.
- Demo: `app.py` provides an interactive Streamlit interface for single-record predictions.

## Repository layout

- `app.py` — Streamlit frontend for single-record predictions
- `data/` — raw (`income.csv`) and processed (`cleaned.csv`) datasets
- `models/` — persisted model artifacts (e.g., `best_xgboost_model.pkl`)
- `tools/` — preprocessing artifacts (encoders, scalers, feature lists)
- `src/` — reusable Python modules adapted from notebooks (`utils.py`, `preprocess.py`, `predict.py`)
- `notebooks/` — analysis and experiment notebooks (archived versions in `notebooks/archive/`)

## Data

The repository includes a raw dataset (`data/income.csv`) and a cleaned/processed copy (`data/cleaned.csv`). The preprocessing pipeline in `src/preprocess.py` handles cleaning, encoding categorical variables, and scaling numeric features. Use the helper `preprocess_and_save_cleaned` to regenerate cleaned data and required preprocessing artifacts.

## Models & tools

Trained model artifacts live in `models/` (pickled estimators). Supporting preprocessing objects (encoders and scalers) are stored in `tools/`. `src/predict.py` provides helpers to load artifacts and run predictions from scripts or the Streamlit app.

## Quick start (local)

1. Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. (Optional) Regenerate cleaned data and preprocessing artifacts:

```bash
python -c "from src.preprocess import preprocess_and_save_cleaned; preprocess_and_save_cleaned()"
```

3. Run the Streamlit app and open the URL shown in the terminal:

```bash
streamlit run app.py
```

The app shows a simple form to enter features and returns a predicted class and probability.

## Example usage (script)

Call the prediction helper directly from Python:

```python
from src.predict import predict_single

sample = {
    "age": 37,
    "workclass": "Private",
    "education": "Bachelors",
    "hours_per_week": 40,
    # include other required features as in `data/cleaned.csv`
}

pred = predict_single(sample)
print(pred)
```

Refer to `src/predict.py` for the exact input schema expected by `predict_single`.

## Troubleshooting

- Ensure `models/` and `tools/` contain the expected `.pkl` artifacts before running `app.py` or `src.predict`.
- If you see missing-column errors during prediction, regenerate `data/cleaned.csv` and the preprocessing artifacts using the step above.

## Next steps / contributions

- Add unit tests for `src/` modules and integration tests for the Streamlit flow.
- Provide a CLI (`tools/` or `src/cli.py`) to run preprocessing, training, and artifact export reproducibly.
- Package preprocessing + model loading as a lightweight API (FastAPI) for deployment.

If you'd like, I can implement any of the next steps above (tests, CLI, or an API).
