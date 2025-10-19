# Income Prediction

A small, reproducible project that predicts whether an individual's annual income exceeds a threshold (e.g., >50K) using tabular census-style data. This repository contains the data, preprocessing code, model artifacts, and a minimal Streamlit app for making single-record predictions.

## Highlights

- End-to-end artifacts: raw and cleaned datasets, preprocessing tools/encoders, trained model(s), and a UI for quick predictions.
- Reusable code: notebooks were converted into modular Python packages under `src/` for easier reuse in scripts or production.
- Lightweight demo: `app.py` is a Streamlit-based frontend to enter a single record and see a prediction.

## Repository layout

- `app.py` — Streamlit frontend for single-record predictions
- `data/` — raw (`income.csv`) and processed (`cleaned.csv`) datasets
- `models/` — persisted model artifacts (for example `best_xgboost_model.pkl`)
- `tools/` — preprocessing artifacts (encoders, scalers, feature lists)
- `src/` — reusable Python modules adapted from notebooks (`utils.py`, `preprocess.py`, `predict.py`)
- `notebooks/` — analysis and experiment notebooks (archived versions in `notebooks/archive/`)

## Data

The repository includes a raw dataset (`data/income.csv`) and a cleaned/processed copy (`data/cleaned.csv`). The preprocessing pipeline in `src/preprocess.py` handles cleaning, encoding categorical variables, and scaling numeric features. If you need to regenerate the cleaned file or the preprocessing artifacts, use the helper function in `src.preprocess`.

## Models & Tools

Trained model artifacts live in `models/` (pickled estimators). Supporting preprocessing objects such as encoders and scalers are stored in `tools/`. `src/predict.py` loads these artifacts and exposes a small prediction helper you can call from scripts or the Streamlit app.

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

The app provides a simple form to enter features and returns the predicted class and probability.

## Example usage (script)

You can also call the prediction helper directly from Python:

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

Check `src/predict.py` for the exact input schema expected by `predict_single`.

## Troubleshooting

- Ensure `models/` and `tools/` contain the expected .pkl artifacts before running `app.py` or `src.predict`.
- If you see errors about missing columns during prediction, regenerate `data/cleaned.csv` and the preprocessing artifacts using the step above.

## Next steps / contributions

- Add unit tests for `src/` modules and simple integration tests for the Streamlit flow.
- Provide a small CLI (`tools/` or `src/cli.py`) to run training, evaluation, and export artifacts reproducibly.
- Package preprocessing + model loading into a lightweight API (FastAPI) for production deployment.

If you'd like, I can implement any of the next steps above (tests, CLI, or an API).
