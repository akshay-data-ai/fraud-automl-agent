# Fraud AutoML Agent 🤖

A standalone, experimental Streamlit application for automated **binary classification** on labeled CSV data, with fraud detection as the primary example.

## Features

- CSV profiling and missing-value summary
- Explicit target, positive-label, chronological column, and leakage-exclusion controls
- Automatic transaction amount-to-balance ratio and log-amount features when source columns exist
- Five candidate models: Logistic Regression, Decision Tree, Random Forest, Extra Trees and HistGradientBoosting
- Training-only preprocessing (imputation, scaling, categorical one-hot encoding)
- Stratified or chronological **60/20/20** split, with time steps kept together
- Validation Average Precision leaderboard and validation-derived thresholds
- One-time selected-model test evaluation
- Batch predictions and downloadable trusted model pipeline

## Run locally

Requires Python **3.11** (dependencies include scikit-learn 1.6.1).

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy separately

On [Streamlit Community Cloud](https://share.streamlit.io/), choose the GitHub repository `akshay-data-ai/fraud-automl-agent`, branch `main`, main file `app.py`, and **Python 3.11**.

## CSV requirements and limitations

- At most 10 MB, 50,000 rows and approximately 50 predictors in the public demo
- Requires a **binary labeled target** with at least ten examples per class
- User must confirm suspected leakage fields and the positive class
- Choose a chronological column (e.g., `step`) if modeling future events
- Model selection is by validation Average Precision, not a guaranteed banking-ready standard
- No automatic SMOTE, exhaustive tuning, or real-world fraud guarantee
- Never upload sensitive customer records to this public demonstration

This application is **experimental**. Results from synthetic datasets are not evidence of production-grade fraud detection. Do not use exported models for actual banking decisions without independent validation, governance and monitoring.

See [detailed guide](docs/AUTOML_AGENT_GUIDE.md).

### Related, independent project

[PaySim Fraud Detection Dashboard](https://github.com/akshay-data-ai/paysim-fraud-detection-ml) — separately trained Random Forest on PaySim; it is **not** the model trained by this AutoML Agent.
