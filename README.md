# Credit Card Fraud Detection

This is my beginner-friendly Machine Learning project for detecting suspicious credit card transactions using Python.

## What it does
- Trains Logistic Regression and Random Forest models.
- Compares precision, recall and F1-score.
- Shows results in a simple Streamlit dashboard.
- Provides a REST API using FastAPI.

This is a learning project, not a real banking security system.

## 1. Get the dataset
Download the Credit Card Fraud Detection dataset from Kaggle:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Extract `creditcard.csv` and place it inside a folder named `data`:

```text
Credit-Card-Fraud-Detection/
├── data/
│   └── creditcard.csv
├── train_model.py
├── app.py
├── api.py
└── requirements.txt
```

The target column is `Class`: 1 means fraud and 0 means normal.

## 2. Install dependencies
Run these commands from the project folder:

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

Linux/macOS:
```bash
source .venv/bin/activate
```

Then:
```bash
pip install -r requirements.txt
```

## 3. Train the models
```bash
python train_model.py
```

This saves the selected model and evaluation results in a local `models` folder.

## 4. Run the dashboard
```bash
streamlit run app.py
```

You can view model metrics and upload a CSV containing exactly one transaction row (all feature columns except `Class`) for a prediction.

## 5. Run the REST API
Open a second terminal, activate the virtual environment, then run:
```bash
uvicorn api:app --reload
```

Open http://127.0.0.1:8000/docs to try the API. The `POST /predict` endpoint expects `Time`, `V1` through `V28`, and `Amount` as numeric JSON fields. The API checks for missing fields and returns a prediction with an estimated fraud probability.

## Notes
- Model performance is measured using a held-out test set.
- Fraud cases are rare, so accuracy alone is not a good measure; precision, recall and F1-score are included.
- The dataset and generated model files are not stored in this repository.
- Do not use real customer payment data. Predictions can be wrong and must not be used for actual payment decisions.
