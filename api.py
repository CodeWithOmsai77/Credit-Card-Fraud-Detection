from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException


MODEL_PATH = Path("models/fraud_model.pkl")

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="A small API for the beginner-friendly fraud detection project.",
    version="1.0.0",
)


def get_saved_model():
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Model not found. Run 'python train_model.py' first.",
        )
    return joblib.load(MODEL_PATH)


@app.get("/")
def home():
    return {
        "message": "Credit Card Fraud Detection API",
        "docs": "/docs",
        "prediction_endpoint": "POST /predict",
    }


@app.post("/predict")
def predict(transaction: dict):
    saved = get_saved_model()
    features = saved["features"]

    missing = [feature for feature in features if feature not in transaction]
    if missing:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Missing required transaction features.",
                "missing": missing,
            },
        )

    try:
        row = pd.DataFrame(
            [{feature: float(transaction[feature]) for feature in features}],
            columns=features,
        )
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="All transaction feature values must be numbers.",
        )

    model = saved["model"]
    prediction = int(model.predict(row)[0])
    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(row)[0][1])

    return {
        "prediction": prediction,
        "label": "Potential fraud" if prediction == 1 else "Likely normal",
        "fraud_probability": probability,
        "model": saved["name"],
        "note": "Educational prediction only; not for real payment decisions.",
    }
