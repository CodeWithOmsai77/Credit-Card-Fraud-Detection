from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("data/creditcard.csv")
MODEL_DIR = Path("models")


def main():
    if not DATA_PATH.exists():
        print("Dataset not found!")
        print("Download creditcard.csv from the Kaggle link in README.md")
        print("and place it inside the data folder.")
        return

    data = pd.read_csv(DATA_PATH)

    if "Class" not in data.columns:
        raise ValueError("The dataset must contain a 'Class' column.")

    # Class is the answer we want the model to learn.
    X = data.drop(columns=["Class"])
    y = data["Class"]

    # Keep some transactions aside for testing the trained models.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            )),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = {}
    best_model = None
    best_name = None
    best_f1 = -1

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)

        results[name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "confusion_matrix": confusion_matrix(
                y_test, predictions
            ).tolist(),
            "classification_report": classification_report(
                y_test,
                predictions,
                output_dict=True,
                zero_division=0,
            ),
        }

        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-score:  {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_name = name

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save the best model and the exact feature names it expects.
    joblib.dump(
        {"model": best_model, "features": list(X.columns), "name": best_name},
        MODEL_DIR / "fraud_model.pkl",
    )

    with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print(f"\nBest model by F1-score: {best_name}")
    print(f"Saved model and results in: {MODEL_DIR}")


if __name__ == "__main__":
    main()
