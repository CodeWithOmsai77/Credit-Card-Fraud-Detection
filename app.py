from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = Path("models/fraud_model.pkl")
METRICS_PATH = Path("models/metrics.json")

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳")
st.title("💳 Credit Card Fraud Detection")
st.write("A small machine learning project to explore suspicious transactions.")

if not MODEL_PATH.exists() or not METRICS_PATH.exists():
    st.warning("The model has not been trained yet.")
    st.code("python train_model.py")
    st.info("First download creditcard.csv and place it inside the data folder. See README.md.")
    st.stop()

with open(METRICS_PATH, "r", encoding="utf-8") as file:
    metrics = json.load(file)

st.subheader("Model comparison")
comparison = pd.DataFrame(metrics).T
st.dataframe(comparison[["precision", "recall", "f1_score"]].style.format("{:.4f}"))

best_model_info = joblib.load(MODEL_PATH)
st.success(f"Model selected using test-set F1-score: {best_model_info['name']}")

st.subheader("Try a transaction")
st.write(
    "Upload a CSV with exactly one transaction row. It must contain all feature "
    "columns from the training dataset except the Class column."
)

uploaded_file = st.file_uploader("Upload one transaction as CSV", type=["csv"])

if uploaded_file is not None:
    try:
        transaction = pd.read_csv(uploaded_file)
        if len(transaction) != 1:
            st.error("Please upload a CSV containing exactly one row.")
        else:
            expected_features = best_model_info["features"]
            missing = [column for column in expected_features if column not in transaction.columns]

            if missing:
                st.error("Missing columns: " + ", ".join(missing))
            else:
                transaction = transaction[expected_features]
                prediction = int(best_model_info["model"].predict(transaction)[0])
                if prediction == 1:
                    st.error("The model flags this transaction as potentially fraudulent.")
                else:
                    st.success("The model classifies this transaction as likely normal.")

                if hasattr(best_model_info["model"], "predict_proba"):
                    probability = best_model_info["model"].predict_proba(transaction)[0][1]
                    st.metric("Estimated fraud probability", f"{probability:.2%}")
                st.caption("This is a model estimate, not a guarantee or a real banking decision.")
    except Exception as error:
        st.error(f"Could not process this file: {error}")
