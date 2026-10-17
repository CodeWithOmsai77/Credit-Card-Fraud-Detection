from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = Path("models/fraud_model.pkl")
METRICS_PATH = Path("models/metrics.json")

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳")
st.title("💳 Credit Card Fraud Detection")
st.write("A beginner-friendly machine learning project.")

if not MODEL_PATH.exists() or not METRICS_PATH.exists():
    st.warning("The model has not been trained yet.")
    st.code("python train_model.py")
    st.info("First download creditcard.csv and put it inside the data folder. See README.md.")
    st.stop()

with open(METRICS_PATH, "r", encoding="utf-8") as file:
    metrics = json.load(file)

st.subheader("Model comparison")
comparison = pd.DataFrame(metrics).T
st.dataframe(comparison[["precision", "recall", "f1_score"]].style.format("{:.4f}"))

saved = joblib.load(MODEL_PATH)
st.success(f"Selected model (based on F1-score): {saved['name']}")

st.subheader("Try a transaction")
st.write("Upload a CSV with exactly one row and all feature columns except Class.")
uploaded_file = st.file_uploader("Transaction CSV", type=["csv"])

if uploaded_file is not None:
    try:
        transaction = pd.read_csv(uploaded_file)
        if len(transaction) != 1:
            st.error("Please upload a CSV with exactly one row.")
        else:
            features = saved["features"]
            missing = [name for name in features if name not in transaction.columns]
            if missing:
                st.error("Missing columns: " + ", ".join(missing))
            else:
                transaction = transaction[features]
                prediction = int(saved["model"].predict(transaction)[0])
                if prediction == 1:
                    st.error("The model flags this transaction as potentially fraudulent.")
                else:
                    st.success("The model classifies this transaction as likely normal.")

                if hasattr(saved["model"], "predict_proba"):
                    probability = saved["model"].predict_proba(transaction)[0][1]
                    st.metric("Estimated fraud probability", f"{probability:.2%}")
                st.caption("This is an educational estimate, not a guarantee.")
    except Exception as error:
        st.error(f"Could not process the file: {error}")
