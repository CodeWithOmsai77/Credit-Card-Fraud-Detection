# 💳 Credit Card Fraud Detection

<p align="center">
  <b>A beginner-friendly Machine Learning project built with Python</b><br>
  Detect potentially fraudulent transactions, compare ML models, and try predictions through a dashboard or API.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Machine%20Learning-scikit--learn-orange" alt="Machine Learning">
  <img src="https://img.shields.io/badge/Dashboard-Streamlit-ff4b4b?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Level-Beginner-success" alt="Beginner friendly">
</p>

---

## 🌟 About the project

Credit card fraud can be difficult to identify because fraudulent transactions are much less common than normal ones. In this project, I use machine learning to explore how transaction data can help identify suspicious activity.

This is a learning project made to practise Python, data handling, model training, and building a small application.

## ✨ Features

- 🤖 **Two ML models** — Logistic Regression and Random Forest
- 📊 **Model comparison** — Precision, recall, F1-score, and confusion matrix data
- 🖥️ **Interactive dashboard** — View results and upload a transaction CSV using Streamlit
- 🔌 **REST API** — Get predictions through FastAPI
- 🧰 **Beginner-friendly code** — Simple files and clear steps to run locally

## 🛠️ Tech stack

| Tool | What it's used for |
| --- | --- |
| Python | Main programming language |
| Pandas | Reading and handling transaction data |
| Scikit-learn | Training and evaluating ML models |
| Joblib | Saving the trained model |
| Streamlit | Building the dashboard |
| FastAPI | Creating the prediction API |

## 📁 Project structure

```text
Credit-Card-Fraud-Detection/
├── data/                 # Put the downloaded dataset here
│   └── creditcard.csv
├── models/               # Created after model training
├── train_model.py        # Train and compare models
├── app.py                # Streamlit dashboard
├── api.py                # FastAPI prediction endpoint
├── requirements.txt      # Python dependencies
├── .gitignore
└── README.md
```

## 🚀 Getting started

### 1. Get the dataset

Download the **Credit Card Fraud Detection** dataset from Kaggle:

👉 https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Extract `creditcard.csv` and put it in the `data/` folder. The target column is `Class`:
- `0` = normal transaction
- `1` = fraudulent transaction

> The dataset is not included in this repository. You need to download it yourself.

### 2. Create a virtual environment

Run these commands in the project folder:

```bash
python -m venv .venv
```

**Windows**
```bash
.venv\Scripts\activate
```

**Linux / macOS**
```bash
source .venv/bin/activate
```

### 3. Install the required libraries

```bash
pip install -r requirements.txt
```

### 4. Train the models

```bash
python train_model.py
```

The script trains both models, compares their F1-scores, and saves the selected model and metrics inside `models/`.

### 5. Start the dashboard

```bash
streamlit run app.py
```

Streamlit will show a local URL in the terminal. Open it in your browser to view model metrics and test a single transaction from a CSV file.

### 6. Start the API

Open a second terminal, activate the same virtual environment, and run:

```bash
uvicorn api:app --reload
```

Then open **http://127.0.0.1:8000/docs** to explore and test the API.

The `POST /predict` endpoint accepts numeric transaction features (`Time`, `V1`–`V28`, and `Amount`) and returns a prediction, label, and estimated fraud probability. If the model has not been trained yet, train it first.

## 📈 How model performance is evaluated

Fraud data is imbalanced: there are far fewer fraudulent transactions than normal ones. For that reason, this project reports more than accuracy:

- **Precision:** How many transactions flagged as fraud were actually fraud in the test data?
- **Recall:** How many of the actual fraud cases did the model identify?
- **F1-score:** A balance between precision and recall.
- **Confusion matrix:** A breakdown of correct and incorrect predictions.

The project selects the model with the better F1-score on a held-out test set. Actual scores will appear after you run the training script.

## 🗺️ Future improvements

- Add automated tests
- Improve the dashboard with charts
- Try more ways to handle imbalanced data
- Deploy the API and dashboard

## ⚠️ Important note

This project is for **education and experimentation only**. It is not a real banking security system. Model predictions can be wrong, so do not use this project for real payment decisions or upload private customer transaction data.

---

<p align="center">
  Made with 🐍 Python while learning Machine Learning
</p>
