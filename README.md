# 📊 Customer Churn Prediction using Machine Learning

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikitlearn" />
  <img src="https://img.shields.io/badge/Status-Production%20Ready-brightgreen" />
  <img src="https://img.shields.io/badge/License-MIT-yellow" />
  <img src="https://img.shields.io/badge/PRs-Welcome-blueviolet" />
</p>

<p align="center">
  An end-to-end Machine Learning project that predicts customer churn for a subscription-based business, enabling proactive retention strategies and revenue protection.
</p>

---

## 📌 Project Overview

Customer churn — when a customer stops doing business with a company — is one of the most critical metrics for subscription-based businesses (telecom, SaaS, banking, streaming services). This project builds a robust, interpretable, and production-ready ML pipeline to **predict which customers are likely to churn**, enabling businesses to take timely, targeted retention action.

The solution covers the complete ML lifecycle: data ingestion, exploratory data analysis, feature engineering, model training, hyperparameter tuning, evaluation, and deployment-ready artifacts.

---

## 💼 Business Problem

Acquiring a new customer costs **5–7x more** than retaining an existing one. Despite this, most companies react to churn *after* it happens rather than predicting it in advance. Without a predictive system, businesses:

- Lose high-value customers silently
- Spend retention budgets inefficiently (targeting the wrong customers)
- Lack early-warning signals tied to customer behavior

**Goal:** Build a model that flags at-risk customers *before* they churn, so retention teams can intervene with personalized offers, support, or engagement campaigns.

---

## 🎯 Objectives

- Analyze historical customer data to identify churn drivers
- Engineer meaningful features from raw behavioral/demographic data
- Train and compare multiple classification models
- Optimize for business-relevant metrics (Recall, F1, ROC-AUC)
- Deliver an interpretable, reproducible, and deployable pipeline
- Provide actionable insights for the business/retention team

---

## 🗂️ Dataset Description

| Attribute | Description |
|---|---|
| **Source** | Public telecom customer dataset (e.g., Kaggle Telco Customer Churn) |
| **Rows** | ~7,000 customer records |
| **Target Variable** | `Churn` (Yes/No) |
| **Key Features** | Tenure, Contract Type, Monthly Charges, Total Charges, Internet Service, Payment Method, Tech Support, Online Security, Demographics |
| **Type** | Structured / Tabular data |

> 📁 Place the raw dataset inside `data/raw/` before running the pipeline.

---

## 🔄 ML Workflow

```
Raw Data → Data Cleaning → EDA → Feature Engineering → Train/Test Split
        → Model Training (Multiple Algorithms) → Hyperparameter Tuning
        → Model Evaluation → Model Selection → Serialization → Reporting
```

1. **Data Ingestion & Cleaning** — handle missing values, fix data types, remove duplicates
2. **Exploratory Data Analysis (EDA)** — churn distribution, correlation analysis, feature trends
3. **Feature Engineering** — encoding categorical variables, scaling, feature creation
4. **Model Training** — Logistic Regression, Random Forest, XGBoost, Gradient Boosting
5. **Hyperparameter Tuning** — GridSearchCV / RandomizedSearchCV
6. **Evaluation** — confusion matrix, ROC-AUC, precision-recall tradeoffs
7. **Model Export** — serialized `.pkl` model ready for deployment

---

## 📁 Folder Structure

```
customer-churn-prediction/
│
├── data/
│   ├── raw/                  # Original dataset
│   └── processed/            # Cleaned & feature-engineered data
│
├── notebooks/
│   └── eda_and_modeling.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── predict.py
│
├── models/
│   └── churn_model.pkl
│
├── reports/
│   └── figures/               # Visualizations & charts
│
├── screenshots/                # App / dashboard / output screenshots
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/customer-churn-prediction.git
cd customer-churn-prediction

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

```bash
# Run preprocessing
python src/data_preprocessing.py

# Train the model
python src/train_model.py

# Evaluate model performance
python src/evaluate_model.py

# Predict churn for new customer data
python src/predict.py --input data/processed/new_customers.csv
```

---

## 🛠️ Technologies Used

| Category | Tools |
|---|---|
| **Language** | Python 3.10+ |
| **Data Handling** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn, XGBoost |
| **Model Tuning** | GridSearchCV, Optuna (optional) |
| **Serialization** | Joblib / Pickle |
| **Environment** | Jupyter Notebook, Virtualenv |

---

## 🧠 Model Pipeline

```
Input Data → Preprocessing Pipeline (Imputer + Scaler + Encoder)
           → Feature Selection
           → Classifier (Best Performing: XGBoost)
           → Probability Output → Churn Decision Threshold
```

The final pipeline is wrapped using `sklearn.pipeline.Pipeline` to ensure consistent preprocessing during both training and inference, minimizing train-serve skew.

---

## 📈 Evaluation Metrics

| Metric | Why It Matters |
|---|---|
| **Accuracy** | Overall correctness (used cautiously due to class imbalance) |
| **Precision** | Minimizes false-positive retention costs |
| **Recall** | Ensures actual churners are correctly identified |
| **F1-Score** | Balance between precision and recall |
| **ROC-AUC** | Measures model's ability to distinguish churn vs. non-churn |
| **Confusion Matrix** | Visual breakdown of prediction errors |

> ✅ Final Model Performance: **ROC-AUC ~0.85**, **Recall ~0.80** *(update with your actual results)*

---

## 🔮 Future Improvements

- [ ] Deploy as a REST API using FastAPI/Flask
- [ ] Build an interactive Streamlit dashboard
- [ ] Add SHAP/LIME for model explainability
- [ ] Implement automated retraining pipeline (CI/CD + Airflow)
- [ ] Integrate with a live CRM/database for real-time scoring
- [ ] Add Docker containerization for portability

---

## 🖼️ Screenshots

> Add visuals of EDA charts, model performance, or dashboard here.

| EDA Insights | Model Performance |
|---|---|
| `screenshots/eda_overview.png` | `screenshots/confusion_matrix.png` |

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Your Name**
*AI / Machine Learning Engineer*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/your-profile)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/your-username)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-green?logo=google-chrome)](https://your-portfolio.com)

---

<p align="center">⭐ If you found this project useful, consider giving it a star!</p>
