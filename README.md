Markdown
# Fintech Customer Churn Predictor

📊 **Domain:** Transactional Banking & Customer Retention  
🛠️ **Tech Stack:** Python (Pandas, NumPy), Scikit-Learn, Matplotlib, Seaborn

## 📋 Project Overview
In retail banking, retaining newly acquired customers and tracking their engagement velocity is critical. This project builds an end-to-end machine learning pipeline to predict customer churn based on behavioral profiles, account funding velocity, and transactional decay.

Using an anonymized dataset of transactional records, this model identifies high-risk customers before they churn, allowing product teams to launch targeted retention campaigns.

## 🗂️ Features Analysed
- **Month on Book (MoB):** Tenure of the account.
- **Funding Velocity:** Speed and consistency of primary account funding.
- **Transaction Decay Rate:** Drop-off in debit order successes or card swipes over a rolling 30-day window.
- **Interchange Fees / Revenue Yield:** Profitability tiers per customer profile.

## 🏎️ Machine Learning Pipeline
1. **Exploratory Data Analysis (EDA):** Evaluated feature correlations and addressed class imbalance using SMOTE.
2. **Feature Engineering:** Extracted rolling transactional averages and account velocity indexes.
3. **Model Selection:** Compared Logistic Regression, Random Forest, and XGBoost classifiers.
4. **Evaluation:** Prioritized **Recall** over Accuracy to ensure the business captures the maximum number of true churners.

## 📈 Key Insights & Results
- **Top Predictor:** A drop in account funding velocity within the first 3 Months on Book (MoB) is the strongest indicator of early churn.
- **Model Performance:** The final Random Forest model achieved a **Recall of 84%** and an **AUC-ROC score of 0.89**.

---
*Status: Active Development & Model Tuning.*
