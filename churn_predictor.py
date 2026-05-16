# churn_predictor.py
# Fintech Customer Churn Predictor
# Covers: EDA, feature engineering, model training, evaluation & visualisation.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, ConfusionMatrixDisplay)
import os, warnings
warnings.filterwarnings('ignore')

COLORS = {
    'primary': '#1B4F72',
    'accent':  '#2ECC71',
    'warning': '#E74C3C',
    'neutral': '#95A5A6',
    'bg':      '#F8F9FA',
}
os.makedirs('../visuals', exist_ok=True)
os.makedirs('../models', exist_ok=True)

# ── Load Data ─────────────────────────────────────────────────────────────────
df = pd.read_csv('/content/sample_data/customer_data.csv')
print(f"✅ Loaded {len(df)} customer records | Churn rate: {df['churned'].mean()*100:.1f}%\n")

# ═════════════════════════════════════════════════════════════════════════════
# 1. EDA — Churn by Segment & Tenure
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor(COLORS['bg'])
fig.suptitle('Customer Churn — Exploratory Analysis', fontsize=15, fontweight='bold')

# Churn by segment
churn_seg = df.groupby('segment')['churned'].mean() * 100
churn_seg.sort_values(ascending=True).plot(kind='barh', ax=axes[0], color=COLORS['primary'])
axes[0].set_title('Churn Rate by Customer Segment', fontweight='bold')
axes[0].set_xlabel('Churn Rate (%)')
axes[0].set_facecolor(COLORS['bg'])
axes[0].spines[['top', 'right', 'left']].set_visible(False)

# Churn by tenure bucket
df['tenure_bucket'] = pd.cut(df['tenure_months'],
    bins=[0, 12, 24, 48, 120],
    labels=['0–12m', '13–24m', '25–48m', '49m+'])
churn_ten = df.groupby('tenure_bucket', observed=True)['churned'].mean() * 100
churn_ten.plot(kind='bar', ax=axes[1], color=COLORS['warning'], rot=0)
axes[1].set_title('Churn Rate by Tenure', fontweight='bold')
axes[1].set_ylabel('Churn Rate (%)')
axes[1].set_facecolor(COLORS['bg'])
axes[1].spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig('../visuals/churn_eda.png', dpi=150)
plt.close()
print("✅ Chart saved: churn_eda.png")

# ═════════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING & PREPROCESSING
# ═════════════════════════════════════════════════════════════════════════════
df_model = df.copy()

# Encode categoricals
le = LabelEncoder()
for col in ['gender', 'segment', 'preferred_channel']:
    df_model[col] = le.fit_transform(df_model[col])

# Drop non-feature columns
drop_cols = ['customer_id', 'churned', 'tenure_bucket']
X = df_model.drop(columns=drop_cols)
y = df_model['churned']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set: {len(X_train)} | Test set: {len(X_test)}")

# ═════════════════════════════════════════════════════════════════════════════
# 3. TRAIN MODELS
# ═════════════════════════════════════════════════════════════════════════════
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=500),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc     = roc_auc_score(y_test, y_proba)
    results[name] = {'model': model, 'pred': y_pred, 'proba': y_proba, 'auc': auc}
    print(f"\n{'='*40}\n  {name}  |  AUC: {auc:.3f}\n{'='*40}")
    print(classification_report(y_test, y_pred, target_names=['Retained', 'Churned']))

# ═════════════════════════════════════════════════════════════════════════════
# 4. ROC CURVE COMPARISON
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor(COLORS['bg'])
ax.set_facecolor(COLORS['bg'])
plot_colors = [COLORS['primary'], COLORS['warning']]
for (name, res), col in zip(results.items(), plot_colors):
    fpr, tpr, _ = roc_curve(y_test, res['proba'])
    ax.plot(fpr, tpr, label=f"{name} (AUC = {res['auc']:.3f})", color=col, linewidth=2)
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
ax.set_xlabel('False Positive Rate', fontsize=11)
ax.set_ylabel('True Positive Rate', fontsize=11)
ax.set_title('ROC Curve — Model Comparison', fontsize=14, fontweight='bold', pad=15)
ax.legend(fontsize=10)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig('../visuals/roc_curve.png', dpi=150)
plt.close()
print("\n✅ Chart saved: roc_curve.png")

# ═════════════════════════════════════════════════════════════════════════════
# 5. FEATURE IMPORTANCE (Random Forest)
# ═════════════════════════════════════════════════════════════════════════════
rf_model = results['Random Forest']['model']
importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(9, 6))
fig.patch.set_facecolor(COLORS['bg'])
ax.set_facecolor(COLORS['bg'])
importances.tail(10).plot(kind='barh', ax=ax, color=COLORS['primary'])
ax.set_title('Top 10 Feature Importances — Random Forest', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Importance Score')
ax.spines[['top', 'right', 'left']].set_visible(False)
plt.tight_layout()
plt.savefig('../visuals/feature_importance.png', dpi=150)
plt.close()
print("✅ Chart saved: feature_importance.png")

# ═════════════════════════════════════════════════════════════════════════════
# 6. CONFUSION MATRIX
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6, 5))
fig.patch.set_facecolor(COLORS['bg'])
cm = confusion_matrix(y_test, results['Random Forest']['pred'])
disp = ConfusionMatrixDisplay(cm, display_labels=['Retained', 'Churned'])
disp.plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title('Confusion Matrix — Random Forest', fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('../visuals/confusion_matrix.png', dpi=150)
plt.close()
print("✅ Chart saved: confusion_matrix.png")

print("\n🎉 Churn prediction complete! Check /visuals for all charts.")
!pip install shap
import shap

# Choose the Random Forest model for SHAP explanation as it performed better
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)

# Plot summary of feature importances (absolute SHAP values)
shap.summary_plot(shap_values, X_test, feature_names=X.columns, plot_type="bar", show=False)
plt.title('SHAP Feature Importance (Random Forest)')
plt.tight_layout()
plt.savefig('../visuals/shap_feature_importance.png', dpi=150)
plt.show()
plt.close()
print("✅ Chart saved: shap_feature_importance.png")
# This summary plot shows the average impact of each feature on the model's output magnitude. The features are ordered by their importance from top to bottom.
import pandas as pd

X_test_df = pd.DataFrame(X_test, columns=X.columns)
shap.summary_plot(shap_values[:, :, 1], X_test_df, show=False)
plt.title('SHAP Summary Plot (Random Forest - Churn Class)')
plt.tight_layout()
plt.savefig('../visuals/shap_summary_plot_churn.png', dpi=150)
plt.show()
plt.close()
print("✅ Chart saved: shap_summary_plot_churn.png")
# In this summary plot (for the churn class, where red indicates higher feature values and blue indicates lower), each point represents a customer. The position on the x-axis shows the SHAP value for that feature, indicating how much that feature's value contributed to pushing the model's output (churn probability) higher or lower. The color indicates the original value of the feature (red=high, blue=low), and the density of points shows how many customers have similar SHAP values.
from sklearn.model_selection import RandomizedSearchCV

print("\n═════════════════════════════════════════════════════════════════════════════")
print("2. HYPERPARAMETER TUNING — RANDOM FOREST")
print("═════════════════════════════════════════════════════════════════════════════")

# Define the parameter space for Random Forest
param_dist_rf = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Initialize RandomizedSearchCV for Random Forest
random_search_rf = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist_rf,
    n_iter=10, # Number of parameter settings that are sampled
    cv=5,
    verbose=1,
    n_jobs=-1, # Use all available cores
    scoring='roc_auc',
    random_state=42
)

# Fit the random search model
random_search_rf.fit(X_train, y_train)

print(f"\n✅ Best parameters for Random Forest: {random_search_rf.best_params_}")
print(f"✅ Best AUC score for Random Forest: {random_search_rf.best_score_:.3f}")

# Store the best Random Forest model
best_rf_model = random_search_rf.best_estimator_

print("\n═════════════════════════════════════════════════════════════════════════════")
print("3. HYPERPARAMETER TUNING — LOGISTIC REGRESSION")
print("═════════════════════════════════════════════════════════════════════════════")

# Define the parameter space for Logistic Regression
param_dist_lr = {
    'penalty': ['l1', 'l2'],
    'C': np.logspace(-4, 4, 20),
    'solver': ['liblinear', 'saga']
}

# Initialize RandomizedSearchCV for Logistic Regression
random_search_lr = RandomizedSearchCV(
    LogisticRegression(random_state=42, max_iter=500),
    param_distributions=param_dist_lr,
    n_iter=10, # Number of parameter settings that are sampled
    cv=5,
    verbose=1,
    n_jobs=-1, # Use all available cores
    scoring='roc_auc',
    random_state=42
)

# Fit the random search model
random_search_lr.fit(X_train, y_train)

print(f"\n✅ Best parameters for Logistic Regression: {random_search_lr.best_params_}")
print(f"✅ Best AUC score for Logistic Regression: {random_search_lr.best_score_:.3f}")

# Store the best Logistic Regression model
best_lr_model = random_search_lr.best_estimator_
### 4. CROSS-VALIDATION
from sklearn.model_selection import cross_val_score

print("\n═════════════════════════════════════════════════════════════════════════════")
print("4. CROSS-VALIDATION — MODEL EVALUATION")
print("═════════════════════════════════════════════════════════════════════════════")

# Cross-validation for the best Random Forest model
rf_cv_scores = cross_val_score(best_rf_model, X_scaled, y, cv=5, scoring='roc_auc', n_jobs=-1)
print(f"\n✅ Random Forest Cross-Validation AUC: {rf_cv_scores.mean():.3f} (+/- {rf_cv_scores.std():.3f})")

# Cross-validation for the best Logistic Regression model
lr_cv_scores = cross_val_score(best_lr_model, X_scaled, y, cv=5, scoring='roc_auc', n_jobs=-1)
print(f"✅ Logistic Regression Cross-Validation AUC: {lr_cv_scores.mean():.3f} (+/- {lr_cv_scores.std():.3f})")

### 5. COMPARING ADDITIONAL MODELS
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

print("\n═════════════════════════════════════════════════════════════════════════════")
print("5. COMPARING ADDITIONAL MODELS")
print("═════════════════════════════════════════════════════════════════════════════")

# Initialize additional models
additional_models = {
    'Support Vector Machine': SVC(probability=True, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Neural Network (MLP)': MLPClassifier(random_state=42, max_iter=1000)
}

# Include the best models from hyperparameter tuning for comparison
comparison_models = {
    'Best Logistic Regression': best_lr_model,
    'Best Random Forest': best_rf_model,
    **additional_models
}

model_performance = {}

for name, model in comparison_models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    model_performance[name] = auc
    print(f"✅ {name} AUC: {auc:.3f}")

# Display overall comparison
print("\n═════════════════════════════════════════════════════════════════════════════")
print("MODEL COMPARISON — AUC Scores")
print("═════════════════════════════════════════════════════════════════════════════")

for name, auc in sorted(model_performance.items(), key=lambda item: item[1], reverse=True):
    print(f"  {name}: {auc:.3f}")
### 6. ERROR ANALYSIS
print("\n═════════════════════════════════════════════════════════════════════════════")
print("6. ERROR ANALYSIS — MISCLASSIFIED SAMPLES")
print("═════════════════════════════════════════════════════════════════════════════")

# Get predictions from the best Random Forest model
y_pred_rf = best_rf_model.predict(X_test)

# Create a DataFrame for test data with original features for easier analysis
X_test_original_features = pd.DataFrame(scaler.inverse_transform(X_test), columns=X.columns)
X_test_original_features['true_churn'] = y_test.values
X_test_original_features['predicted_churn'] = y_pred_rf
X_test_original_features['prediction_proba_churn'] = best_rf_model.predict_proba(X_test)[:, 1]

# Identify misclassified samples
misclassified_samples = X_test_original_features[X_test_original_features['true_churn'] != X_test_original_features['predicted_churn']]

print(f"\nTotal misclassified samples by Random Forest: {len(misclassified_samples)}")

if not misclassified_samples.empty:
    # Separate false positives and false negatives
    false_positives = misclassified_samples[misclassified_samples['true_churn'] == 0] # Predicted churn, but did not churn
    false_negatives = misclassified_samples[misclassified_samples['true_churn'] == 1] # Predicted no churn, but did churn

    print(f"  False Positives (predicted churn, true no churn): {len(false_positives)}")
    print(f"  False Negatives (predicted no churn, true churn): {len(false_negatives)}")

    print("\n═══════════════════════")
    print("  False Positives Sample")
    print("═══════════════════════")
    if not false_positives.empty:
        print(false_positives.head())
    else:
        print("No False Positives found.")

    print("\n═══════════════════════")
    print("  False Negatives Sample")
    print("═══════════════════════")
    if not false_negatives.empty:
        print(false_negatives.head())
    else:
        print("No False Negatives found.")

    # Further analysis: Look at descriptive statistics of misclassified groups
    print("\n═════════════════════════════════════════════════════════════════════════════")
    print("Descriptive Statistics for False Positives (Predicted Churn, True No Churn)")
    print("═════════════════════════════════════════════════════════════════════════════")
    if not false_positives.empty:
        print(false_positives.describe())
    else:
        print("No False Positives to describe.")

    print("\n═════════════════════════════════════════════════════════════════════════════")
    print("Descriptive Statistics for False Negatives (Predicted No Churn, True Churn)")
    print("═════════════════════════════════════════════════════════════════════════════")
    if not false_negatives.empty:
        print(false_negatives.describe())
    else:
        print("No False Negatives to describe.")

else:
    print("No misclassified samples found")
