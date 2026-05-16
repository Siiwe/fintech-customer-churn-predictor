"""
churn_predictor.py
Fintech Customer Churn Predictor
Covers: EDA, feature engineering, model training, evaluation & visualisation.
"""

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
df = pd.read_csv('../data/customer_data.csv')
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
