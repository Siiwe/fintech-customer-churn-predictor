"""
generate_data.py
Generates a realistic synthetic banking customer dataset for churn prediction.
"""

import pandas as pd
import numpy as np
import os

np.random.seed(99)
n = 1000

products = ['Savings Account', 'Current Account', 'Credit Card', 'Personal Loan', 'Fixed Deposit']
segments = ['Youth', 'Mass Market', 'Emerging Affluent', 'Affluent']
channels = ['Branch', 'Mobile App', 'Online Banking', 'ATM']

df = pd.DataFrame({
    'customer_id': [f'CUST{str(i).zfill(5)}' for i in range(1, n+1)],
    'age': np.random.randint(18, 70, n),
    'gender': np.random.choice(['Female', 'Male'], n, p=[0.48, 0.52]),
    'segment': np.random.choice(segments, n, p=[0.15, 0.45, 0.25, 0.15]),
    'tenure_months': np.random.randint(1, 120, n),
    'num_products': np.random.choice([1, 2, 3, 4], n, p=[0.4, 0.35, 0.18, 0.07]),
    'monthly_balance_zar': np.round(np.random.exponential(8000, n) + 500, 2),
    'monthly_transactions': np.random.randint(0, 60, n),
    'avg_transaction_value': np.round(np.random.exponential(1500, n), 2),
    'digital_logins_monthly': np.random.randint(0, 40, n),
    'complaints_last_12m': np.random.choice([0, 1, 2, 3], n, p=[0.65, 0.22, 0.09, 0.04]),
    'preferred_channel': np.random.choice(channels, n),
    'credit_score': np.random.randint(300, 850, n),
    'missed_payments_6m': np.random.choice([0, 1, 2, 3], n, p=[0.70, 0.18, 0.08, 0.04]),
})

# Simulate churn probability based on logical drivers
churn_score = (
    (df['tenure_months'] < 12).astype(int) * 0.25 +
    (df['num_products'] == 1).astype(int) * 0.2 +
    (df['monthly_balance_zar'] < 1000).astype(int) * 0.15 +
    (df['complaints_last_12m'] > 1).astype(int) * 0.2 +
    (df['digital_logins_monthly'] < 3).astype(int) * 0.1 +
    (df['missed_payments_6m'] > 1).astype(int) * 0.1 +
    np.random.uniform(0, 0.1, n)
)

df['churned'] = (churn_score > 0.45).astype(int)

os.makedirs('../data', exist_ok=True)
df.to_csv('../data/customer_data.csv', index=False)
print(f"✅ Dataset saved: {len(df)} customers, {df['churned'].sum()} churned ({df['churned'].mean()*100:.1f}%)")
print(df.head())
