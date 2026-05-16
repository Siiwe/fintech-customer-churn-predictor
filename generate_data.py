"""
generate_data.py
Generates a realistic synthetic HR dataset for the People Analytics Dashboard.
"""

import pandas as pd
import numpy as np
import os

np.random.seed(42)
n = 300

departments = ['Finance', 'Technology', 'Operations', 'HR', 'Marketing', 'Risk & Compliance']
job_levels = ['Junior', 'Mid-level', 'Senior', 'Lead', 'Manager']
genders = ['Female', 'Male', 'Non-binary']
race_groups = ['Black African', 'Coloured', 'Indian/Asian', 'White']
exit_reasons = ['Resignation', 'Retrenchment', 'Contract End', 'Retirement', None]

df = pd.DataFrame({
    'employee_id': [f'EMP{str(i).zfill(4)}' for i in range(1, n+1)],
    'department': np.random.choice(departments, n, p=[0.2, 0.25, 0.2, 0.1, 0.15, 0.1]),
    'job_level': np.random.choice(job_levels, n, p=[0.3, 0.3, 0.2, 0.1, 0.1]),
    'gender': np.random.choice(genders, n, p=[0.52, 0.45, 0.03]),
    'race_group': np.random.choice(race_groups, n, p=[0.55, 0.15, 0.1, 0.2]),
    'age': np.random.randint(22, 58, n),
    'tenure_years': np.round(np.random.exponential(3.5, n), 1),
    'salary_zar': np.random.randint(18000, 120000, n),
    'performance_score': np.random.choice([1, 2, 3, 4, 5], n, p=[0.05, 0.1, 0.35, 0.35, 0.15]),
    'training_hours': np.random.randint(0, 80, n),
    'leave_days_taken': np.random.randint(0, 21, n),
    'active': np.random.choice([1, 0], n, p=[0.82, 0.18]),
})

# Assign exit reason only to inactive employees
df['exit_reason'] = df['active'].apply(
    lambda x: None if x == 1 else np.random.choice(exit_reasons[:4], p=[0.55, 0.2, 0.15, 0.1])
)

# Derive attrition flag
df['attrition'] = df['active'].apply(lambda x: 'Yes' if x == 0 else 'No')

os.makedirs('../data', exist_ok=True)
df.to_csv('../data/hr_data.csv', index=False)
print(f"✅ Dataset saved: {len(df)} employees, {df['attrition'].value_counts()['Yes']} attritions")
print(df.head())
