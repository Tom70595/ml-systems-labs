import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import joblib
import hashlib
import yaml
import os

# Setup folder
os.makedirs('results', exist_ok=True)

print("--- TASK 2: INGESTION ---")
adult = fetch_openml(name='adult', version=2, as_frame=True)
df = adult.frame
df = df.replace('?', np.nan)
print(f"Row count: {df.shape[0]}, Column count: {df.shape[1]}")

print("\n--- TASK 3: VALIDATION REPORT ---")
missing = df.isnull().sum()
print("Missing values per column:\n", missing[missing > 0])

df_num = df.copy()
for col in ['age', 'hours-per-week', 'capital-gain', 'capital-loss']:
    df_num[col] = pd.to_numeric(df_num[col], errors='coerce')

age_viol = len(df_num[(df_num['age'] < 17) | (df_num['age'] > 90)])
hours_viol = len(df_num[(df_num['hours-per-week'] < 1) | (df_num['hours-per-week'] > 99)])
gain_viol = len(df_num[(df_num['capital-gain'] < 0) | (df_num['capital-gain'] > 99999)])
loss_viol = len(df_num[(df_num['capital-loss'] < 0) | (df_num['capital-loss'] > 4356)])

print(f"Violations -> Age: {age_viol}, Hours: {hours_viol}, Gain: {gain_viol}, Loss: {loss_viol}")

print("\n--- TASK 4: CLEANING & TRANSFORMATION ---")
X = df.drop('class', axis=1)
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

train_df = pd.concat([X_train, y_train], axis=1).drop_duplicates()
X_train = train_df.drop('class', axis=1)
y_train = train_df['class']

num_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
cat_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']

num_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
cat_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(transformers=[('num', num_transformer, num_cols), ('cat', cat_transformer, cat_cols)])
preprocessor.fit(X_train)
joblib.dump(preprocessor, 'results/preprocessor.joblib')
print("Fitted transformer saved.")

print("\n--- TASK 5: BASELINE VS ENGINEERED ---")
# Baseline
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(preprocessor.transform(X_train), y_train)
y_pred = model.predict(preprocessor.transform(X_test))
print(f"Baseline F1: {f1_score(y_test, y_pred, pos_label='>50K'):.4f}")

# Engineered
def add_features(X_data):
    X_new = X_data.copy()
    for c in ['capital-gain', 'capital-loss', 'hours-per-week', 'age']:
        X_new[c] = pd.to_numeric(X_new[c], errors='coerce')
    X_new['capital_net'] = X_new['capital-gain'] - X_new['capital-loss']
    X_new['hours_per_age'] = X_new['hours-per-week'] / (X_new['age'] - 16)
    return X_new

X_train_eng = add_features(X_train)
X_test_eng = add_features(X_test)
num_cols_eng = num_cols + ['capital_net', 'hours_per_age']

preprocessor_eng = ColumnTransformer(transformers=[('num', num_transformer, num_cols_eng), ('cat', cat_transformer, cat_cols)])
preprocessor_eng.fit(X_train_eng)
joblib.dump(preprocessor_eng, 'results/preprocessor.joblib') 

model_eng = LogisticRegression(max_iter=1000, random_state=42)
model_eng.fit(preprocessor_eng.transform(X_train_eng), y_train)
y_pred_eng = model_eng.predict(preprocessor_eng.transform(X_test_eng))
print(f"Engineered F1: {f1_score(y_test, y_pred_eng, pos_label='>50K'):.4f}")


print("\n--- TASK 6: LABEL QUALITY CHECK ---")

# Find duplicate rows based on features only
features_only = df.drop('class', axis=1)
duplicates = df[df.duplicated(subset=features_only.columns, keep=False)]

# Create a single string key for each row so Pandas doesn't try to group by 14 different columns
duplicates['row_key'] = duplicates[features_only.columns].astype(str).agg('_'.join, axis=1)

# Count how many unique labels exist for each row_key
conflicts = duplicates.groupby('row_key')['class'].nunique()

# Sum up the ones that have more than 1 unique label
conflict_count = (conflicts > 1).sum()

print(f"Number of duplicate feature rows with conflicting income labels: {conflict_count}")

print("\n--- TASK 7: REPRODUCIBILITY FILES ---")
df_string = df.to_csv(index=False)
hash_val = hashlib.sha256(df_string.encode()).hexdigest()

config = {'dataset': 'adult', 'version': 2, 'hash': hash_val, 'train_size': 0.7, 'random_state': 42}
with open('results/config.yaml', 'w') as f:
    yaml.dump(config, f)

data_card = f"""# Data Card: UCI Adult Dataset
**Source:** OpenML (v2) | **Size:** {df.shape[0]} rows | **Hash:** {hash_val}
**Biases:** 1994 Census data, skewed towards white males.
**Privacy:** Contains demographic data, requires masking in production.
"""
with open('results/data_card.md', 'w') as f:
    f.write(data_card)

print("Saved config.yaml and data_card.md. Pipeline complete.")