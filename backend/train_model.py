"""
Loan Approval Prediction - Model Tuning Script (v3.0)
========================================================
Real-Life Banking System with CIBIL Score & Hyperparameter Tuning

This script:
1. Loads 300,000 records from the LendingClub dataset
2. Maps FICO scores → CIBIL scores (Indian 300–900 scale)
3. Engineers real banking features including composite ratios (LTI)
4. Uses RandomizedSearchCV to perfectly tune the HistGradientBoosting
5. Validates for overfitting/underfitting with CV
6. Saves model.pkl + model_metadata.json
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, f1_score
)
from sklearn.inspection import permutation_importance
import pickle
import json
import os
import sys
import io
import warnings
from scipy.stats import loguniform, randint, uniform

warnings.filterwarnings('ignore')

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ─── Configuration ─────────────────────────────────────────────────
SAMPLE_SIZE = 300_000          # Scaled up to 300,000 records
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5                   

# ─── 1. Load Dataset ──────────────────────────────────────────────
print("=" * 65)
print("  LOAN APPROVAL PREDICTION v3.0 — TUNING & SCALING")
print("=" * 65)

dataset_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'accepted_2007_to_2018Q4.csv'
)

print(f"\n[DATA] Loading LendingClub accepted dataset...")
use_cols = [
    'loan_amnt', 'term', 'int_rate', 'grade', 'emp_length',
    'home_ownership', 'annual_inc', 'purpose', 'dti',
    'fico_range_low', 'fico_range_high', 'open_acc', 'revol_util',
    'total_acc', 'pub_rec', 'mort_acc', 'loan_status',
    'delinq_2yrs', 'inq_last_6mths'
]

df = pd.read_csv(dataset_path, usecols=use_cols, low_memory=False)

# ─── 2. Target Variable ───────────────────────────────────────────
valid_statuses = {'Fully Paid': 1, 'Charged Off': 0, 'Default': 0}
df = df[df['loan_status'].isin(valid_statuses.keys())].copy()
df['target'] = df['loan_status'].map(valid_statuses)

# ─── 3. Balance & Sample ──────────────────────────────────────────
print(f"\n[SAMPLE] Balancing and scaling up to {SAMPLE_SIZE:,} records...")

approved = df[df['target'] == 1]
rejected = df[df['target'] == 0]

n_per_class = min(SAMPLE_SIZE // 2, len(rejected))
approved_sample = approved.sample(n=n_per_class, random_state=RANDOM_STATE)
rejected_sample = rejected.sample(n=min(n_per_class, len(rejected)), random_state=RANDOM_STATE)

df = pd.concat([approved_sample, rejected_sample], ignore_index=True)
df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

print(f"   Balanced dataset: {len(df):,} rows")

# ─── 4. Feature Engineering ───────────────────────────────────────
print(f"\n[FEATURES] Engineering banking features & composite ratios...")

fico_avg = (pd.to_numeric(df['fico_range_low'], errors='coerce') + pd.to_numeric(df['fico_range_high'], errors='coerce')) / 2
df['cibil_score'] = (300 + (fico_avg - 300) * (600 / 550)).clip(300, 900).round(0)
df['cibil_score'] = df['cibil_score'].fillna(df['cibil_score'].median())

df['annual_income'] = pd.to_numeric(df['annual_inc'], errors='coerce').clip(upper=500000)
df.loc[(df['annual_income'].isna()) | (df['annual_income'] <= 0), 'annual_income'] = df['annual_income'].median()

df['loan_amount'] = pd.to_numeric(df['loan_amnt'], errors='coerce').fillna(df['loan_amnt'].median())
df['int_rate_clean'] = pd.to_numeric(df['int_rate'], errors='coerce').fillna(df['int_rate'].median())
df['dti_clean'] = pd.to_numeric(df['dti'], errors='coerce').clip(0, 60).fillna(df['dti'].median())

def parse_emp_length(val):
    if pd.isna(val): return np.nan
    val = str(val).strip()
    if val == '< 1 year': return 0.5
    if val == '10+ years': return 10.0
    if 'year' in val:
        try: return float(val.split()[0])
        except: return np.nan
    return np.nan

df['emp_length_years'] = df['emp_length'].apply(parse_emp_length).fillna(df['emp_length'].apply(parse_emp_length).median())
df['term_months'] = df['term'].astype(str).str.strip().str.extract(r'(\d+)')[0].astype(float).fillna(36)
df['open_acc_clean'] = pd.to_numeric(df['open_acc'], errors='coerce').fillna(df['open_acc'].median())
df['revol_util_clean'] = pd.to_numeric(df['revol_util'], errors='coerce').clip(0, 150).fillna(df['revol_util'].median())
df['total_acc_clean'] = pd.to_numeric(df['total_acc'], errors='coerce').fillna(df['total_acc'].median())
df['pub_rec_clean'] = pd.to_numeric(df['pub_rec'], errors='coerce').fillna(0)
df['mort_acc_clean'] = pd.to_numeric(df['mort_acc'], errors='coerce').fillna(0)
df['delinq_2yrs_clean'] = pd.to_numeric(df['delinq_2yrs'], errors='coerce').fillna(0)
df['inq_last_6mths_clean'] = pd.to_numeric(df['inq_last_6mths'], errors='coerce').fillna(0)

# --- Composite Features ---
# 1. Loan to Income Ratio
df['loan_to_income'] = (df['loan_amount'] / df['annual_income']).clip(0, 5).fillna(0)
# 2. Credit Burden (DTI * Utilization)
df['credit_burden'] = (df['dti_clean'] * (df['revol_util_clean'] / 100)).fillna(0)

# ─── 5. Encode Categoricals ───────────────────────────────────────
home_map = {'RENT': 0, 'OWN': 1, 'MORTGAGE': 2, 'OTHER': 3, 'NONE': 3, 'ANY': 3}
df['home_encoded'] = df['home_ownership'].map(home_map).fillna(3).astype(int)

purpose_le = LabelEncoder()
df['purpose_clean'] = df['purpose'].fillna('other').str.lower().str.strip()
df['purpose_encoded'] = purpose_le.fit_transform(df['purpose_clean'])

grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
df['grade_encoded'] = df['grade'].map(grade_map).fillna(4).astype(int)

encoding_maps = {
    'home_ownership': home_map,
    'purpose': {label: int(idx) for idx, label in enumerate(purpose_le.classes_)},
    'grade': grade_map
}

# ─── 6. Prepare Feature Matrix ────────────────────────────────────
feature_cols = [
    'cibil_score', 'annual_income', 'loan_amount', 'int_rate_clean', 'dti_clean',
    'emp_length_years', 'term_months', 'home_encoded', 'purpose_encoded',
    'grade_encoded', 'open_acc_clean', 'revol_util_clean', 'total_acc_clean', 'pub_rec_clean',
    'mort_acc_clean', 'delinq_2yrs_clean', 'inq_last_6mths_clean',
    'loan_to_income', 'credit_burden' # New engineered features
]

feature_api_names = [
    'cibil_score', 'annual_income', 'loan_amount', 'int_rate', 'dti',
    'emp_length_years', 'term_months', 'home_encoded', 'purpose_encoded',
    'grade_encoded', 'open_acc', 'revol_util', 'total_acc', 'pub_rec',
    'mort_acc', 'delinq_2yrs', 'inq_last_6mths',
    'loan_to_income', 'credit_burden'
]

feature_display_names = [
    'CIBIL Score', 'Annual Income', 'Loan Amount', 'Interest Rate', 'DTI',
    'Employment Length', 'Loan Term', 'Home Ownership', 'Loan Purpose',
    'Credit Grade', 'Open Accounts', 'Revolving Utilization', 'Total Accounts',
    'Public Records', 'Mortgage Accounts', 'Delinquencies (2yr)', 'Inquiries (6mo)',
    'Loan-to-Income (LTI)', 'Total Credit Burden'
]

X = df[feature_cols].copy()
y = df['target'].copy()

# NaN check fallback
for col in feature_cols:
    X[col] = X[col].fillna(X[col].median())

print(f"   Features: {len(feature_cols)} (Added 2 Composite Ratios)")
print(f"   Samples: {len(X):,}")

# ─── 7. Train/Test Split ──────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# ─── 8. Hyperparameter Tuning ─────────────────────────────────────
print(f"\n[TUNING] Running RandomizedSearchCV to find perfect model fit...")

param_distributions = {
    'learning_rate': loguniform(0.01, 0.2),
    'max_iter': randint(100, 400),
    'max_depth': [5, 6, 8, 10, None],
    'min_samples_leaf': randint(20, 100),
    'l2_regularization': uniform(0.0, 1.0),
    'max_bins': [255]
}

base_model = HistGradientBoostingClassifier(
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=RANDOM_STATE,
)

# Limit to 15 iterations to ensure it finishes in a reasonable time (~5 mins)
search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_distributions,
    n_iter=15,
    scoring='roc_auc',  # Optimize for AUC-ROC
    cv=3,               # 3-Fold CV for speed during tuning
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=1
)

search.fit(X_train, y_train)

best_model = search.best_estimator_
print(f"\n[TUNING] Best Hyperparameters Found:")
for k, v in search.best_params_.items():
    if isinstance(v, float):
        print(f"   {k}: {v:.4f}")
    else:
        print(f"   {k}: {v}")

# ─── 9. Robust Overfit Verification (5-Fold CV) ───────────────────
print(f"\n[CV] Verifying tuned model with Stratified 5-Fold Cross-Validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)

print(f"   CV Mean Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
if cv_scores.std() < 0.02:
    print(f"   ✅ Variance is minimal, highly stable model.")

# ─── 10. Final Evaluation ─────────────────────────────────────────
y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)
y_proba_test = best_model.predict_proba(X_test)[:, 1]

train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)
test_auc = roc_auc_score(y_test, y_proba_test)
test_f1 = f1_score(y_test, y_pred_test)

overfit_gap = train_accuracy - test_accuracy

print(f"\n{'='*65}")
print(f"  TUNED MODEL EVALUATION RESULTS")
print(f"{'='*65}")
print(f"   Train Accuracy: {train_accuracy:.4f}")
print(f"   Test Accuracy:  {test_accuracy:.4f}")
print(f"   Test AUC-ROC:   {test_auc:.4f}")
print(f"   Test F1 Score:  {test_f1:.4f}")

if overfit_gap < 0.02:
    print(f"\n   ✅ Overfitting Check: PASSED (gap={overfit_gap:.4f} < 0.02)")
else:
    print(f"\n   ⚠️ Overfitting Check: WARNING (gap={overfit_gap:.4f} >= 0.02)")

print(f"\n[IMPORTANCE] Calculating Permutation Feature Importance...")
perm_importance = permutation_importance(best_model, X_test, y_test, n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1)
feature_importance = dict(zip(feature_api_names, [round(float(x), 4) for x in perm_importance.importances_mean]))
sorted_features = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

# ─── 11. Metadata Generation ──────────────────────────────────────
metadata = {
    'model_type': 'Tuned HistGradientBoosting',
    'accuracy': round(float(test_accuracy), 4),
    'accuracy_percent': round(float(test_accuracy * 100), 1),
    'train_accuracy': round(float(train_accuracy), 4),
    'cv_mean_accuracy': round(float(cv_scores.mean()), 4),
    'cv_std_accuracy': round(float(cv_scores.std()), 4),
    'auc_roc': round(float(test_auc), 4),
    'f1_score': round(float(test_f1), 4),
    'overfit_gap': round(float(overfit_gap), 4),
    'n_estimators_actual': int(best_model.n_iter_),
    'learning_rate': round(float(search.best_params_['learning_rate']), 4),
    'max_depth': search.best_params_['max_depth'] or 'None',
    'feature_columns': feature_api_names,
    'feature_display_names': feature_display_names,
    'encoding_maps': {k: {str(kk): int(vv) for kk, vv in v.items()} for k, v in encoding_maps.items()},
    'feature_importance': sorted_features,
    'dataset_stats': {
        'total_records': len(df),
        'approved_count': int((df['target'] == 1).sum()),
        'rejected_count': int((df['target'] == 0).sum()),
        'approval_rate': round(int((df['target'] == 1).sum()) / len(df) * 100, 1),
    }
}

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(best_model, f)

metadata_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_metadata.json')
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n[SAVE] Saved perfectly tuned model & metadata.")
print(f"  Training Time & Scaling Complete!")
print(f"{'='*65}")
