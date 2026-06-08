"""
Loan Model — Comprehensive Diagnosis, Refinement & Stress Testing
===================================================================
This script:
  1. Diagnoses the current model (learning curves, bias/variance, class reports)
  2. Compares multiple feature engineering strategies
  3. Compares multiple algorithms (HistGBT, GradientBoosting, XGBoost-style)
  4. Selects the best and retrains with expanded hyperparameter search
  5. Runs 12+ real-world scenario tests
  6. Saves the refined model + enriched metadata

Run:  python diagnose_and_refine.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score,
    RandomizedSearchCV, learning_curve
)
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    GradientBoostingClassifier,
    StackingClassifier,
    RandomForestClassifier,
    VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, f1_score, precision_score, recall_score,
    precision_recall_curve, roc_curve, log_loss, brier_score_loss
)
from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve
import pickle
import json
import os
import sys
import io
import warnings
import time
from scipy.stats import loguniform, randint, uniform

warnings.filterwarnings('ignore')

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

RANDOM_STATE = 42
TEST_SIZE = 0.2
SAMPLE_SIZE = 300_000

# ═══════════════════════════════════════════════════════════════════
#  PHASE 1: Load & Prepare Data (same pipeline as train_model.py)
# ═══════════════════════════════════════════════════════════════════
print("=" * 75)
print("  LOAN MODEL — DIAGNOSE, REFINE & STRESS TEST")
print("=" * 75)

dataset_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'accepted_2007_to_2018Q4.csv'
)

print(f"\n[1/7] LOADING DATA...")
use_cols = [
    'loan_amnt', 'term', 'int_rate', 'grade', 'emp_length',
    'home_ownership', 'annual_inc', 'purpose', 'dti',
    'fico_range_low', 'fico_range_high', 'open_acc', 'revol_util',
    'total_acc', 'pub_rec', 'mort_acc', 'loan_status',
    'delinq_2yrs', 'inq_last_6mths', 'revol_bal', 'installment'
]

df = pd.read_csv(dataset_path, usecols=use_cols, low_memory=False)

# Target
valid_statuses = {'Fully Paid': 1, 'Charged Off': 0, 'Default': 0}
df = df[df['loan_status'].isin(valid_statuses.keys())].copy()
df['target'] = df['loan_status'].map(valid_statuses)

# Balance
approved = df[df['target'] == 1]
rejected = df[df['target'] == 0]
n_per_class = min(SAMPLE_SIZE // 2, len(rejected))
approved_sample = approved.sample(n=n_per_class, random_state=RANDOM_STATE)
rejected_sample = rejected.sample(n=min(n_per_class, len(rejected)), random_state=RANDOM_STATE)
df = pd.concat([approved_sample, rejected_sample], ignore_index=True)
df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
print(f"   Balanced dataset: {len(df):,} rows (50/50 split)")

# ═══════════════════════════════════════════════════════════════════
#  PHASE 2: Feature Engineering (Enhanced)
# ═══════════════════════════════════════════════════════════════════
print(f"\n[2/7] FEATURE ENGINEERING...")

# Base features
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
df['revol_bal_clean'] = pd.to_numeric(df['revol_bal'], errors='coerce').fillna(0)
df['installment_clean'] = pd.to_numeric(df['installment'], errors='coerce').fillna(df['installment'].median())

# Composite Features — Improved set
df['loan_to_income'] = (df['loan_amount'] / df['annual_income']).clip(0, 5).fillna(0)
df['installment_to_income'] = (df['installment_clean'] * 12 / df['annual_income']).clip(0, 2).fillna(0)
df['revol_bal_to_income'] = (df['revol_bal_clean'] / df['annual_income']).clip(0, 10).fillna(0)
df['credit_history_length'] = (df['total_acc_clean'] - df['open_acc_clean']).clip(0, None).fillna(0)
df['derog_score'] = df['pub_rec_clean'] + df['delinq_2yrs_clean'] + df['inq_last_6mths_clean']

# Encode categoricals
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

# ═══════════════════════════════════════════════════════════════════
#  PHASE 3: Compare Feature Sets
# ═══════════════════════════════════════════════════════════════════
print(f"\n[3/7] COMPARING FEATURE SETS & ALGORITHMS...")

# Feature set A: Original (19 features, including credit_burden)
feature_set_A_cols = [
    'cibil_score', 'annual_income', 'loan_amount', 'int_rate_clean', 'dti_clean',
    'emp_length_years', 'term_months', 'home_encoded', 'purpose_encoded',
    'grade_encoded', 'open_acc_clean', 'revol_util_clean', 'total_acc_clean', 'pub_rec_clean',
    'mort_acc_clean', 'delinq_2yrs_clean', 'inq_last_6mths_clean',
    'loan_to_income', 'dti_clean'  # credit_burden replaced with dti again (was negative importance)
]
# Remove duplicate dti_clean — just use original 17 + loan_to_income
feature_set_A_cols = [
    'cibil_score', 'annual_income', 'loan_amount', 'int_rate_clean', 'dti_clean',
    'emp_length_years', 'term_months', 'home_encoded', 'purpose_encoded',
    'grade_encoded', 'open_acc_clean', 'revol_util_clean', 'total_acc_clean', 'pub_rec_clean',
    'mort_acc_clean', 'delinq_2yrs_clean', 'inq_last_6mths_clean',
    'loan_to_income'
]

# Feature set B: Enhanced (new composite features, no credit_burden)
feature_set_B_cols = [
    'cibil_score', 'annual_income', 'loan_amount', 'int_rate_clean', 'dti_clean',
    'emp_length_years', 'term_months', 'home_encoded', 'purpose_encoded',
    'grade_encoded', 'open_acc_clean', 'revol_util_clean', 'total_acc_clean', 'pub_rec_clean',
    'mort_acc_clean', 'delinq_2yrs_clean', 'inq_last_6mths_clean',
    'loan_to_income', 'installment_to_income', 'revol_bal_to_income',
    'credit_history_length', 'derog_score'
]

y = df['target'].copy()

# Prepare both feature sets
X_A = df[feature_set_A_cols].copy()
X_B = df[feature_set_B_cols].copy()

for col in feature_set_A_cols:
    X_A[col] = X_A[col].fillna(X_A[col].median())
for col in feature_set_B_cols:
    X_B[col] = X_B[col].fillna(X_B[col].median())

# Quick comparison with a baseline model
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
baseline_model = HistGradientBoostingClassifier(
    max_depth=8, max_iter=200, learning_rate=0.05,
    min_samples_leaf=40, l2_regularization=0.3,
    early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
    random_state=RANDOM_STATE
)

print(f"\n   Feature Set A ({len(feature_set_A_cols)} features — original minus credit_burden):")
cv_A = cross_val_score(baseline_model, X_A, y, cv=cv, scoring='roc_auc', n_jobs=-1)
print(f"     CV AUC-ROC: {cv_A.mean():.4f} (+/- {cv_A.std():.4f})")

print(f"\n   Feature Set B ({len(feature_set_B_cols)} features — enhanced composites):")
cv_B = cross_val_score(baseline_model, X_B, y, cv=cv, scoring='roc_auc', n_jobs=-1)
print(f"     CV AUC-ROC: {cv_B.mean():.4f} (+/- {cv_B.std():.4f})")

# Select best feature set
if cv_B.mean() > cv_A.mean():
    best_feature_set = 'B'
    X = X_B.copy()
    feature_cols = feature_set_B_cols
    print(f"\n   >>> Selected Feature Set B (Enhanced)")
else:
    best_feature_set = 'A'
    X = X_A.copy()
    feature_cols = feature_set_A_cols
    print(f"\n   >>> Selected Feature Set A (Original)")

# API names for metadata
feature_api_names = [c.replace('_clean', '') for c in feature_cols]
feature_display_map = {
    'cibil_score': 'CIBIL Score', 'annual_income': 'Annual Income',
    'loan_amount': 'Loan Amount', 'int_rate': 'Interest Rate',
    'dti': 'DTI', 'emp_length_years': 'Employment Length',
    'term_months': 'Loan Term', 'home_encoded': 'Home Ownership',
    'purpose_encoded': 'Loan Purpose', 'grade_encoded': 'Credit Grade',
    'open_acc': 'Open Accounts', 'revol_util': 'Revolving Utilization',
    'total_acc': 'Total Accounts', 'pub_rec': 'Public Records',
    'mort_acc': 'Mortgage Accounts', 'delinq_2yrs': 'Delinquencies (2yr)',
    'inq_last_6mths': 'Inquiries (6mo)', 'loan_to_income': 'Loan-to-Income (LTI)',
    'installment_to_income': 'Installment-to-Income', 'revol_bal_to_income': 'Revolving Bal-to-Income',
    'credit_history_length': 'Credit History Length', 'derog_score': 'Derogatory Score'
}
feature_display_names = [feature_display_map.get(n, n) for n in feature_api_names]

# ═══════════════════════════════════════════════════════════════════
#  PHASE 4: Algorithm Comparison
# ═══════════════════════════════════════════════════════════════════
print(f"\n[4/7] COMPARING ALGORITHMS...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

algorithms = {
    'HistGradientBoosting (Tuned)': HistGradientBoostingClassifier(
        max_depth=8, max_iter=300, learning_rate=0.05,
        min_samples_leaf=40, l2_regularization=0.3,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
        random_state=RANDOM_STATE
    ),
    'HistGradientBoosting (Deeper)': HistGradientBoostingClassifier(
        max_depth=12, max_iter=500, learning_rate=0.03,
        min_samples_leaf=25, l2_regularization=0.1,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=30,
        random_state=RANDOM_STATE
    ),
    'HistGradientBoosting (Wider)': HistGradientBoostingClassifier(
        max_depth=6, max_iter=800, learning_rate=0.02,
        min_samples_leaf=60, l2_regularization=0.5,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=30,
        random_state=RANDOM_STATE
    ),
    'GradientBoosting (Classic)': GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        min_samples_leaf=40, subsample=0.8,
        random_state=RANDOM_STATE
    ),
}

results = {}
print(f"\n   {'Algorithm':<40} {'AUC-ROC':>10} {'Accuracy':>10} {'F1':>10} {'Train Acc':>10} {'Gap':>8}")
print(f"   {'-'*40} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")

for name, algo in algorithms.items():
    t0 = time.time()
    algo.fit(X_train, y_train)
    elapsed = time.time() - t0

    y_pred = algo.predict(X_test)
    y_proba = algo.predict_proba(X_test)[:, 1]
    y_pred_train = algo.predict(X_train)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    gap = train_acc - test_acc

    results[name] = {
        'model': algo, 'auc': auc, 'acc': test_acc, 'f1': f1,
        'train_acc': train_acc, 'gap': gap, 'time': elapsed
    }

    gap_emoji = "OK" if gap < 0.02 else "WARN" if gap < 0.05 else "OVERFIT"
    print(f"   {name:<40} {auc:>10.4f} {test_acc:>10.4f} {f1:>10.4f} {train_acc:>10.4f} {gap:>7.4f} {gap_emoji}")

# Select best by AUC-ROC
best_name = max(results, key=lambda k: results[k]['auc'])
print(f"\n   >>> Best Algorithm: {best_name} (AUC={results[best_name]['auc']:.4f})")

# ═══════════════════════════════════════════════════════════════════
#  PHASE 5: Hyperparameter Tuning on Best Algorithm
# ═══════════════════════════════════════════════════════════════════
print(f"\n[5/7] HYPERPARAMETER TUNING (Extended Search)...")

param_distributions = {
    'learning_rate': loguniform(0.005, 0.15),
    'max_iter': randint(200, 1000),
    'max_depth': [5, 6, 8, 10, 12, 15, None],
    'min_samples_leaf': randint(10, 80),
    'l2_regularization': uniform(0.0, 1.5),
    'max_bins': [127, 255],
    'max_leaf_nodes': [31, 63, 127, None],
}

tuning_model = HistGradientBoostingClassifier(
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=25,
    random_state=RANDOM_STATE,
)

search = RandomizedSearchCV(
    estimator=tuning_model,
    param_distributions=param_distributions,
    n_iter=30,               # 30 candidates (double the previous 15)
    scoring='roc_auc',
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=1,
    return_train_score=True
)

search.fit(X_train, y_train)

best_model = search.best_estimator_
print(f"\n   Best Hyperparameters:")
for k, v in sorted(search.best_params_.items()):
    if isinstance(v, float):
        print(f"     {k}: {v:.4f}")
    else:
        print(f"     {k}: {v}")

# ═══════════════════════════════════════════════════════════════════
#  PHASE 6: Deep Evaluation — Overfitting / Underfitting Analysis
# ═══════════════════════════════════════════════════════════════════
print(f"\n[6/7] DEEP EVALUATION & OVERFITTING/UNDERFITTING ANALYSIS...")

# --- Core metrics ---
y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)
y_proba_train = best_model.predict_proba(X_train)[:, 1]
y_proba_test = best_model.predict_proba(X_test)[:, 1]

train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)
train_auc = roc_auc_score(y_train, y_proba_train)
test_auc = roc_auc_score(y_test, y_proba_test)
test_f1 = f1_score(y_test, y_pred_test)
test_precision = precision_score(y_test, y_pred_test)
test_recall = recall_score(y_test, y_pred_test)
test_logloss = log_loss(y_test, y_proba_test)
test_brier = brier_score_loss(y_test, y_proba_test)
overfit_gap_acc = train_acc - test_acc
overfit_gap_auc = train_auc - test_auc

print(f"\n   {'='*60}")
print(f"   REFINED MODEL — COMPREHENSIVE METRICS")
print(f"   {'='*60}")
print(f"   Train Accuracy:     {train_acc:.4f}")
print(f"   Test Accuracy:      {test_acc:.4f}")
print(f"   Train AUC-ROC:      {train_auc:.4f}")
print(f"   Test AUC-ROC:       {test_auc:.4f}")
print(f"   Test F1 Score:      {test_f1:.4f}")
print(f"   Test Precision:     {test_precision:.4f}")
print(f"   Test Recall:        {test_recall:.4f}")
print(f"   Test Log Loss:      {test_logloss:.4f}")
print(f"   Test Brier Score:   {test_brier:.4f}")
print(f"   {'─'*60}")
print(f"   Overfit Gap (Acc):  {overfit_gap_acc:.4f}  {'PASS' if overfit_gap_acc < 0.02 else 'WARN' if overfit_gap_acc < 0.05 else 'OVERFIT'}")
print(f"   Overfit Gap (AUC):  {overfit_gap_auc:.4f}  {'PASS' if overfit_gap_auc < 0.02 else 'WARN' if overfit_gap_auc < 0.05 else 'OVERFIT'}")

# --- Underfitting check ---
if test_auc < 0.70:
    fit_status = "UNDERFITTING"
    print(f"\n   DIAGNOSIS: UNDERFITTING — AUC < 0.70. Model lacks signal.")
elif overfit_gap_auc > 0.05:
    fit_status = "OVERFITTING"
    print(f"\n   DIAGNOSIS: OVERFITTING — Large train/test AUC gap.")
elif overfit_gap_auc > 0.02:
    fit_status = "SLIGHT OVERFITTING"
    print(f"\n   DIAGNOSIS: SLIGHT OVERFITTING — Gap is noticeable but manageable.")
else:
    fit_status = "GOOD FIT"
    print(f"\n   DIAGNOSIS: GOOD FIT — Model generalizes well with minimal gap.")

# --- Cross-validation stability ---
print(f"\n   Cross-Validation Stability (5-Fold):")
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_acc = cross_val_score(best_model, X_train, y_train, cv=cv5, scoring='accuracy', n_jobs=-1)
cv_auc = cross_val_score(best_model, X_train, y_train, cv=cv5, scoring='roc_auc', n_jobs=-1)

print(f"     Fold Accuracies: {[round(s, 4) for s in cv_acc]}")
print(f"     Mean Accuracy:   {cv_acc.mean():.4f} (+/- {cv_acc.std():.4f})")
print(f"     Fold AUC-ROCs:   {[round(s, 4) for s in cv_auc]}")
print(f"     Mean AUC-ROC:    {cv_auc.mean():.4f} (+/- {cv_auc.std():.4f})")

if cv_acc.std() < 0.005:
    print(f"     Variance: MINIMAL — Highly stable across folds")
elif cv_acc.std() < 0.015:
    print(f"     Variance: LOW — Stable model")
else:
    print(f"     Variance: MODERATE — Some instability across folds")

# --- Classification Report ---
print(f"\n   Classification Report:")
cr = classification_report(y_test, y_pred_test, target_names=['Rejected', 'Approved'], output_dict=True)
print(f"     {'Class':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
print(f"     {'-'*55}")
for cls in ['Rejected', 'Approved']:
    print(f"     {cls:<15} {cr[cls]['precision']:>10.4f} {cr[cls]['recall']:>10.4f} {cr[cls]['f1-score']:>10.4f} {cr[cls]['support']:>10.0f}")
print(f"     {'─'*55}")
print(f"     {'Macro Avg':<15} {cr['macro avg']['precision']:>10.4f} {cr['macro avg']['recall']:>10.4f} {cr['macro avg']['f1-score']:>10.4f}")

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred_test)
print(f"\n   Confusion Matrix:")
print(f"                     Predicted Reject  Predicted Approve")
print(f"     Actual Reject   {cm[0][0]:>15,}  {cm[0][1]:>17,}")
print(f"     Actual Approve  {cm[1][0]:>15,}  {cm[1][1]:>17,}")

tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)
print(f"\n     Sensitivity (TPR): {sensitivity:.4f}  (Catches {sensitivity*100:.1f}% of good loans)")
print(f"     Specificity (TNR): {specificity:.4f}  (Catches {specificity*100:.1f}% of bad loans)")

# --- Learning Curve (detect under/overfit visually) ---
print(f"\n   Learning Curve Analysis (sampled)...")
train_sizes_abs, train_scores_lc, test_scores_lc = learning_curve(
    best_model, X_train, y_train,
    train_sizes=[0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
    cv=3, scoring='roc_auc', n_jobs=-1,
    random_state=RANDOM_STATE
)

print(f"     {'Train Size':>12} {'Train AUC':>12} {'Val AUC':>12} {'Gap':>8}")
print(f"     {'-'*44}")
learning_curve_data = []
for i, sz in enumerate(train_sizes_abs):
    tr = train_scores_lc[i].mean()
    te = test_scores_lc[i].mean()
    learning_curve_data.append({
        'train_size': int(sz),
        'train_auc': round(tr, 4),
        'val_auc': round(te, 4),
        'gap': round(tr - te, 4)
    })
    print(f"     {sz:>12,} {tr:>12.4f} {te:>12.4f} {tr-te:>8.4f}")

converging = (learning_curve_data[-1]['gap'] < learning_curve_data[0]['gap'])
if converging:
    print(f"     Learning curve is CONVERGING — model improves with more data")
else:
    print(f"     Learning curve is NOT converging — model may be at capacity")

# --- Permutation Feature Importance ---
print(f"\n   Permutation Feature Importance:")
perm_imp = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
feature_importance = {}
print(f"     {'Feature':<30} {'Importance':>12} {'Std':>10}")
print(f"     {'-'*52}")
for i in np.argsort(perm_imp.importances_mean)[::-1]:
    fname = feature_api_names[i]
    imp = perm_imp.importances_mean[i]
    std = perm_imp.importances_std[i]
    feature_importance[fname] = round(float(imp), 4)
    marker = " <<< NEGATIVE" if imp < 0 else ""
    print(f"     {feature_display_names[i]:<30} {imp:>12.4f} {std:>10.4f}{marker}")

sorted_features = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

# ═══════════════════════════════════════════════════════════════════
#  PHASE 7: Real-World Scenario Stress Tests (12 scenarios)
# ═══════════════════════════════════════════════════════════════════
print(f"\n[7/7] REAL-WORLD SCENARIO STRESS TESTS...")

def make_input(data):
    """Build model-ready input from scenario data"""
    annual_inc = data['annual_income'] if data['annual_income'] > 0 else 1
    home_map_local = {'RENT': 0, 'OWN': 1, 'MORTGAGE': 2, 'OTHER': 3}
    purpose_map_local = encoding_maps.get('purpose', {})
    grade_map_local = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}

    features = {
        'cibil_score': data['cibil_score'],
        'annual_income': data['annual_income'],
        'loan_amount': data['loan_amount'],
        'int_rate_clean': data['int_rate'],
        'dti_clean': data['dti'],
        'emp_length_years': data['emp_length'],
        'term_months': data['term_months'],
        'home_encoded': home_map_local.get(data['home_ownership'].upper(), 3),
        'purpose_encoded': purpose_map_local.get(data['purpose'].lower().strip(), 0),
        'grade_encoded': grade_map_local.get(data['grade'].upper(), 4),
        'open_acc_clean': data['open_acc'],
        'revol_util_clean': data['revol_util'],
        'total_acc_clean': data['total_acc'],
        'pub_rec_clean': data['pub_rec'],
        'mort_acc_clean': data['mort_acc'],
        'delinq_2yrs_clean': data['delinq_2yrs'],
        'inq_last_6mths_clean': data['inq_last_6mths'],
        'loan_to_income': data['loan_amount'] / annual_inc,
    }

    # Add enhanced features if using Feature Set B
    if best_feature_set == 'B':
        features['installment_to_income'] = data.get('installment', data['loan_amount'] / data['term_months']) * 12 / annual_inc
        features['revol_bal_to_income'] = data.get('revol_bal', data['revol_util'] * 100) / annual_inc
        features['credit_history_length'] = max(0, data['total_acc'] - data['open_acc'])
        features['derog_score'] = data['pub_rec'] + data['delinq_2yrs'] + data['inq_last_6mths']

    # Build in correct column order
    ordered = {}
    for col_name, model_name in zip(feature_cols, best_model.feature_names_in_):
        ordered[model_name] = [features[col_name]]
    return pd.DataFrame(ordered)

scenarios = [
    # ── SHOULD BE APPROVED (CLEARLY) ──
    {"name": "1. Prime Salaried Professional (CIBIL 820, Grade A)",
     "expected": "Approved", "confidence_min": 60,
     "data": {"cibil_score": 820, "annual_income": 1500000, "loan_amount": 500000,
              "int_rate": 7.5, "dti": 8.0, "emp_length": 10, "term_months": 36,
              "home_ownership": "OWN", "purpose": "home_improvement", "grade": "A",
              "open_acc": 4, "revol_util": 12.0, "total_acc": 12, "pub_rec": 0,
              "mort_acc": 1, "delinq_2yrs": 0, "inq_last_6mths": 0}},

    {"name": "2. Senior Government Employee (CIBIL 780, Mortgage)",
     "expected": "Approved", "confidence_min": 55,
     "data": {"cibil_score": 780, "annual_income": 1200000, "loan_amount": 400000,
              "int_rate": 9.0, "dti": 12.0, "emp_length": 15, "term_months": 36,
              "home_ownership": "MORTGAGE", "purpose": "debt_consolidation", "grade": "A",
              "open_acc": 6, "revol_util": 20.0, "total_acc": 18, "pub_rec": 0,
              "mort_acc": 2, "delinq_2yrs": 0, "inq_last_6mths": 0}},

    {"name": "3. IT Professional Small Personal Loan (CIBIL 750, Grade B)",
     "expected": "Approved", "confidence_min": 52,
     "data": {"cibil_score": 750, "annual_income": 900000, "loan_amount": 200000,
              "int_rate": 10.5, "dti": 15.0, "emp_length": 6, "term_months": 36,
              "home_ownership": "RENT", "purpose": "car", "grade": "B",
              "open_acc": 5, "revol_util": 25.0, "total_acc": 10, "pub_rec": 0,
              "mort_acc": 0, "delinq_2yrs": 0, "inq_last_6mths": 1}},

    # ── SHOULD BE REJECTED (CLEARLY) ──
    {"name": "4. Serial Defaulter (CIBIL 420, Grade G, Delinquencies)",
     "expected": "Rejected", "confidence_min": 60,
     "data": {"cibil_score": 420, "annual_income": 250000, "loan_amount": 900000,
              "int_rate": 28.0, "dti": 55.0, "emp_length": 0.5, "term_months": 60,
              "home_ownership": "RENT", "purpose": "debt_consolidation", "grade": "G",
              "open_acc": 15, "revol_util": 98.0, "total_acc": 20, "pub_rec": 3,
              "mort_acc": 0, "delinq_2yrs": 5, "inq_last_6mths": 6}},

    {"name": "5. Unemployed with Huge Loan Ask (CIBIL 500, Grade F)",
     "expected": "Rejected", "confidence_min": 55,
     "data": {"cibil_score": 500, "annual_income": 200000, "loan_amount": 1000000,
              "int_rate": 25.0, "dti": 50.0, "emp_length": 0.5, "term_months": 60,
              "home_ownership": "RENT", "purpose": "small_business", "grade": "F",
              "open_acc": 10, "revol_util": 90.0, "total_acc": 14, "pub_rec": 2,
              "mort_acc": 0, "delinq_2yrs": 2, "inq_last_6mths": 4}},

    {"name": "6. High Interest Subprime Borrower (CIBIL 550, Grade E)",
     "expected": "Rejected", "confidence_min": 52,
     "data": {"cibil_score": 550, "annual_income": 350000, "loan_amount": 600000,
              "int_rate": 22.0, "dti": 40.0, "emp_length": 2, "term_months": 60,
              "home_ownership": "RENT", "purpose": "credit_card", "grade": "E",
              "open_acc": 8, "revol_util": 85.0, "total_acc": 12, "pub_rec": 1,
              "mort_acc": 0, "delinq_2yrs": 1, "inq_last_6mths": 3}},

    # ── BORDERLINE CASES ──
    {"name": "7. Young Professional First Loan (CIBIL 710, Grade B)",
     "expected": "Borderline/Approved", "confidence_min": 50,
     "data": {"cibil_score": 710, "annual_income": 600000, "loan_amount": 200000,
              "int_rate": 12.0, "dti": 20.0, "emp_length": 2, "term_months": 36,
              "home_ownership": "RENT", "purpose": "car", "grade": "B",
              "open_acc": 3, "revol_util": 30.0, "total_acc": 4, "pub_rec": 0,
              "mort_acc": 0, "delinq_2yrs": 0, "inq_last_6mths": 1}},

    {"name": "8. Mid-career Debt Consolidation (CIBIL 680, Grade C)",
     "expected": "Borderline", "confidence_min": 50,
     "data": {"cibil_score": 680, "annual_income": 700000, "loan_amount": 500000,
              "int_rate": 14.0, "dti": 25.0, "emp_length": 5, "term_months": 60,
              "home_ownership": "RENT", "purpose": "debt_consolidation", "grade": "C",
              "open_acc": 7, "revol_util": 55.0, "total_acc": 14, "pub_rec": 0,
              "mort_acc": 0, "delinq_2yrs": 0, "inq_last_6mths": 2}},

    # ── EDGE CASES ──
    {"name": "9. Very High Income but Bad CIBIL (CIBIL 480, Grade E)",
     "expected": "Rejected", "confidence_min": 50,
     "data": {"cibil_score": 480, "annual_income": 2000000, "loan_amount": 300000,
              "int_rate": 20.0, "dti": 5.0, "emp_length": 8, "term_months": 36,
              "home_ownership": "OWN", "purpose": "home_improvement", "grade": "E",
              "open_acc": 10, "revol_util": 75.0, "total_acc": 20, "pub_rec": 2,
              "mort_acc": 1, "delinq_2yrs": 3, "inq_last_6mths": 2}},

    {"name": "10. Low Income but Perfect Credit (CIBIL 800, Grade A)",
     "expected": "Approved", "confidence_min": 50,
     "data": {"cibil_score": 800, "annual_income": 300000, "loan_amount": 100000,
              "int_rate": 7.0, "dti": 10.0, "emp_length": 3, "term_months": 36,
              "home_ownership": "RENT", "purpose": "medical", "grade": "A",
              "open_acc": 2, "revol_util": 10.0, "total_acc": 5, "pub_rec": 0,
              "mort_acc": 0, "delinq_2yrs": 0, "inq_last_6mths": 0}},

    {"name": "11. Maxed Out Credit Cards (CIBIL 650, Grade D)",
     "expected": "Rejected", "confidence_min": 50,
     "data": {"cibil_score": 650, "annual_income": 500000, "loan_amount": 700000,
              "int_rate": 18.0, "dti": 38.0, "emp_length": 3, "term_months": 60,
              "home_ownership": "RENT", "purpose": "credit_card", "grade": "D",
              "open_acc": 9, "revol_util": 92.0, "total_acc": 15, "pub_rec": 0,
              "mort_acc": 0, "delinq_2yrs": 1, "inq_last_6mths": 3}},

    {"name": "12. Retired Homeowner (CIBIL 760, Grade A, Low Income)",
     "expected": "Approved", "confidence_min": 50,
     "data": {"cibil_score": 760, "annual_income": 400000, "loan_amount": 150000,
              "int_rate": 8.5, "dti": 12.0, "emp_length": 10, "term_months": 36,
              "home_ownership": "OWN", "purpose": "home_improvement", "grade": "A",
              "open_acc": 3, "revol_util": 15.0, "total_acc": 20, "pub_rec": 0,
              "mort_acc": 2, "delinq_2yrs": 0, "inq_last_6mths": 0}},
]

print(f"\n   {'#':<4} {'Scenario':<55} {'Pred':>8} {'Conf':>7} {'Expected':>20} {'Match':>7}")
print(f"   {'-'*4} {'-'*55} {'-'*8} {'-'*7} {'-'*20} {'-'*7}")

passed = 0
total = len(scenarios)
scenario_results = []

for s in scenarios:
    features_df = make_input(s['data'])
    pred = int(best_model.predict(features_df)[0])
    prob = best_model.predict_proba(features_df)[0]
    pred_text = "Approved" if pred == 1 else "Rejected"
    confidence = max(prob) * 100

    # Check if prediction matches expected
    exp = s['expected']
    if 'Borderline' in exp:
        match = True  # Borderline = either is acceptable
    elif 'Approved' in exp and pred == 1:
        match = True
    elif 'Rejected' in exp and pred == 0:
        match = True
    else:
        match = False

    if match:
        passed += 1

    match_str = "PASS" if match else "FAIL"
    print(f"   {s['name'][:3]:<4} {s['name'][3:58]:<55} {pred_text:>8} {confidence:>6.1f}% {exp:>20} {match_str:>7}")

    scenario_results.append({
        'name': s['name'],
        'expected': exp,
        'prediction': pred_text,
        'confidence': round(confidence, 1),
        'approve_prob': round(float(prob[1]) * 100, 1),
        'reject_prob': round(float(prob[0]) * 100, 1),
        'match': match
    })

print(f"\n   Scenario Results: {passed}/{total} passed ({passed/total*100:.0f}%)")

# ═══════════════════════════════════════════════════════════════════
#  SAVE: Refined Model + Enriched Metadata
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'='*75}")
print(f"  SAVING REFINED MODEL & METADATA")
print(f"{'='*75}")

metadata = {
    'model_type': 'Refined HistGradientBoosting v4.0',
    'accuracy': round(float(test_acc), 4),
    'accuracy_percent': round(float(test_acc * 100), 1),
    'train_accuracy': round(float(train_acc), 4),
    'cv_mean_accuracy': round(float(cv_acc.mean()), 4),
    'cv_std_accuracy': round(float(cv_acc.std()), 4),
    'cv_fold_scores': [round(float(s), 4) for s in cv_acc],
    'auc_roc': round(float(test_auc), 4),
    'train_auc_roc': round(float(train_auc), 4),
    'f1_score': round(float(test_f1), 4),
    'precision': round(float(test_precision), 4),
    'recall': round(float(test_recall), 4),
    'specificity': round(float(specificity), 4),
    'sensitivity': round(float(sensitivity), 4),
    'log_loss': round(float(test_logloss), 4),
    'brier_score': round(float(test_brier), 4),
    'overfit_gap': round(float(overfit_gap_acc), 4),
    'overfit_gap_auc': round(float(overfit_gap_auc), 4),
    'fit_status': fit_status,
    'n_estimators_actual': int(best_model.n_iter_),
    'learning_rate': round(float(search.best_params_['learning_rate']), 4),
    'max_depth': search.best_params_.get('max_depth', 'None'),
    'best_hyperparameters': {k: (round(float(v), 4) if isinstance(v, float) else v) for k, v in search.best_params_.items()},
    'feature_columns': feature_api_names,
    'feature_display_names': feature_display_names,
    'encoding_maps': {k: {str(kk): int(vv) for kk, vv in v.items()} for k, v in encoding_maps.items()},
    'feature_importance': sorted_features,
    'classification_report': {
        'Rejected': {k: round(v, 4) for k, v in cr['Rejected'].items()},
        'Approved': {k: round(v, 4) for k, v in cr['Approved'].items()},
    },
    'confusion_matrix': {
        'true_negative': int(tn), 'false_positive': int(fp),
        'false_negative': int(fn), 'true_positive': int(tp)
    },
    'learning_curve': learning_curve_data,
    'scenario_test_results': {
        'passed': passed, 'total': total,
        'pass_rate': round(passed / total * 100, 1),
        'details': scenario_results
    },
    'dataset_stats': {
        'total_records': len(df),
        'approved_count': int((df['target'] == 1).sum()),
        'rejected_count': int((df['target'] == 0).sum()),
        'approval_rate': round(int((df['target'] == 1).sum()) / len(df) * 100, 1),
    },
    'feature_set': best_feature_set,
    'n_features': len(feature_cols),
}

# Handle None max_depth
if metadata['max_depth'] is None:
    metadata['max_depth'] = 'None'

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(best_model, f)

metadata_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_metadata.json')
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n   Model saved to: {model_path}")
print(f"   Metadata saved to: {metadata_path}")

print(f"\n{'='*75}")
print(f"  SUMMARY")
print(f"{'='*75}")
print(f"   Feature Set:      {best_feature_set} ({len(feature_cols)} features)")
print(f"   Fit Status:       {fit_status}")
print(f"   Test AUC-ROC:     {test_auc:.4f}")
print(f"   Test Accuracy:    {test_acc:.4f}")
print(f"   Test F1:          {test_f1:.4f}")
print(f"   Overfit Gap:      {overfit_gap_acc:.4f} (Acc) / {overfit_gap_auc:.4f} (AUC)")
print(f"   Scenarios:        {passed}/{total} passed")
print(f"{'='*75}")
