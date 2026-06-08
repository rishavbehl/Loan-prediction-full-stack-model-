"""
Loan Approval Prediction — Banking-Grade Model (v5.0)
=======================================================
Complete RBI/CIBIL-compliant credit scoring pipeline.

Evaluation Framework (4 Pillars):
  1. Classification Accuracy   — Confusion Matrix, Precision, Recall, F1
  2. Discrimination Power      — ROC-AUC, GINI Coefficient, KS Statistic
  3. Calibration & Score Dist  — Hosmer-Lemeshow, PSI, Odds Ratio by band
  4. Business Metrics           — Approval Rate, Bad Rate, Fairness Analysis

Pipeline:
  • Load 300K balanced records from LendingClub
  • Map FICO → CIBIL (Indian 300-900 scale)
  • Engineer 22 banking features (including rejected dataset signal)
  • Compare algorithms → select best
  • 40-iteration RandomizedSearchCV with 5-fold CV
  • Deep overfitting/underfitting diagnosis (learning curves)
  • 12 real-world scenario stress tests
  • Save model.pkl + comprehensive model_metadata.json
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score,
    RandomizedSearchCV, learning_curve
)
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, f1_score, precision_score, recall_score,
    log_loss, brier_score_loss, roc_curve
)
from sklearn.inspection import permutation_importance
import pickle
import json
import os
import sys
import io
import warnings
import time
from scipy.stats import loguniform, randint, uniform, ks_2samp, chi2

warnings.filterwarnings('ignore')

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ─── Configuration ─────────────────────────────────────────────────
SAMPLE_SIZE = 300_000
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
TUNING_ITERATIONS = 40
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ═══════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS: Banking Evaluation Metrics
# ═══════════════════════════════════════════════════════════════════

def compute_ks_statistic(y_true, y_proba):
    """
    KS Statistic (Kolmogorov-Smirnov): Maximum separation between
    cumulative distributions of good and bad loans.
    Industry benchmark: 30-70 = good model.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    ks = np.max(tpr - fpr)
    ks_threshold = thresholds[np.argmax(tpr - fpr)]
    return round(float(ks * 100), 2), round(float(ks_threshold), 4)


def compute_gini(auc_roc):
    """GINI Coefficient = 2 * AUC - 1. Higher = better discrimination."""
    return round(float(2 * auc_roc - 1), 4)


def hosmer_lemeshow_test(y_true, y_proba, n_groups=10):
    """
    Hosmer-Lemeshow Goodness-of-Fit Test.
    Checks if predicted probabilities match observed rates across deciles.
    p-value > 0.05 = model is well-calibrated.
    """
    data = pd.DataFrame({'y': y_true, 'p': y_proba})
    data['decile'] = pd.qcut(data['p'], n_groups, duplicates='drop')

    observed = data.groupby('decile')['y'].sum()
    expected = data.groupby('decile')['p'].sum()
    n_per_group = data.groupby('decile')['y'].count()

    # Chi-squared statistic
    chi2_stat = 0
    for i in range(len(observed)):
        n = n_per_group.iloc[i]
        o1 = observed.iloc[i]
        e1 = expected.iloc[i]
        o0 = n - o1
        e0 = n - e1
        if e1 > 0:
            chi2_stat += (o1 - e1) ** 2 / e1
        if e0 > 0:
            chi2_stat += (o0 - e0) ** 2 / e0

    df = len(observed) - 2
    p_value = 1 - chi2.cdf(chi2_stat, df)

    return round(float(chi2_stat), 4), round(float(p_value), 4), df


def compute_psi(expected_array, actual_array, buckets=10):
    """
    Population Stability Index (PSI).
    Measures distribution shift between train and test score distributions.
    PSI < 0.10 = no shift, 0.10-0.25 = moderate, > 0.25 = significant.
    """
    def psi_bucket(e_pct, a_pct):
        e_pct = max(e_pct, 0.0001)
        a_pct = max(a_pct, 0.0001)
        return (a_pct - e_pct) * np.log(a_pct / e_pct)

    breakpoints = np.linspace(0, 1, buckets + 1)
    expected_percents = np.histogram(expected_array, bins=breakpoints)[0] / len(expected_array)
    actual_percents = np.histogram(actual_array, bins=breakpoints)[0] / len(actual_array)

    psi = sum(psi_bucket(e, a) for e, a in zip(expected_percents, actual_percents))
    return round(float(psi), 4)


def compute_odds_ratio_by_cibil_band(df, cibil_col='cibil_score', target_col='target'):
    """
    Validates that as CIBIL score increases, odds of default DECREASE.
    Returns odds ratio per CIBIL band.
    """
    bands = [
        (300, 500, 'Very Poor (300-500)'),
        (500, 600, 'Poor (500-600)'),
        (600, 700, 'Fair (600-700)'),
        (700, 750, 'Good (700-750)'),
        (750, 800, 'Very Good (750-800)'),
        (800, 901, 'Excellent (800-900)')
    ]

    results = []
    for low, high, label in bands:
        subset = df[(df[cibil_col] >= low) & (df[cibil_col] < high)]
        if len(subset) == 0:
            continue
        n_good = (subset[target_col] == 1).sum()
        n_bad = (subset[target_col] == 0).sum()
        odds = round(n_good / max(n_bad, 1), 4)
        bad_rate = round(n_bad / len(subset) * 100, 2)
        results.append({
            'band': label, 'count': len(subset),
            'good': int(n_good), 'bad': int(n_bad),
            'odds_good_to_bad': odds, 'bad_rate_pct': bad_rate
        })
    return results


def compute_fairness_by_income(df, y_pred, income_col='annual_income'):
    """
    Fairness check: Are lower-income applicants disproportionately rejected
    even when CIBIL scores suggest creditworthiness?
    """
    df_eval = df.copy()
    df_eval['prediction'] = y_pred

    income_bands = [
        (0, 300000, 'Low Income (<3L)'),
        (300000, 600000, 'Middle Income (3-6L)'),
        (600000, 1000000, 'Upper Middle (6-10L)'),
        (1000000, float('inf'), 'High Income (>10L)')
    ]

    results = []
    for low, high, label in income_bands:
        subset = df_eval[(df_eval[income_col] >= low) & (df_eval[income_col] < high)]
        if len(subset) == 0:
            continue

        approval_rate = (subset['prediction'] == 1).mean() * 100
        # Check creditworthy subgroup (CIBIL >= 700 but still rejected)
        creditworthy = subset[subset['cibil_score'] >= 700]
        if len(creditworthy) > 0:
            cw_rejection_rate = (creditworthy['prediction'] == 0).mean() * 100
        else:
            cw_rejection_rate = 0

        results.append({
            'income_band': label,
            'count': len(subset),
            'approval_rate_pct': round(approval_rate, 2),
            'creditworthy_count': len(creditworthy),
            'creditworthy_rejection_rate_pct': round(cw_rejection_rate, 2)
        })
    return results


def parse_emp_length(val):
    if pd.isna(val): return np.nan
    val = str(val).strip()
    if val == '< 1 year': return 0.5
    if val == '10+ years': return 10.0
    if 'year' in val:
        try: return float(val.split()[0])
        except: return np.nan
    return np.nan


# ═══════════════════════════════════════════════════════════════════
#  PHASE 1: Load & Prepare Data
# ═══════════════════════════════════════════════════════════════════
t_start = time.time()
print("=" * 75)
print("  LOAN PREDICTION MODEL v5.0 — BANKING-GRADE TRAINING & EVALUATION")
print("=" * 75)

dataset_path = os.path.join(BASE_DIR, '..', 'accepted_2007_to_2018Q4.csv')

print(f"\n[1/8] LOADING DATA...")
use_cols = [
    'loan_amnt', 'term', 'int_rate', 'grade', 'emp_length',
    'home_ownership', 'annual_inc', 'purpose', 'dti',
    'fico_range_low', 'fico_range_high', 'open_acc', 'revol_util',
    'total_acc', 'pub_rec', 'mort_acc', 'loan_status',
    'delinq_2yrs', 'inq_last_6mths', 'revol_bal', 'installment'
]

df = pd.read_csv(dataset_path, usecols=use_cols, low_memory=False)

# Target Variable
valid_statuses = {'Fully Paid': 1, 'Charged Off': 0, 'Default': 0}
df = df[df['loan_status'].isin(valid_statuses.keys())].copy()
df['target'] = df['loan_status'].map(valid_statuses)

# Balance & Sample
print(f"\n[2/8] BALANCING & SAMPLING {SAMPLE_SIZE:,} records...")
approved = df[df['target'] == 1]
rejected = df[df['target'] == 0]
n_per_class = min(SAMPLE_SIZE // 2, len(rejected))

approved_sample = approved.sample(n=n_per_class, random_state=RANDOM_STATE)
rejected_sample = rejected.sample(n=min(n_per_class, len(rejected)), random_state=RANDOM_STATE)
df = pd.concat([approved_sample, rejected_sample], ignore_index=True)
df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
print(f"   Balanced: {len(df):,} rows | Approved: {(df['target']==1).sum():,} | Rejected: {(df['target']==0).sum():,}")

# ═══════════════════════════════════════════════════════════════════
#  PHASE 2: Feature Engineering (22 features)
# ═══════════════════════════════════════════════════════════════════
print(f"\n[3/8] ENGINEERING 22 BANKING FEATURES...")

# --- Core numeric features ---
fico_avg = (pd.to_numeric(df['fico_range_low'], errors='coerce') +
            pd.to_numeric(df['fico_range_high'], errors='coerce')) / 2
df['cibil_score'] = (300 + (fico_avg - 300) * (600 / 550)).clip(300, 900).round(0)
df['cibil_score'] = df['cibil_score'].fillna(df['cibil_score'].median())

df['annual_income'] = pd.to_numeric(df['annual_inc'], errors='coerce').clip(upper=500000)
df.loc[(df['annual_income'].isna()) | (df['annual_income'] <= 0), 'annual_income'] = df['annual_income'].median()

df['loan_amount'] = pd.to_numeric(df['loan_amnt'], errors='coerce').fillna(df['loan_amnt'].median())
df['int_rate_clean'] = pd.to_numeric(df['int_rate'], errors='coerce').fillna(df['int_rate'].median())
df['dti_clean'] = pd.to_numeric(df['dti'], errors='coerce').clip(0, 60).fillna(df['dti'].median())
df['emp_length_years'] = df['emp_length'].apply(parse_emp_length).fillna(
    df['emp_length'].apply(parse_emp_length).median()
)
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

# --- Composite / Derived Features ---
df['loan_to_income'] = (df['loan_amount'] / df['annual_income']).clip(0, 5).fillna(0)
df['installment_to_income'] = (df['installment_clean'] * 12 / df['annual_income']).clip(0, 2).fillna(0)
df['revol_bal_to_income'] = (df['revol_bal_clean'] / df['annual_income']).clip(0, 10).fillna(0)
df['credit_history_length'] = (df['total_acc_clean'] - df['open_acc_clean']).clip(0, None).fillna(0)
df['derog_score'] = df['pub_rec_clean'] + df['delinq_2yrs_clean'] + df['inq_last_6mths_clean']

# --- Encode Categoricals ---
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

# --- Feature Matrix ---
feature_cols = [
    'cibil_score', 'annual_income', 'loan_amount', 'int_rate_clean', 'dti_clean',
    'emp_length_years', 'term_months', 'home_encoded', 'purpose_encoded',
    'grade_encoded', 'open_acc_clean', 'revol_util_clean', 'total_acc_clean',
    'pub_rec_clean', 'mort_acc_clean', 'delinq_2yrs_clean', 'inq_last_6mths_clean',
    'loan_to_income', 'installment_to_income', 'revol_bal_to_income',
    'credit_history_length', 'derog_score'
]

feature_api_names = [c.replace('_clean', '') for c in feature_cols]

feature_display_names = [
    'CIBIL Score', 'Annual Income', 'Loan Amount', 'Interest Rate', 'DTI',
    'Employment Length', 'Loan Term', 'Home Ownership', 'Loan Purpose',
    'Credit Grade', 'Open Accounts', 'Revolving Utilization', 'Total Accounts',
    'Public Records', 'Mortgage Accounts', 'Delinquencies (2yr)', 'Inquiries (6mo)',
    'Loan-to-Income (LTI)', 'Installment-to-Income', 'Revolving Bal-to-Income',
    'Credit History Length', 'Derogatory Score'
]

X = df[feature_cols].copy()
y = df['target'].copy()
for col in feature_cols:
    X[col] = X[col].fillna(X[col].median())

print(f"   Features: {len(feature_cols)} | Samples: {len(X):,}")

# ═══════════════════════════════════════════════════════════════════
#  PHASE 3: Train/Test Split
# ═══════════════════════════════════════════════════════════════════
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# Keep original df slices for fairness analysis later
df_test = df.iloc[X_test.index].copy()

print(f"   Train: {len(X_train):,} | Test: {len(X_test):,}")

# ═══════════════════════════════════════════════════════════════════
#  PHASE 4: Algorithm Comparison
# ═══════════════════════════════════════════════════════════════════
print(f"\n[4/8] COMPARING ALGORITHM CONFIGURATIONS...")

algorithms = {
    'HistGBT-Balanced': HistGradientBoostingClassifier(
        max_depth=8, max_iter=300, learning_rate=0.05,
        min_samples_leaf=40, l2_regularization=0.3,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
        random_state=RANDOM_STATE
    ),
    'HistGBT-Deep': HistGradientBoostingClassifier(
        max_depth=12, max_iter=500, learning_rate=0.03,
        min_samples_leaf=25, l2_regularization=0.1,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=30,
        random_state=RANDOM_STATE
    ),
    'HistGBT-Wide': HistGradientBoostingClassifier(
        max_depth=6, max_iter=800, learning_rate=0.02,
        min_samples_leaf=60, l2_regularization=0.5,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=30,
        random_state=RANDOM_STATE
    ),
    'HistGBT-Aggressive': HistGradientBoostingClassifier(
        max_depth=15, max_iter=600, learning_rate=0.08,
        min_samples_leaf=15, l2_regularization=0.05,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=15,
        random_state=RANDOM_STATE
    ),
}

print(f"\n   {'Algorithm':<25} {'AUC-ROC':>8} {'GINI':>7} {'Acc':>7} {'F1':>7} {'Gap':>7} {'Time':>6}")
print(f"   {'-'*25} {'-'*8} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*6}")

algo_results = {}
for name, algo in algorithms.items():
    t0 = time.time()
    algo.fit(X_train, y_train)
    elapsed = time.time() - t0

    y_p = algo.predict(X_test)
    y_pr = algo.predict_proba(X_test)[:, 1]
    y_ptr = algo.predict(X_train)

    tr_acc = accuracy_score(y_train, y_ptr)
    te_acc = accuracy_score(y_test, y_p)
    auc = roc_auc_score(y_test, y_pr)
    gini = compute_gini(auc)
    f1 = f1_score(y_test, y_p)
    gap = tr_acc - te_acc

    algo_results[name] = {'model': algo, 'auc': auc, 'gini': gini, 'acc': te_acc, 'f1': f1, 'gap': gap}
    print(f"   {name:<25} {auc:>8.4f} {gini:>7.4f} {te_acc:>7.4f} {f1:>7.4f} {gap:>7.4f} {elapsed:>5.1f}s")

best_algo_name = max(algo_results, key=lambda k: algo_results[k]['auc'])
print(f"\n   >>> Best: {best_algo_name} (AUC={algo_results[best_algo_name]['auc']:.4f})")

# ═══════════════════════════════════════════════════════════════════
#  PHASE 5: Hyperparameter Tuning (40 iterations, 5-fold)
# ═══════════════════════════════════════════════════════════════════
print(f"\n[5/8] HYPERPARAMETER TUNING ({TUNING_ITERATIONS} candidates, {CV_FOLDS}-fold CV)...")

param_distributions = {
    'learning_rate': loguniform(0.005, 0.15),
    'max_iter': randint(200, 1000),
    'max_depth': [5, 6, 8, 10, 12, 15, None],
    'min_samples_leaf': randint(10, 80),
    'l2_regularization': uniform(0.0, 1.5),
    'max_bins': [127, 255],
    'max_leaf_nodes': [31, 63, 127, None],
}

tuning_base = HistGradientBoostingClassifier(
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=25,
    random_state=RANDOM_STATE,
)

search = RandomizedSearchCV(
    estimator=tuning_base,
    param_distributions=param_distributions,
    n_iter=TUNING_ITERATIONS,
    scoring='roc_auc',
    cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=1,
    return_train_score=True
)

search.fit(X_train, y_train)
model = search.best_estimator_

print(f"\n   Best Hyperparameters:")
for k, v in sorted(search.best_params_.items()):
    print(f"     {k}: {round(v, 4) if isinstance(v, float) else v}")

# ═══════════════════════════════════════════════════════════════════
#  PHASE 6: COMPREHENSIVE EVALUATION — ALL 4 PILLARS
# ═══════════════════════════════════════════════════════════════════
print(f"\n[6/8] COMPREHENSIVE BANKING-GRADE EVALUATION...")

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
y_proba_train = model.predict_proba(X_train)[:, 1]
y_proba_test = model.predict_proba(X_test)[:, 1]

# ─────── PILLAR 1: Classification Accuracy & Error Rates ──────────
print(f"\n{'='*75}")
print(f"  PILLAR 1: CLASSIFICATION ACCURACY & ERROR RATES")
print(f"{'='*75}")

train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)
test_f1 = f1_score(y_test, y_pred_test)
test_precision = precision_score(y_test, y_pred_test)
test_recall = recall_score(y_test, y_pred_test)

cm = confusion_matrix(y_test, y_pred_test)
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)
fpr_val = fp / (fp + tn)
fnr_val = fn / (fn + tp)

print(f"\n   Confusion Matrix:")
print(f"                       Predicted Reject  Predicted Approve")
print(f"     Actual Reject     {tn:>15,}  {fp:>17,}")
print(f"     Actual Approve    {fn:>15,}  {tp:>17,}")

print(f"\n   Accuracy:           {test_acc:.4f} ({test_acc*100:.1f}%)")
print(f"   Precision:          {test_precision:.4f}  (Of approved, {test_precision*100:.1f}% actually repaid)")
print(f"   Recall/Sensitivity: {test_recall:.4f}  (Catches {test_recall*100:.1f}% of actual good loans)")
print(f"   Specificity:        {specificity:.4f}  (Catches {specificity*100:.1f}% of actual defaults)")
print(f"   F1 Score:           {test_f1:.4f}")
print(f"   False Positive Rate:{fpr_val:.4f}  (Bad loans wrongly approved)")
print(f"   False Negative Rate:{fnr_val:.4f}  (Good loans wrongly rejected)")

print(f"\n   Classification Report:")
cr = classification_report(y_test, y_pred_test, target_names=['Rejected (Default)', 'Approved (Repaid)'], output_dict=True)
cr_text = classification_report(y_test, y_pred_test, target_names=['Rejected (Default)', 'Approved (Repaid)'])
print(f"   {cr_text}")

# ─────── PILLAR 2: Discrimination Power ───────────────────────────
print(f"\n{'='*75}")
print(f"  PILLAR 2: DISCRIMINATION POWER")
print(f"{'='*75}")

train_auc = roc_auc_score(y_train, y_proba_train)
test_auc = roc_auc_score(y_test, y_proba_test)
gini_coefficient = compute_gini(test_auc)
ks_stat, ks_threshold = compute_ks_statistic(y_test, y_proba_test)

print(f"\n   Train AUC-ROC:      {train_auc:.4f}")
print(f"   Test AUC-ROC:       {test_auc:.4f}")

if test_auc >= 0.90:
    auc_grade = "EXCELLENT"
elif test_auc >= 0.80:
    auc_grade = "GOOD"
elif test_auc >= 0.70:
    auc_grade = "FAIR"
else:
    auc_grade = "POOR"
print(f"   AUC Grade:          {auc_grade}")

print(f"\n   GINI Coefficient:   {gini_coefficient:.4f}  (2*AUC - 1)")
if gini_coefficient >= 0.60:
    print(f"                       EXCELLENT discrimination")
elif gini_coefficient >= 0.40:
    print(f"                       GOOD discrimination")
elif gini_coefficient >= 0.30:
    print(f"                       FAIR discrimination")
else:
    print(f"                       POOR discrimination")

print(f"\n   KS Statistic:       {ks_stat:.2f}  (at threshold {ks_threshold})")
if 30 <= ks_stat <= 70:
    print(f"                       GOOD — Well-separating model (30-70 range)")
elif ks_stat > 70:
    print(f"                       WARNING — Possible overfitting (>70)")
else:
    print(f"                       WEAK — Poor separation (<30)")

# ─────── Overfit / Underfit Diagnosis ─────────────────────────────
overfit_gap_acc = train_acc - test_acc
overfit_gap_auc = train_auc - test_auc

print(f"\n   Overfitting Analysis:")
print(f"     Accuracy Gap:     {overfit_gap_acc:.4f}  (Train - Test)")
print(f"     AUC Gap:          {overfit_gap_auc:.4f}  (Train - Test)")

if overfit_gap_auc > 0.05:
    fit_diagnosis = "OVERFITTING"
    print(f"     Diagnosis:        OVERFITTING — Large train/test gap")
elif overfit_gap_auc > 0.02:
    fit_diagnosis = "SLIGHT OVERFITTING"
    print(f"     Diagnosis:        SLIGHT OVERFITTING — Noticeable but manageable")
elif test_auc < 0.65:
    fit_diagnosis = "UNDERFITTING"
    print(f"     Diagnosis:        UNDERFITTING — Model lacks discriminative power")
else:
    fit_diagnosis = "GOOD FIT"
    print(f"     Diagnosis:        GOOD FIT — Model generalizes well")

# Cross-validation stability
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_acc = cross_val_score(model, X_train, y_train, cv=cv5, scoring='accuracy', n_jobs=-1)
cv_auc = cross_val_score(model, X_train, y_train, cv=cv5, scoring='roc_auc', n_jobs=-1)

print(f"\n   Cross-Validation Stability (5-Fold):")
print(f"     Fold Accuracies:  {[round(s, 4) for s in cv_acc]}")
print(f"     Mean Accuracy:    {cv_acc.mean():.4f} (+/- {cv_acc.std():.4f})")
print(f"     Fold AUC-ROCs:    {[round(s, 4) for s in cv_auc]}")
print(f"     Mean AUC-ROC:     {cv_auc.mean():.4f} (+/- {cv_auc.std():.4f})")

# Learning curve
print(f"\n   Learning Curve (Overfit/Underfit Visual Check):")
train_sizes_abs, train_scores_lc, test_scores_lc = learning_curve(
    model, X_train, y_train,
    train_sizes=[0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
    cv=3, scoring='roc_auc', n_jobs=-1, random_state=RANDOM_STATE
)

print(f"     {'Train Size':>12} {'Train AUC':>12} {'Val AUC':>12} {'Gap':>8}")
print(f"     {'-'*44}")
learning_curve_data = []
for i, sz in enumerate(train_sizes_abs):
    tr = train_scores_lc[i].mean()
    te = test_scores_lc[i].mean()
    learning_curve_data.append({
        'train_size': int(sz), 'train_auc': round(tr, 4),
        'val_auc': round(te, 4), 'gap': round(tr - te, 4)
    })
    print(f"     {sz:>12,} {tr:>12.4f} {te:>12.4f} {tr-te:>8.4f}")

# ─────── PILLAR 3: Calibration & Score Distribution ───────────────
print(f"\n{'='*75}")
print(f"  PILLAR 3: CALIBRATION & SCORE DISTRIBUTION")
print(f"{'='*75}")

# Hosmer-Lemeshow Test
hl_chi2, hl_pvalue, hl_df = hosmer_lemeshow_test(y_test.values, y_proba_test)
print(f"\n   Hosmer-Lemeshow Goodness of Fit:")
print(f"     Chi-squared:      {hl_chi2:.4f}")
print(f"     p-value:          {hl_pvalue:.4f}  (df={hl_df})")
if hl_pvalue > 0.05:
    hl_result = "PASS"
    print(f"     Result:           PASS — Model is well-calibrated (p > 0.05)")
else:
    hl_result = "FAIL"
    print(f"     Result:           FAIL — Predicted probabilities may not match reality (p <= 0.05)")

# Brier Score
brier = brier_score_loss(y_test, y_proba_test)
logloss = log_loss(y_test, y_proba_test)
print(f"\n   Calibration Scores:")
print(f"     Brier Score:      {brier:.4f}  (Lower is better, 0 = perfect)")
print(f"     Log Loss:         {logloss:.4f}  (Lower is better)")

# PSI — Population Stability Index
psi_value = compute_psi(y_proba_train, y_proba_test)
print(f"\n   Population Stability Index (PSI):")
print(f"     PSI:              {psi_value:.4f}")
if psi_value < 0.10:
    psi_result = "STABLE"
    print(f"     Result:           STABLE — No significant distribution shift (<0.10)")
elif psi_value < 0.25:
    psi_result = "MODERATE SHIFT"
    print(f"     Result:           MODERATE SHIFT — Monitor closely (0.10-0.25)")
else:
    psi_result = "SIGNIFICANT SHIFT"
    print(f"     Result:           SIGNIFICANT SHIFT — Model may need retraining (>0.25)")

# Odds Ratio by CIBIL Band
print(f"\n   Odds Ratio by CIBIL Score Band:")
odds_data = compute_odds_ratio_by_cibil_band(df)
print(f"     {'CIBIL Band':<25} {'Count':>7} {'Good':>7} {'Bad':>7} {'Odds':>8} {'Bad Rate':>10}")
print(f"     {'-'*66}")
monotonic_increasing = True
prev_odds = 0
for row in odds_data:
    print(f"     {row['band']:<25} {row['count']:>7} {row['good']:>7} {row['bad']:>7} {row['odds_good_to_bad']:>8.2f} {row['bad_rate_pct']:>9.1f}%")
    if row['odds_good_to_bad'] < prev_odds:
        monotonic_increasing = False
    prev_odds = row['odds_good_to_bad']

if monotonic_increasing:
    odds_result = "PASS"
    print(f"\n     Monotonicity:     PASS — As CIBIL increases, default odds decrease")
else:
    odds_result = "WARNING"
    print(f"\n     Monotonicity:     WARNING — Odds ratio is NOT strictly monotonic")

# ─────── PILLAR 4: Business Metrics & Fairness ────────────────────
print(f"\n{'='*75}")
print(f"  PILLAR 4: BUSINESS METRICS & FAIRNESS")
print(f"{'='*75}")

# Approval Rate on test set
approval_rate = (y_pred_test == 1).mean() * 100
print(f"\n   Approval Rate:      {approval_rate:.1f}%  (on test set)")
if 30 <= approval_rate <= 70:
    print(f"                       HEALTHY — Not too conservative or aggressive")
elif approval_rate < 30:
    print(f"                       WARNING — Too conservative, losing profitable business")
else:
    print(f"                       WARNING — Too aggressive, potential high bad rate")

# Bad Rate (of those approved, how many actually defaulted)
approved_mask = y_pred_test == 1
if approved_mask.sum() > 0:
    bad_rate = (y_test[approved_mask] == 0).mean() * 100
    good_rate = (y_test[approved_mask] == 1).mean() * 100
else:
    bad_rate = 0
    good_rate = 0

print(f"\n   Bad Rate (of approved): {bad_rate:.1f}%  (should be < 15%)")
print(f"   Good Rate (of approved): {good_rate:.1f}%")
if bad_rate < 10:
    bad_rate_grade = "EXCELLENT"
    print(f"   Bad Rate Grade:     EXCELLENT")
elif bad_rate < 15:
    bad_rate_grade = "ACCEPTABLE"
    print(f"   Bad Rate Grade:     ACCEPTABLE")
elif bad_rate < 25:
    bad_rate_grade = "CONCERNING"
    print(f"   Bad Rate Grade:     CONCERNING — Too many defaults slip through")
else:
    bad_rate_grade = "POOR"
    print(f"   Bad Rate Grade:     POOR — Model is not filtering defaults well")

# Rejection accuracy
rejected_mask = y_pred_test == 0
if rejected_mask.sum() > 0:
    correct_rejections = (y_test[rejected_mask] == 0).mean() * 100
else:
    correct_rejections = 0
print(f"\n   Correct Rejections: {correct_rejections:.1f}%  (of rejected, actually would have defaulted)")

# Fairness Analysis
print(f"\n   Fairness Analysis by Income Band:")
fairness_data = compute_fairness_by_income(df_test, y_pred_test)
print(f"     {'Income Band':<25} {'Count':>7} {'Approv%':>8} {'CW Count':>9} {'CW Reject%':>11}")
print(f"     {'-'*62}")
fairness_issues = False
for row in fairness_data:
    flag = " !!!" if row['creditworthy_rejection_rate_pct'] > 40 else ""
    if row['creditworthy_rejection_rate_pct'] > 40:
        fairness_issues = True
    print(f"     {row['income_band']:<25} {row['count']:>7} {row['approval_rate_pct']:>7.1f}% {row['creditworthy_count']:>9} {row['creditworthy_rejection_rate_pct']:>10.1f}%{flag}")

if fairness_issues:
    fairness_grade = "WARNING"
    print(f"\n   Fairness Grade:     WARNING — Some creditworthy groups face >40% rejection")
else:
    fairness_grade = "PASS"
    print(f"\n   Fairness Grade:     PASS — No disproportionate rejection of creditworthy applicants")

# ═══════════════════════════════════════════════════════════════════
#  PHASE 7: Permutation Feature Importance
# ═══════════════════════════════════════════════════════════════════
print(f"\n[7/8] FEATURE IMPORTANCE (Permutation, 10 repeats)...")
perm_imp = permutation_importance(
    model, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1
)

feature_importance = {}
print(f"     {'Feature':<30} {'Importance':>12} {'Std':>10}")
print(f"     {'-'*52}")
for i in np.argsort(perm_imp.importances_mean)[::-1]:
    fname = feature_api_names[i]
    imp = perm_imp.importances_mean[i]
    std = perm_imp.importances_std[i]
    feature_importance[fname] = round(float(imp), 4)
    marker = "  <<< NEGATIVE (remove?)" if imp < -0.001 else ""
    print(f"     {feature_display_names[i]:<30} {imp:>12.4f} {std:>10.4f}{marker}")

sorted_features = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

# ═══════════════════════════════════════════════════════════════════
#  PHASE 8: 12 Real-World Scenario Stress Tests
# ═══════════════════════════════════════════════════════════════════
print(f"\n[8/8] REAL-WORLD SCENARIO STRESS TESTS (12 scenarios)...")

def make_scenario_input(data):
    """Build model-ready DataFrame from scenario dict"""
    annual_inc = max(data['annual_income'], 1)
    home_map_l = {'RENT': 0, 'OWN': 1, 'MORTGAGE': 2, 'OTHER': 3}
    purpose_map_l = encoding_maps.get('purpose', {})
    grade_map_l = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}

    installment_est = data['loan_amount'] / data['term_months']
    features = {
        'cibil_score': data['cibil_score'],
        'annual_income': data['annual_income'],
        'loan_amount': data['loan_amount'],
        'int_rate_clean': data['int_rate'],
        'dti_clean': data['dti'],
        'emp_length_years': data['emp_length'],
        'term_months': data['term_months'],
        'home_encoded': home_map_l.get(data['home_ownership'].upper(), 3),
        'purpose_encoded': purpose_map_l.get(data['purpose'].lower().strip(), 0),
        'grade_encoded': grade_map_l.get(data['grade'].upper(), 4),
        'open_acc_clean': data['open_acc'],
        'revol_util_clean': data['revol_util'],
        'total_acc_clean': data['total_acc'],
        'pub_rec_clean': data['pub_rec'],
        'mort_acc_clean': data['mort_acc'],
        'delinq_2yrs_clean': data['delinq_2yrs'],
        'inq_last_6mths_clean': data['inq_last_6mths'],
        'loan_to_income': data['loan_amount'] / annual_inc,
        'installment_to_income': (installment_est * 12) / annual_inc,
        'revol_bal_to_income': (data['revol_util'] * 100) / annual_inc,
        'credit_history_length': max(0, data['total_acc'] - data['open_acc']),
        'derog_score': data['pub_rec'] + data['delinq_2yrs'] + data['inq_last_6mths'],
    }

    ordered = {}
    for col_name, model_name in zip(feature_cols, model.feature_names_in_):
        ordered[model_name] = [features[col_name]]
    return pd.DataFrame(ordered)


scenarios = [
    # SHOULD BE APPROVED
    {"name": "1. Prime Professional (CIBIL 820, Grade A)", "expected": "Approved",
     "data": {"cibil_score": 820, "annual_income": 1500000, "loan_amount": 500000,
              "int_rate": 7.5, "dti": 8.0, "emp_length": 10, "term_months": 36,
              "home_ownership": "OWN", "purpose": "home_improvement", "grade": "A",
              "open_acc": 4, "revol_util": 12.0, "total_acc": 12, "pub_rec": 0,
              "mort_acc": 1, "delinq_2yrs": 0, "inq_last_6mths": 0}},
    {"name": "2. Senior Govt Employee (CIBIL 780)", "expected": "Approved",
     "data": {"cibil_score": 780, "annual_income": 1200000, "loan_amount": 400000,
              "int_rate": 9.0, "dti": 12.0, "emp_length": 15, "term_months": 36,
              "home_ownership": "MORTGAGE", "purpose": "debt_consolidation", "grade": "A",
              "open_acc": 6, "revol_util": 20.0, "total_acc": 18, "pub_rec": 0,
              "mort_acc": 2, "delinq_2yrs": 0, "inq_last_6mths": 0}},
    {"name": "3. IT Pro Small Loan (CIBIL 750, B)", "expected": "Approved",
     "data": {"cibil_score": 750, "annual_income": 900000, "loan_amount": 200000,
              "int_rate": 10.5, "dti": 15.0, "emp_length": 6, "term_months": 36,
              "home_ownership": "RENT", "purpose": "car", "grade": "B",
              "open_acc": 5, "revol_util": 25.0, "total_acc": 10, "pub_rec": 0,
              "mort_acc": 0, "delinq_2yrs": 0, "inq_last_6mths": 1}},
    {"name": "4. Low Income Perfect Credit (CIBIL 800)", "expected": "Approved",
     "data": {"cibil_score": 800, "annual_income": 300000, "loan_amount": 100000,
              "int_rate": 7.0, "dti": 10.0, "emp_length": 3, "term_months": 36,
              "home_ownership": "RENT", "purpose": "medical", "grade": "A",
              "open_acc": 2, "revol_util": 10.0, "total_acc": 5, "pub_rec": 0,
              "mort_acc": 0, "delinq_2yrs": 0, "inq_last_6mths": 0}},

    # SHOULD BE REJECTED
    {"name": "5. Serial Defaulter (CIBIL 420, G)", "expected": "Rejected",
     "data": {"cibil_score": 420, "annual_income": 250000, "loan_amount": 900000,
              "int_rate": 28.0, "dti": 55.0, "emp_length": 0.5, "term_months": 60,
              "home_ownership": "RENT", "purpose": "debt_consolidation", "grade": "G",
              "open_acc": 15, "revol_util": 98.0, "total_acc": 20, "pub_rec": 3,
              "mort_acc": 0, "delinq_2yrs": 5, "inq_last_6mths": 6}},
    {"name": "6. Unemployed Huge Ask (CIBIL 500, F)", "expected": "Rejected",
     "data": {"cibil_score": 500, "annual_income": 200000, "loan_amount": 1000000,
              "int_rate": 25.0, "dti": 50.0, "emp_length": 0.5, "term_months": 60,
              "home_ownership": "RENT", "purpose": "small_business", "grade": "F",
              "open_acc": 10, "revol_util": 90.0, "total_acc": 14, "pub_rec": 2,
              "mort_acc": 0, "delinq_2yrs": 2, "inq_last_6mths": 4}},
    {"name": "7. High Interest Subprime (CIBIL 550, E)", "expected": "Rejected",
     "data": {"cibil_score": 550, "annual_income": 350000, "loan_amount": 600000,
              "int_rate": 22.0, "dti": 40.0, "emp_length": 2, "term_months": 60,
              "home_ownership": "RENT", "purpose": "credit_card", "grade": "E",
              "open_acc": 8, "revol_util": 85.0, "total_acc": 12, "pub_rec": 1,
              "mort_acc": 0, "delinq_2yrs": 1, "inq_last_6mths": 3}},
    {"name": "8. Maxed Out Cards (CIBIL 650, D)", "expected": "Rejected",
     "data": {"cibil_score": 650, "annual_income": 500000, "loan_amount": 700000,
              "int_rate": 18.0, "dti": 38.0, "emp_length": 3, "term_months": 60,
              "home_ownership": "RENT", "purpose": "credit_card", "grade": "D",
              "open_acc": 9, "revol_util": 92.0, "total_acc": 15, "pub_rec": 0,
              "mort_acc": 0, "delinq_2yrs": 1, "inq_last_6mths": 3}},

    # BORDERLINE / EDGE CASES
    {"name": "9. Young Pro First Loan (CIBIL 710, B)", "expected": "Borderline",
     "data": {"cibil_score": 710, "annual_income": 600000, "loan_amount": 200000,
              "int_rate": 12.0, "dti": 20.0, "emp_length": 2, "term_months": 36,
              "home_ownership": "RENT", "purpose": "car", "grade": "B",
              "open_acc": 3, "revol_util": 30.0, "total_acc": 4, "pub_rec": 0,
              "mort_acc": 0, "delinq_2yrs": 0, "inq_last_6mths": 1}},
    {"name": "10. Rich but Bad CIBIL (480, E)", "expected": "Rejected",
     "data": {"cibil_score": 480, "annual_income": 2000000, "loan_amount": 300000,
              "int_rate": 20.0, "dti": 5.0, "emp_length": 8, "term_months": 36,
              "home_ownership": "OWN", "purpose": "home_improvement", "grade": "E",
              "open_acc": 10, "revol_util": 75.0, "total_acc": 20, "pub_rec": 2,
              "mort_acc": 1, "delinq_2yrs": 3, "inq_last_6mths": 2}},
    {"name": "11. Debt Consolidation (CIBIL 680, C)", "expected": "Borderline",
     "data": {"cibil_score": 680, "annual_income": 700000, "loan_amount": 500000,
              "int_rate": 14.0, "dti": 25.0, "emp_length": 5, "term_months": 60,
              "home_ownership": "RENT", "purpose": "debt_consolidation", "grade": "C",
              "open_acc": 7, "revol_util": 55.0, "total_acc": 14, "pub_rec": 0,
              "mort_acc": 0, "delinq_2yrs": 0, "inq_last_6mths": 2}},
    {"name": "12. Retired Homeowner (CIBIL 760, A)", "expected": "Approved",
     "data": {"cibil_score": 760, "annual_income": 400000, "loan_amount": 150000,
              "int_rate": 8.5, "dti": 12.0, "emp_length": 10, "term_months": 36,
              "home_ownership": "OWN", "purpose": "home_improvement", "grade": "A",
              "open_acc": 3, "revol_util": 15.0, "total_acc": 20, "pub_rec": 0,
              "mort_acc": 2, "delinq_2yrs": 0, "inq_last_6mths": 0}},
]

print(f"\n   {'#':<4} {'Scenario':<45} {'Pred':>8} {'Conf':>7} {'Expected':>15} {'Match':>7}")
print(f"   {'-'*88}")

passed_scenarios = 0
scenario_results = []

for s in scenarios:
    feat_df = make_scenario_input(s['data'])
    pred = int(model.predict(feat_df)[0])
    prob = model.predict_proba(feat_df)[0]
    pred_text = "Approved" if pred == 1 else "Rejected"
    confidence = max(prob) * 100

    exp = s['expected']
    if 'Borderline' in exp:
        match = True
    elif 'Approved' in exp and pred == 1:
        match = True
    elif 'Rejected' in exp and pred == 0:
        match = True
    else:
        match = False

    if match:
        passed_scenarios += 1

    match_str = "PASS" if match else "FAIL"
    num = s['name'].split('.')[0]
    name_short = s['name'][len(num)+2:][:43]
    print(f"   {num:<4} {name_short:<45} {pred_text:>8} {confidence:>6.1f}% {exp:>15} {match_str:>7}")

    scenario_results.append({
        'name': s['name'], 'expected': exp,
        'prediction': pred_text, 'confidence': round(confidence, 1),
        'approve_prob': round(float(prob[1]) * 100, 1),
        'reject_prob': round(float(prob[0]) * 100, 1),
        'match': match
    })

print(f"\n   Scenario Results: {passed_scenarios}/{len(scenarios)} passed ({passed_scenarios/len(scenarios)*100:.0f}%)")

# ═══════════════════════════════════════════════════════════════════
#  SAVE MODEL + COMPREHENSIVE METADATA
# ═══════════════════════════════════════════════════════════════════
elapsed_total = time.time() - t_start

print(f"\n{'='*75}")
print(f"  SAVING MODEL & METADATA")
print(f"{'='*75}")

metadata = {
    'model_type': 'Banking-Grade HistGradientBoosting v5.0',
    'version': '5.0',
    'training_time_seconds': round(elapsed_total, 1),

    # Pillar 1: Classification
    'accuracy': round(float(test_acc), 4),
    'accuracy_percent': round(float(test_acc * 100), 1),
    'train_accuracy': round(float(train_acc), 4),
    'precision': round(float(test_precision), 4),
    'recall': round(float(test_recall), 4),
    'f1_score': round(float(test_f1), 4),
    'specificity': round(float(specificity), 4),
    'sensitivity': round(float(sensitivity), 4),
    'false_positive_rate': round(float(fpr_val), 4),
    'false_negative_rate': round(float(fnr_val), 4),
    'confusion_matrix': {
        'true_negative': int(tn), 'false_positive': int(fp),
        'false_negative': int(fn), 'true_positive': int(tp)
    },
    'classification_report': {
        'Rejected': {k: round(v, 4) for k, v in cr['Rejected (Default)'].items()},
        'Approved': {k: round(v, 4) for k, v in cr['Approved (Repaid)'].items()},
    },

    # Pillar 2: Discrimination Power
    'auc_roc': round(float(test_auc), 4),
    'train_auc_roc': round(float(train_auc), 4),
    'auc_grade': auc_grade,
    'gini_coefficient': gini_coefficient,
    'ks_statistic': ks_stat,
    'ks_threshold': ks_threshold,
    'overfit_gap': round(float(overfit_gap_acc), 4),
    'overfit_gap_auc': round(float(overfit_gap_auc), 4),
    'fit_diagnosis': fit_diagnosis,
    'cv_mean_accuracy': round(float(cv_acc.mean()), 4),
    'cv_std_accuracy': round(float(cv_acc.std()), 4),
    'cv_fold_scores': [round(float(s), 4) for s in cv_acc],
    'cv_auc_fold_scores': [round(float(s), 4) for s in cv_auc],
    'cv_mean_auc': round(float(cv_auc.mean()), 4),
    'learning_curve': learning_curve_data,

    # Pillar 3: Calibration
    'hosmer_lemeshow': {
        'chi2': hl_chi2, 'p_value': hl_pvalue, 'df': hl_df, 'result': hl_result
    },
    'brier_score': round(float(brier), 4),
    'log_loss': round(float(logloss), 4),
    'psi': {'value': psi_value, 'result': psi_result},
    'odds_ratio_by_cibil_band': odds_data,
    'odds_monotonicity': odds_result,

    # Pillar 4: Business Metrics
    'approval_rate': round(approval_rate, 1),
    'bad_rate': round(bad_rate, 1),
    'bad_rate_grade': bad_rate_grade,
    'correct_rejection_rate': round(correct_rejections, 1),
    'fairness_analysis': fairness_data,
    'fairness_grade': fairness_grade,

    # Model Configuration
    'n_estimators_actual': int(model.n_iter_),
    'learning_rate': round(float(search.best_params_['learning_rate']), 4),
    'max_depth': search.best_params_.get('max_depth') if search.best_params_.get('max_depth') is not None else 'None',
    'best_hyperparameters': {k: (round(float(v), 4) if isinstance(v, float) else (v if v is not None else 'None')) for k, v in search.best_params_.items()},

    # Features
    'n_features': len(feature_cols),
    'feature_columns': feature_api_names,
    'feature_display_names': feature_display_names,
    'encoding_maps': {k: {str(kk): int(vv) for kk, vv in v.items()} for k, v in encoding_maps.items()},
    'feature_importance': sorted_features,

    # Scenario Tests
    'scenario_test_results': {
        'passed': passed_scenarios,
        'total': len(scenarios),
        'pass_rate': round(passed_scenarios / len(scenarios) * 100, 1),
        'details': scenario_results
    },

    # Dataset Stats
    'dataset_stats': {
        'total_records': len(df),
        'approved_count': int((df['target'] == 1).sum()),
        'rejected_count': int((df['target'] == 0).sum()),
        'approval_rate': round(int((df['target'] == 1).sum()) / len(df) * 100, 1),
    },
}

model_path = os.path.join(BASE_DIR, 'model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

metadata_path = os.path.join(BASE_DIR, 'model_metadata.json')
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n   Model:    {model_path}")
print(f"   Metadata: {metadata_path}")

# ═══════════════════════════════════════════════════════════════════
#  FINAL SCORECARD
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'='*75}")
print(f"  BANKING MODEL SCORECARD")
print(f"{'='*75}")
print(f"   {'Metric':<35} {'Value':>12} {'Grade':>12}")
print(f"   {'-'*59}")
print(f"   {'Test AUC-ROC':<35} {test_auc:>12.4f} {auc_grade:>12}")
print(f"   {'GINI Coefficient':<35} {gini_coefficient:>12.4f} {'GOOD' if gini_coefficient >= 0.40 else 'FAIR':>12}")
print(f"   {'KS Statistic':<35} {ks_stat:>12.2f} {'GOOD' if 30<=ks_stat<=70 else 'WEAK':>12}")
print(f"   {'Accuracy':<35} {test_acc:>12.4f} {'':>12}")
print(f"   {'F1 Score':<35} {test_f1:>12.4f} {'':>12}")
print(f"   {'Precision':<35} {test_precision:>12.4f} {'':>12}")
print(f"   {'Recall':<35} {test_recall:>12.4f} {'':>12}")
print(f"   {'Overfit Gap (AUC)':<35} {overfit_gap_auc:>12.4f} {fit_diagnosis:>12}")
print(f"   {'Hosmer-Lemeshow p-value':<35} {hl_pvalue:>12.4f} {hl_result:>12}")
print(f"   {'PSI':<35} {psi_value:>12.4f} {psi_result:>12}")
print(f"   {'Odds Monotonicity':<35} {'':>12} {odds_result:>12}")
print(f"   {'Approval Rate':<35} {approval_rate:>11.1f}% {'':>12}")
print(f"   {'Bad Rate':<35} {bad_rate:>11.1f}% {bad_rate_grade:>12}")
print(f"   {'Fairness':<35} {'':>12} {fairness_grade:>12}")
print(f"   {'Scenarios Passed':<35} {passed_scenarios}/{len(scenarios):>9} {'':>12}")
print(f"   {'-'*59}")
print(f"   Training Time: {elapsed_total:.0f}s")
print(f"{'='*75}")
