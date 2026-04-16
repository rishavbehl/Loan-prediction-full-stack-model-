"""
Loan Approval Prediction - Model Training Script
================================================
This script:
1. Loads the credit_risk.csv dataset
2. Cleans & preprocesses the data (handles missing values, encodes categoricals)
3. Trains a Random Forest Classifier
4. Saves the trained model (model.pkl) and metadata (model_metadata.json)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import json
import os
import sys
import io

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# --- 1. Load Dataset ---
print("=" * 60)
print("  LOAN APPROVAL PREDICTION - MODEL TRAINING")
print("=" * 60)

# Go up one level to find the CSV in the project root
dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'credit_risk.csv')
df = pd.read_csv(dataset_path)

print(f"\n[DATA] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"   Columns: {list(df.columns)}")

# --- 2. Data Exploration ---
print("\n[INFO] Data Types:")
print(df.dtypes)
print(f"\n[INFO] Missing Values:\n{df.isnull().sum()}")
print(f"\n[INFO] Target Distribution (Status):\n{df['Status'].value_counts()}")

# --- 3. Data Cleaning ---
print("\n[CLEAN] Cleaning data...")

# Drop the Id column (not a feature)
df = df.drop('Id', axis=1)

# Handle missing values
# Fill numeric missing values with median
numeric_cols = ['Age', 'Income', 'Emp_length', 'Amount', 'Rate', 'Percent_income', 'Cred_length']
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"   Filled {col} missing values with median: {median_val}")

# Cap Age at reasonable values (remove outliers like 123, 144)
df.loc[df['Age'] > 100, 'Age'] = df.loc[df['Age'] <= 100, 'Age'].median()
print(f"   Capped Age outliers to median")

# --- 4. Encode Categorical Variables ---
print("\n[ENCODE] Encoding categorical variables...")

categorical_cols = ['Home', 'Intent', 'Default']
label_encoders = {}
encoding_maps = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
    encoding_maps[col] = {label: int(idx) for idx, label in enumerate(le.classes_)}
    print(f"   {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# --- 5. Feature Engineering ---
print("\n[FEATURES] Preparing features...")

# Features (everything except Status which is our target)
feature_cols = ['Age', 'Income', 'Home', 'Emp_length', 'Intent', 'Amount', 'Rate', 'Percent_income', 'Default', 'Cred_length']
X = df[feature_cols]
y = df['Status']

print(f"   Features: {feature_cols}")
print(f"   Target: Status (0 = Rejected, 1 = Approved)")
print(f"   X shape: {X.shape}, y shape: {y.shape}")

# --- 6. Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\n[SPLIT] Train/Test Split:")
print(f"   Training: {X_train.shape[0]} samples")
print(f"   Testing:  {X_test.shape[0]} samples")

# --- 7. Train Model ---
print("\n[TRAIN] Training Random Forest Classifier...")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# --- 8. Evaluate Model ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n[RESULT] Model Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
print(f"\n[REPORT] Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Rejected (0)', 'Approved (1)']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(f"[MATRIX] Confusion Matrix:")
print(f"   {cm}")

# Feature Importance
feature_importance = dict(zip(feature_cols, [round(float(x), 4) for x in model.feature_importances_]))
sorted_features = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
print(f"\n[IMPORTANCE] Feature Importance:")
for feat, imp in sorted_features.items():
    bar = "#" * int(imp * 50)
    print(f"   {feat:20s}: {imp:.4f} {bar}")

# --- 9. Save Model ---
print("\n[SAVE] Saving model...")

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(model, f)
print(f"   Model saved to: {model_path}")

# --- 10. Save Metadata ---
# Reload original data for stats
df_original = pd.read_csv(dataset_path)

# Calculate dataset statistics for the dashboard
total_records = len(df_original)
approved_count = int(df_original['Status'].sum())
rejected_count = total_records - approved_count

# Intent distribution
intent_counts = df_original['Intent'].value_counts().to_dict()

# Home ownership distribution
home_counts = df_original['Home'].value_counts().to_dict()

# Default distribution
default_counts = df_original['Default'].value_counts().to_dict()

# Age distribution (binned)
age_bins = [0, 25, 30, 35, 40, 50, 200]
age_labels = ['18-25', '26-30', '31-35', '36-40', '41-50', '50+']
df_original['Age_bin'] = pd.cut(df_original['Age'], bins=age_bins, labels=age_labels)
age_distribution = df_original['Age_bin'].value_counts().sort_index().to_dict()
age_distribution = {str(k): int(v) for k, v in age_distribution.items()}

# Income statistics
income_stats = {
    'mean': round(float(df_original['Income'].mean()), 2),
    'median': round(float(df_original['Income'].median()), 2),
    'min': round(float(df_original['Income'].min()), 2),
    'max': round(float(df_original['Income'].max()), 2),
    'std': round(float(df_original['Income'].std()), 2)
}

# Amount statistics
amount_stats = {
    'mean': round(float(df_original['Amount'].mean()), 2),
    'median': round(float(df_original['Amount'].median()), 2),
    'min': round(float(df_original['Amount'].min()), 2),
    'max': round(float(df_original['Amount'].max()), 2)
}

# Rate statistics
rate_clean = df_original['Rate'].dropna()
rate_stats = {
    'mean': round(float(rate_clean.mean()), 2),
    'median': round(float(rate_clean.median()), 2),
    'min': round(float(rate_clean.min()), 2),
    'max': round(float(rate_clean.max()), 2)
}

# Approval rate by intent
approval_by_intent = {}
for intent in df_original['Intent'].unique():
    subset = df_original[df_original['Intent'] == intent]
    approval_rate = round(float(subset['Status'].mean() * 100), 1)
    approval_by_intent[intent] = approval_rate

# Approval rate by home ownership
approval_by_home = {}
for home in df_original['Home'].unique():
    subset = df_original[df_original['Home'] == home]
    approval_rate = round(float(subset['Status'].mean() * 100), 1)
    approval_by_home[home] = approval_rate

metadata = {
    'model_type': 'Random Forest Classifier',
    'accuracy': round(float(accuracy), 4),
    'accuracy_percent': round(float(accuracy * 100), 1),
    'n_estimators': 100,
    'feature_columns': feature_cols,
    'encoding_maps': encoding_maps,
    'feature_importance': sorted_features,
    'classification_report': {
        'precision_rejected': round(float(classification_report(y_test, y_pred, output_dict=True)['0']['precision']), 4),
        'recall_rejected': round(float(classification_report(y_test, y_pred, output_dict=True)['0']['recall']), 4),
        'precision_approved': round(float(classification_report(y_test, y_pred, output_dict=True)['1']['precision']), 4),
        'recall_approved': round(float(classification_report(y_test, y_pred, output_dict=True)['1']['recall']), 4),
    },
    'dataset_stats': {
        'total_records': total_records,
        'approved_count': approved_count,
        'rejected_count': rejected_count,
        'approval_rate': round(approved_count / total_records * 100, 1),
        'intent_distribution': {k: int(v) for k, v in intent_counts.items()},
        'home_distribution': {k: int(v) for k, v in home_counts.items()},
        'default_distribution': {k: int(v) for k, v in default_counts.items()},
        'age_distribution': age_distribution,
        'income_stats': income_stats,
        'amount_stats': amount_stats,
        'rate_stats': rate_stats,
        'approval_by_intent': approval_by_intent,
        'approval_by_home': approval_by_home
    },
    'confusion_matrix': cm.tolist()
}

metadata_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_metadata.json')
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"   Metadata saved to: {metadata_path}")

print("\n" + "=" * 60)
print(f"  TRAINING COMPLETE! Accuracy: {accuracy*100:.1f}%")
print("=" * 60)
