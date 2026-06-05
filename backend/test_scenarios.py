import pickle
import pandas as pd
import numpy as np
import json
import os
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'model.pkl')
metadata_path = os.path.join(BASE_DIR, 'model_metadata.json')

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(metadata_path, 'r') as f:
    metadata = json.load(f)

encoding_maps = metadata.get('encoding_maps', {})
feature_names = metadata.get('feature_columns', [])

def encode_input(app):
    home_map = encoding_maps.get('home_ownership', {})
    home_encoded = home_map.get(app['home_ownership'].upper(), 3)
    
    purpose_map = encoding_maps.get('purpose', {})
    purpose_encoded = purpose_map.get(app['purpose'].lower().strip(), 0)
    
    grade_map = encoding_maps.get('grade', {})
    grade_encoded = grade_map.get(app['grade'].upper(), 4)
    
    annual_inc = app['annual_income'] if app['annual_income'] > 0 else 1
    features = {
        'cibil_score': app['cibil_score'],
        'annual_income': app['annual_income'],
        'loan_amount': app['loan_amount'],
        'int_rate': app['int_rate'],
        'dti': app['dti'],
        'emp_length_years': app['emp_length'],
        'term_months': app['term_months'],
        'home_encoded': home_encoded,
        'purpose_encoded': purpose_encoded,
        'grade_encoded': grade_encoded,
        'open_acc': app['open_acc'],
        'revol_util': app['revol_util'],
        'total_acc': app['total_acc'],
        'pub_rec': app['pub_rec'],
        'mort_acc': app['mort_acc'],
        'delinq_2yrs': app['delinq_2yrs'],
        'inq_last_6mths': app['inq_last_6mths'],
        'loan_to_income': app['loan_amount'] / annual_inc,
        'credit_burden': app['dti'] * (app['revol_util'] / 100.0)
    }
    
    # Ensure order and names match training perfectly
    # The model expects names like 'dti_clean' instead of 'dti' because of the training script.
    ordered_features = {}
    for api_name, model_name in zip(feature_names, model.feature_names_in_):
        ordered_features[model_name] = [features[api_name]]
    return pd.DataFrame(ordered_features)

# REAL WORLD SCENARIOS
scenarios = [
    {
        "name": "Scenario 1: Ideal Prime Customer (High Income, 800+ CIBIL, Low DTI)",
        "expected": "Approved (High Confidence)",
        "data": {
            "cibil_score": 820, "annual_income": 1500000, "loan_amount": 500000,
            "int_rate": 8.5, "dti": 10.0, "emp_length": 8, "term_months": 36,
            "home_ownership": "OWN", "purpose": "home_improvement", "grade": "A",
            "open_acc": 5, "revol_util": 15.0, "total_acc": 10, "pub_rec": 0,
            "mort_acc": 1, "delinq_2yrs": 0, "inq_last_6mths": 0
        }
    },
    {
        "name": "Scenario 2: High Risk Defaulter (Poor CIBIL, Recent Delinquencies)",
        "expected": "Rejected (High Confidence)",
        "data": {
            "cibil_score": 520, "annual_income": 300000, "loan_amount": 800000,
            "int_rate": 24.0, "dti": 45.0, "emp_length": 1, "term_months": 60,
            "home_ownership": "RENT", "purpose": "debt_consolidation", "grade": "F",
            "open_acc": 12, "revol_util": 95.0, "total_acc": 15, "pub_rec": 2,
            "mort_acc": 0, "delinq_2yrs": 3, "inq_last_6mths": 5
        }
    },
    {
        "name": "Scenario 3: Over-leveraged Middle Class (Good Income but Maxed Credit Cards)",
        "expected": "Likely Rejected or Borderline",
        "data": {
            "cibil_score": 680, "annual_income": 800000, "loan_amount": 1000000,
            "int_rate": 15.0, "dti": 55.0, "emp_length": 4, "term_months": 60,
            "home_ownership": "RENT", "purpose": "credit_card", "grade": "C",
            "open_acc": 8, "revol_util": 88.0, "total_acc": 12, "pub_rec": 0,
            "mort_acc": 0, "delinq_2yrs": 0, "inq_last_6mths": 2
        }
    },
    {
        "name": "Scenario 4: Young Professional First Loan (Average CIBIL, Thin File)",
        "expected": "Borderline / Approved with Medium Confidence",
        "data": {
            "cibil_score": 710, "annual_income": 600000, "loan_amount": 200000,
            "int_rate": 12.0, "dti": 20.0, "emp_length": 1, "term_months": 36,
            "home_ownership": "RENT", "purpose": "car", "grade": "B",
            "open_acc": 3, "revol_util": 30.0, "total_acc": 4, "pub_rec": 0,
            "mort_acc": 0, "delinq_2yrs": 0, "inq_last_6mths": 1
        }
    }
]

print("="*70)
print("  TESTING REAL-WORLD BANKING SCENARIOS")
print("="*70)

for idx, s in enumerate(scenarios):
    print(f"\n[{idx+1}] {s['name']}")
    print(f"    Expected logical outcome: {s['expected']}")
    
    features = encode_input(s['data'])
    pred = int(model.predict(features)[0])
    prob = model.predict_proba(features)[0]
    
    pred_text = "APPROVED" if pred == 1 else "REJECTED"
    confidence = max(prob) * 100
    
    print(f"    MODEL PREDICTION: ----> {pred_text} (Confidence: {confidence:.1f}%)")
    print(f"    Probabilities: Approve={prob[1]*100:.1f}%, Reject={prob[0]*100:.1f}%")
    
    if (pred == 1 and "Approved" in s['expected']) or (pred == 0 and "Rejected" in s['expected']):
        print("    ✅ Logical Match")
    else:
        print("    ⚠️ Logical Mismatch or Borderline case")

print("\n" + "="*70)
