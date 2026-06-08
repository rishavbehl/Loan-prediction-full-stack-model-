"""
Loan Approval Prediction - FastAPI Backend (v2.0)
==================================================
Real-Life Banking System with CIBIL Score

Endpoints:
  POST /predict         → Predict loan approval (with CIBIL-based analysis)
  GET  /stats           → Dataset statistics for dashboard
  GET  /health          → Health check
  GET  /model-info      → Model metadata & feature importance
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pickle
import json
import numpy as np
import os
import sys
import io

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ─── Initialize FastAPI ────────────────────────────────────────────
app = FastAPI(
    title="Loan Approval Prediction API — CIBIL Score Banking System",
    description="ML-powered loan approval prediction using Gradient Boosting with CIBIL Score",
    version="2.0.0"
)

# ─── CORS (allow React frontend to connect) ────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Load Model & Metadata ─────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)

model_path = os.path.join(BASE_DIR, 'model.pkl')
metadata_path = os.path.join(BASE_DIR, 'model_metadata.json')

# Load trained model
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Model not found! Run train_model.py first.")
    model = None

# Load metadata
try:
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    print("Metadata loaded successfully!")
except FileNotFoundError:
    print("Metadata not found! Run train_model.py first.")
    metadata = {}


# ─── Request/Response Models ───────────────────────────────────────
class LoanApplication(BaseModel):
    """Input schema for loan prediction — Real Banking System"""
    cibil_score: int = Field(..., ge=300, le=900, description="CIBIL Score (300-900)")
    annual_income: float = Field(..., gt=0, description="Annual income in INR")
    loan_amount: float = Field(..., gt=0, description="Loan amount requested in INR")
    int_rate: float = Field(..., gt=0, le=35, description="Interest rate (%)")
    dti: float = Field(..., ge=0, le=60, description="Debt-to-Income ratio")
    emp_length: float = Field(..., ge=0, le=40, description="Employment length in years")
    term_months: int = Field(..., description="Loan term: 36 or 60 months")
    home_ownership: str = Field(..., description="Home ownership: RENT, OWN, MORTGAGE, OTHER")
    purpose: str = Field(..., description="Loan purpose")
    grade: str = Field(..., description="Credit grade: A, B, C, D, E, F, G")
    open_acc: int = Field(..., ge=0, description="Number of open credit accounts")
    revol_util: float = Field(..., ge=0, le=150, description="Revolving utilization rate (%)")
    total_acc: int = Field(..., ge=0, description="Total number of credit accounts")
    pub_rec: int = Field(..., ge=0, description="Number of public derogatory records")
    mort_acc: int = Field(..., ge=0, description="Number of mortgage accounts")
    delinq_2yrs: int = Field(default=0, ge=0, description="Delinquencies in last 2 years")
    inq_last_6mths: int = Field(default=0, ge=0, description="Credit inquiries in last 6 months")

    model_config = {
        "json_schema_extra": {
            "example": {
                "cibil_score": 720,
                "annual_income": 800000,
                "loan_amount": 500000,
                "int_rate": 12.5,
                "dti": 15.0,
                "emp_length": 5,
                "term_months": 36,
                "home_ownership": "RENT",
                "purpose": "debt_consolidation",
                "grade": "B",
                "open_acc": 8,
                "revol_util": 45.0,
                "total_acc": 15,
                "pub_rec": 0,
                "mort_acc": 0,
                "delinq_2yrs": 0,
                "inq_last_6mths": 1
            }
        }
    }


class PredictionResponse(BaseModel):
    """Output schema for loan prediction"""
    prediction: str        # "Approved" or "Rejected"
    prediction_code: int   # 1 or 0
    confidence: float      # Probability percentage
    risk_level: str        # "Low Risk", "Medium Risk", "High Risk"
    credit_grade: str      # A-E based on CIBIL
    cibil_category: str    # "Excellent", "Good", "Fair", "Poor"
    factors: dict          # Key factors influencing the decision


# ─── Helper Functions ──────────────────────────────────────────────
def encode_input(application: LoanApplication) -> np.ndarray:
    """Convert application data to model-ready numpy array (v5.0 — 22 features)"""
    encoding_maps = metadata.get('encoding_maps', {})

    # Encode categoricals
    home_map = encoding_maps.get('home_ownership', {})
    home_encoded = home_map.get(application.home_ownership.upper(), 3)

    purpose_map = encoding_maps.get('purpose', {})
    purpose_encoded = purpose_map.get(application.purpose.lower().strip(), 0)

    grade_map = encoding_maps.get('grade', {})
    grade_encoded = grade_map.get(application.grade.upper(), 4)

    # Calculate composite features
    annual_inc = application.annual_income if application.annual_income > 0 else 1
    loan_to_income = application.loan_amount / annual_inc
    installment_est = application.loan_amount / application.term_months
    installment_to_income = (installment_est * 12) / annual_inc
    revol_bal_to_income = (application.revol_util * 100) / annual_inc
    credit_history_length = max(0, application.total_acc - application.open_acc)
    derog_score = application.pub_rec + application.delinq_2yrs + application.inq_last_6mths

    # Feature order MUST match training (22 features):
    features = np.array([[
        application.cibil_score,
        application.annual_income,
        application.loan_amount,
        application.int_rate,
        application.dti,
        application.emp_length,
        application.term_months,
        home_encoded,
        purpose_encoded,
        grade_encoded,
        application.open_acc,
        application.revol_util,
        application.total_acc,
        application.pub_rec,
        application.mort_acc,
        application.delinq_2yrs,
        application.inq_last_6mths,
        loan_to_income,
        installment_to_income,
        revol_bal_to_income,
        credit_history_length,
        derog_score
    ]])

    return features


def get_cibil_category(score: int) -> str:
    """Categorize CIBIL score into banking tiers"""
    if score >= 750:
        return "Excellent"
    elif score >= 700:
        return "Good"
    elif score >= 650:
        return "Fair"
    elif score >= 550:
        return "Below Average"
    else:
        return "Poor"


def get_credit_grade_from_cibil(score: int) -> str:
    """Map CIBIL score to credit grade"""
    if score >= 800:
        return "A"
    elif score >= 750:
        return "B"
    elif score >= 700:
        return "C"
    elif score >= 650:
        return "D"
    else:
        return "E"


def get_risk_level(confidence: float, prediction: int) -> str:
    """Determine risk level based on confidence and prediction"""
    if prediction == 1:  # Approved
        if confidence >= 80:
            return "Low Risk"
        elif confidence >= 60:
            return "Medium Risk"
        else:
            return "High Risk"
    else:  # Rejected
        if confidence >= 80:
            return "High Risk"
        elif confidence >= 60:
            return "Medium Risk"
        else:
            return "Low Risk"


def get_key_factors(application: LoanApplication) -> dict:
    """Analyze key factors influencing the prediction — Bank-style analysis"""
    factors = {}

    # 1. CIBIL Score (most important)
    cibil = application.cibil_score
    if cibil >= 750:
        factors['cibil_score'] = f'✅ Excellent CIBIL Score ({cibil}) — Strong creditworthiness'
    elif cibil >= 700:
        factors['cibil_score'] = f'✅ Good CIBIL Score ({cibil}) — Above average credit profile'
    elif cibil >= 650:
        factors['cibil_score'] = f'⚡ Fair CIBIL Score ({cibil}) — Average credit standing'
    elif cibil >= 550:
        factors['cibil_score'] = f'⚠️ Below Average CIBIL Score ({cibil}) — Higher risk profile'
    else:
        factors['cibil_score'] = f'🔴 Poor CIBIL Score ({cibil}) — Significant credit risk'

    # 2. Debt-to-Income Ratio
    if application.dti > 35:
        factors['dti'] = f'⚠️ High DTI Ratio ({application.dti:.1f}%) — Debt burden is concerning'
    elif application.dti > 20:
        factors['dti'] = f'⚡ Moderate DTI Ratio ({application.dti:.1f}%) — Manageable debt level'
    else:
        factors['dti'] = f'✅ Low DTI Ratio ({application.dti:.1f}%) — Healthy debt profile'

    # 3. Interest Rate
    if application.int_rate > 18:
        factors['interest_rate'] = f'⚠️ Very High Interest Rate ({application.int_rate:.1f}%) — Subprime lending'
    elif application.int_rate > 12:
        factors['interest_rate'] = f'⚡ Moderate Interest Rate ({application.int_rate:.1f}%)'
    else:
        factors['interest_rate'] = f'✅ Competitive Interest Rate ({application.int_rate:.1f}%)'

    # 4. Annual Income
    if application.annual_income >= 1000000:
        factors['income'] = f'✅ Strong Income (₹{application.annual_income:,.0f}/year)'
    elif application.annual_income >= 500000:
        factors['income'] = f'⚡ Moderate Income (₹{application.annual_income:,.0f}/year)'
    else:
        factors['income'] = f'⚠️ Lower Income (₹{application.annual_income:,.0f}/year)'

    # 5. Employment Length
    if application.emp_length >= 5:
        factors['employment'] = f'✅ Stable Employment ({application.emp_length:.0f} years)'
    elif application.emp_length >= 2:
        factors['employment'] = f'⚡ Moderate Employment ({application.emp_length:.0f} years)'
    else:
        factors['employment'] = f'⚠️ Short Employment ({application.emp_length:.1f} years)'

    # 6. Revolving Utilization
    if application.revol_util > 80:
        factors['revolving_util'] = f'⚠️ High Credit Utilization ({application.revol_util:.1f}%)'
    elif application.revol_util > 50:
        factors['revolving_util'] = f'⚡ Moderate Credit Utilization ({application.revol_util:.1f}%)'
    else:
        factors['revolving_util'] = f'✅ Low Credit Utilization ({application.revol_util:.1f}%)'

    # 7. Public Records
    if application.pub_rec > 0:
        factors['public_records'] = f'⚠️ {application.pub_rec} Public Record(s) — Negative credit event'
    else:
        factors['public_records'] = '✅ No Public Records — Clean credit history'

    # 8. Delinquencies
    if application.delinq_2yrs > 0:
        factors['delinquencies'] = f'⚠️ {application.delinq_2yrs} Delinquency(ies) in past 2 years'
    else:
        factors['delinquencies'] = '✅ No Recent Delinquencies'

    # 9. Loan-to-Income Ratio
    lti = application.loan_amount / application.annual_income if application.annual_income > 0 else 999
    if lti > 0.5:
        factors['loan_to_income'] = f'⚠️ High Loan-to-Income ({lti:.1%}) — Loan is large relative to income'
    elif lti > 0.25:
        factors['loan_to_income'] = f'⚡ Moderate Loan-to-Income ({lti:.1%})'
    else:
        factors['loan_to_income'] = f'✅ Low Loan-to-Income ({lti:.1%}) — Comfortable repayment capacity'

    return factors


# ─── API Endpoints ─────────────────────────────────────────────────

@app.get("/")
async def root():
    """Root endpoint - API info"""
    return {
        "message": "Loan Approval Prediction API — CIBIL Score Banking System",
        "version": "2.0.0",
        "endpoints": {
            "POST /predict": "Submit a loan application for prediction",
            "GET /stats": "Get dataset statistics for dashboard",
            "GET /model-info": "Get model metadata and performance",
            "GET /health": "Health check"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "metadata_loaded": len(metadata) > 0,
        "model_version": "2.0 — CIBIL Banking"
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_loan(application: LoanApplication):
    """
    Predict loan approval based on applicant data.
    
    Returns prediction (Approved/Rejected), confidence %, risk level,
    CIBIL-based credit grade, and detailed factor analysis.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train_model.py first.")

    try:
        # Encode input features
        features = encode_input(application)

        # Get prediction and probability
        prediction = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0]

        # Confidence is the probability of the predicted class
        confidence = round(float(max(probabilities)) * 100, 1)

        # Get analysis
        risk_level = get_risk_level(confidence, prediction)
        cibil_category = get_cibil_category(application.cibil_score)
        credit_grade = get_credit_grade_from_cibil(application.cibil_score)
        factors = get_key_factors(application)

        return PredictionResponse(
            prediction="Approved" if prediction == 1 else "Rejected",
            prediction_code=prediction,
            confidence=confidence,
            risk_level=risk_level,
            credit_grade=credit_grade,
            cibil_category=cibil_category,
            factors=factors
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/stats")
async def get_stats():
    """
    Get dataset statistics for the dashboard.
    """
    if not metadata:
        raise HTTPException(status_code=503, detail="Metadata not loaded. Run train_model.py first.")

    return metadata.get('dataset_stats', {})


@app.get("/model-info")
async def get_model_info():
    """
    Get model performance info and feature importance.
    """
    if not metadata:
        raise HTTPException(status_code=503, detail="Metadata not loaded. Run train_model.py first.")

    return {
        'model_type': metadata.get('model_type'),
        'accuracy': metadata.get('accuracy'),
        'accuracy_percent': metadata.get('accuracy_percent'),
        'train_accuracy': metadata.get('train_accuracy'),
        'cv_mean_accuracy': metadata.get('cv_mean_accuracy'),
        'cv_std_accuracy': metadata.get('cv_std_accuracy'),
        'auc_roc': metadata.get('auc_roc'),
        'f1_score': metadata.get('f1_score'),
        'overfit_gap': metadata.get('overfit_gap'),
        'n_estimators_actual': metadata.get('n_estimators_actual'),
        'learning_rate': metadata.get('learning_rate'),
        'max_depth': metadata.get('max_depth'),
        'feature_importance': metadata.get('feature_importance'),
        'feature_display_names': metadata.get('feature_display_names'),
        'classification_report': metadata.get('classification_report'),
        'confusion_matrix': metadata.get('confusion_matrix'),
        'feature_columns': metadata.get('feature_columns'),
        'cv_fold_scores': metadata.get('cv_fold_scores'),
    }


# ─── Run Server ────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("\nStarting Loan Prediction API Server (v2.0 — CIBIL Banking)...")
    print("   Docs:   http://localhost:8000/docs")
    print("   Health: http://localhost:8000/health")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
