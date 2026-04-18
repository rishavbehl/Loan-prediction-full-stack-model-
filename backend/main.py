"""
Loan Approval Prediction - FastAPI Backend
==========================================
Endpoints:
  POST /predict         → Predict loan approval
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
    title="Loan Approval Prediction API",
    description="ML-powered loan approval prediction using Random Forest",
    version="1.0.0"
)

# ─── CORS (allow React frontend to connect) ────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
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
    print("✅ Model loaded successfully!")
except FileNotFoundError:
    print("❌ Model not found! Run train_model.py first.")
    model = None

# Load metadata
try:
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    print("✅ Metadata loaded successfully!")
except FileNotFoundError:
    print("❌ Metadata not found! Run train_model.py first.")
    metadata = {}


# ─── Request/Response Models ───────────────────────────────────────
class LoanApplication(BaseModel):
    """Input schema for loan prediction"""
    age: int = Field(..., ge=18, le=100, description="Applicant's age (18-100)")
    income: float = Field(..., gt=0, description="Annual income")
    home: str = Field(..., description="Home ownership: RENT, OWN, MORTGAGE, OTHER")
    emp_length: float = Field(..., ge=0, description="Employment length in years")
    intent: str = Field(..., description="Loan intent: PERSONAL, EDUCATION, MEDICAL, VENTURE, HOMEIMPROVEMENT, DEBTCONSOLIDATION")
    amount: float = Field(..., gt=0, description="Loan amount requested")
    rate: float = Field(..., gt=0, le=30, description="Interest rate")
    percent_income: float = Field(..., ge=0, le=1, description="Loan amount as % of income (0-1)")
    default: str = Field(..., description="Previous default: Y or N")
    cred_length: int = Field(..., ge=0, description="Credit history length in years")

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 25,
                "income": 60000,
                "home": "RENT",
                "emp_length": 3,
                "intent": "PERSONAL",
                "amount": 10000,
                "rate": 11.5,
                "percent_income": 0.17,
                "default": "N",
                "cred_length": 4
            }
        }
    }


class PredictionResponse(BaseModel):
    """Output schema for loan prediction"""
    prediction: str  # "Approved" or "Rejected"
    prediction_code: int  # 1 or 0
    confidence: float  # Probability percentage
    risk_level: str  # "Low Risk", "Medium Risk", "High Risk"
    factors: dict  # Key factors influencing the decision


# ─── Helper Functions ──────────────────────────────────────────────
def encode_input(application: LoanApplication) -> np.ndarray:
    """Convert application data to model-ready numpy array"""
    encoding_maps = metadata.get('encoding_maps', {})

    # Encode 'Home'
    home_map = encoding_maps.get('Home', {})
    home_encoded = home_map.get(application.home.upper(), 0)

    # Encode 'Intent'
    intent_map = encoding_maps.get('Intent', {})
    intent_encoded = intent_map.get(application.intent.upper(), 0)

    # Encode 'Default'
    default_map = encoding_maps.get('Default', {})
    default_encoded = default_map.get(application.default.upper(), 0)

    # Feature order: Age, Income, Home, Emp_length, Intent, Amount, Rate, Percent_income, Default, Cred_length
    features = np.array([[
        application.age,
        application.income,
        home_encoded,
        application.emp_length,
        intent_encoded,
        application.amount,
        application.rate,
        application.percent_income,
        default_encoded,
        application.cred_length
    ]])

    return features


def get_risk_level(confidence: float, prediction: int) -> str:
    """Determine risk level based on confidence and prediction"""
    if prediction == 0:  # Approved (0 = Non-default)
        if confidence >= 80:
            return "Low Risk"
        elif confidence >= 60:
            return "Medium Risk"
        else:
            return "High Risk"
    else:  # Rejected (1 = Default)
        if confidence >= 80:
            return "High Risk"
        elif confidence >= 60:
            return "Medium Risk"
        else:
            return "Low Risk"


def get_key_factors(application: LoanApplication) -> dict:
    """Analyze key factors influencing the prediction"""
    factors = {}

    # Debt-to-income ratio analysis
    if application.percent_income > 0.4:
        factors['debt_to_income'] = '⚠️ High debt-to-income ratio ({:.0%})'.format(application.percent_income)
    elif application.percent_income > 0.2:
        factors['debt_to_income'] = '⚡ Moderate debt-to-income ratio ({:.0%})'.format(application.percent_income)
    else:
        factors['debt_to_income'] = '✅ Low debt-to-income ratio ({:.0%})'.format(application.percent_income)

    # Interest rate analysis
    if application.rate > 15:
        factors['interest_rate'] = '⚠️ High interest rate ({:.1f}%)'.format(application.rate)
    elif application.rate > 10:
        factors['interest_rate'] = '⚡ Moderate interest rate ({:.1f}%)'.format(application.rate)
    else:
        factors['interest_rate'] = '✅ Low interest rate ({:.1f}%)'.format(application.rate)

    # Income analysis
    if application.income >= 100000:
        factors['income'] = '✅ Strong income (${:,.0f})'.format(application.income)
    elif application.income >= 50000:
        factors['income'] = '⚡ Moderate income (${:,.0f})'.format(application.income)
    else:
        factors['income'] = '⚠️ Lower income (${:,.0f})'.format(application.income)

    # Credit history
    if application.cred_length >= 4:
        factors['credit_history'] = '✅ Good credit history ({} years)'.format(application.cred_length)
    elif application.cred_length >= 2:
        factors['credit_history'] = '⚡ Short credit history ({} years)'.format(application.cred_length)
    else:
        factors['credit_history'] = '⚠️ Very short credit history ({} years)'.format(application.cred_length)

    # Previous default
    if application.default.upper() == 'Y':
        factors['previous_default'] = '⚠️ Has previous default on record'
    else:
        factors['previous_default'] = '✅ No previous defaults'

    # Employment length
    if application.emp_length >= 5:
        factors['employment'] = '✅ Stable employment ({:.0f} years)'.format(application.emp_length)
    elif application.emp_length >= 2:
        factors['employment'] = '⚡ Moderate employment history ({:.0f} years)'.format(application.emp_length)
    else:
        factors['employment'] = '⚠️ Short employment history ({:.0f} years)'.format(application.emp_length)

    return factors


# ─── API Endpoints ─────────────────────────────────────────────────

@app.get("/")
async def root():
    """Root endpoint - API info"""
    return {
        "message": "🏦 Loan Approval Prediction API",
        "version": "1.0.0",
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
        "metadata_loaded": len(metadata) > 0
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_loan(application: LoanApplication):
    """
    Predict loan approval based on applicant data.
    
    Returns prediction (Approved/Rejected), confidence %, risk level, and key factors.
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

        # Get risk level and key factors
        risk_level = get_risk_level(confidence, prediction)
        factors = get_key_factors(application)

        return PredictionResponse(
            prediction="Approved" if prediction == 0 else "Rejected",
            prediction_code=prediction,
            confidence=confidence,
            risk_level=risk_level,
            factors=factors
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/stats")
async def get_stats():
    """
    Get dataset statistics for the dashboard.
    
    Returns approval rates, distributions, income stats, etc.
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
        'n_estimators': metadata.get('n_estimators'),
        'feature_importance': metadata.get('feature_importance'),
        'classification_report': metadata.get('classification_report'),
        'confusion_matrix': metadata.get('confusion_matrix'),
        'feature_columns': metadata.get('feature_columns')
    }


# ─── Run Server ────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting Loan Prediction API Server...")
    print("   Docs:   http://localhost:8000/docs")
    print("   Health: http://localhost:8000/health")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
