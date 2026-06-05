import { useState } from 'react';
import { predictLoan } from '../api/api';
import './PredictionForm.css';

/**
 * PredictionForm Page (v2.0) — Multi-Step Banking Application
 * =============================================================
 * Real-life banking form with:
 * - Step 1: Personal & Employment Info
 * - Step 2: Credit Profile (CIBIL Score with visual gauge)
 * - Step 3: Loan Details
 * - Animated result with credit grade badge
 */
function PredictionForm() {
  const [step, setStep] = useState(1);
  const totalSteps = 3;

  const [formData, setFormData] = useState({
    cibil_score: '',
    annual_income: '',
    loan_amount: '',
    int_rate: '',
    dti: '',
    emp_length: '',
    term_months: '36',
    home_ownership: 'RENT',
    purpose: 'debt_consolidation',
    grade: 'C',
    open_acc: '',
    revol_util: '',
    total_acc: '',
    pub_rec: '0',
    mort_acc: '0',
    delinq_2yrs: '0',
    inq_last_6mths: '0',
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleChange = (e) => {
    const { name, value } = e.target;
    const updated = { ...formData, [name]: value };

    // Auto-calculate DTI when income or loan amount changes
    if ((name === 'annual_income' || name === 'loan_amount') && updated.annual_income && updated.loan_amount) {
      const income = parseFloat(updated.annual_income);
      const amount = parseFloat(updated.loan_amount);
      if (income > 0) {
        updated.dti = ((amount / income) * 100).toFixed(1);
      }
    }

    // Auto-suggest grade based on CIBIL score
    if (name === 'cibil_score') {
      const score = parseInt(value);
      if (!isNaN(score)) {
        if (score >= 800) updated.grade = 'A';
        else if (score >= 750) updated.grade = 'B';
        else if (score >= 700) updated.grade = 'C';
        else if (score >= 650) updated.grade = 'D';
        else if (score >= 550) updated.grade = 'E';
        else if (score >= 450) updated.grade = 'F';
        else updated.grade = 'G';
      }
    }

    setFormData(updated);
    setError('');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setResult(null);

    try {
      const payload = {
        cibil_score: parseInt(formData.cibil_score),
        annual_income: parseFloat(formData.annual_income),
        loan_amount: parseFloat(formData.loan_amount),
        int_rate: parseFloat(formData.int_rate),
        dti: parseFloat(formData.dti),
        emp_length: parseFloat(formData.emp_length),
        term_months: parseInt(formData.term_months),
        home_ownership: formData.home_ownership,
        purpose: formData.purpose,
        grade: formData.grade,
        open_acc: parseInt(formData.open_acc),
        revol_util: parseFloat(formData.revol_util),
        total_acc: parseInt(formData.total_acc),
        pub_rec: parseInt(formData.pub_rec),
        mort_acc: parseInt(formData.mort_acc),
        delinq_2yrs: parseInt(formData.delinq_2yrs),
        inq_last_6mths: parseInt(formData.inq_last_6mths),
      };

      const prediction = await predictLoan(payload);
      setResult(prediction);
    } catch (err) {
      setError(err.message || 'Failed to get prediction. Make sure the backend is running!');
    } finally {
      setLoading(false);
    }
  };

  const resetForm = () => {
    setFormData({
      cibil_score: '', annual_income: '', loan_amount: '', int_rate: '',
      dti: '', emp_length: '', term_months: '36', home_ownership: 'RENT',
      purpose: 'debt_consolidation', grade: 'C', open_acc: '', revol_util: '',
      total_acc: '', pub_rec: '0', mort_acc: '0', delinq_2yrs: '0',
      inq_last_6mths: '0',
    });
    setResult(null);
    setError('');
    setStep(1);
  };

  const nextStep = () => setStep(Math.min(step + 1, totalSteps));
  const prevStep = () => setStep(Math.max(step - 1, 1));

  // CIBIL Score gauge helpers
  const cibilScore = parseInt(formData.cibil_score) || 300;
  const cibilPercent = Math.min(Math.max((cibilScore - 300) / 600 * 100, 0), 100);
  const cibilColor = cibilScore >= 750 ? '#22c55e' :
                     cibilScore >= 700 ? '#84cc16' :
                     cibilScore >= 650 ? '#f59e0b' :
                     cibilScore >= 550 ? '#f97316' : '#ef4444';
  const cibilCategory = cibilScore >= 750 ? 'Excellent' :
                        cibilScore >= 700 ? 'Good' :
                        cibilScore >= 650 ? 'Fair' :
                        cibilScore >= 550 ? 'Below Average' : 'Poor';

  // SVG gauge arc calculation
  const gaugeRadius = 80;
  const gaugeCircumference = Math.PI * gaugeRadius;
  const gaugeFill = (cibilPercent / 100) * gaugeCircumference;

  return (
    <div className="predict-page">
      {/* Hero Section */}
      <div className="predict-hero animate-fade-in-up">
        <div className="hero-badge">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="14" height="14">
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
          </svg>
          Indian Banking System — CIBIL Score
        </div>
        <h1 className="hero-title">
          Loan Approval <span className="gradient-text">Predictor</span>
        </h1>
        <p className="hero-subtitle">
          AI-powered banking decision engine using CIBIL Score, income analysis, credit profile,
          and 17 real banking features for accurate loan approval predictions.
        </p>
      </div>

      <div className="predict-content">
        {/* Step Progress Bar */}
        <div className="step-progress animate-fade-in-up stagger-1">
          {[1, 2, 3].map((s) => (
            <div key={s} className={`step-item ${step >= s ? 'active' : ''} ${step === s ? 'current' : ''}`} onClick={() => setStep(s)}>
              <div className="step-circle">
                {step > s ? (
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
                    <polyline points="20 6 9 17 4 12"/>
                  </svg>
                ) : s}
              </div>
              <span className="step-label">
                {s === 1 ? 'Personal Info' : s === 2 ? 'Credit Profile' : 'Loan Details'}
              </span>
            </div>
          ))}
          <div className="step-connector">
            <div className="step-connector-fill" style={{ width: `${((step - 1) / (totalSteps - 1)) * 100}%` }}></div>
          </div>
        </div>

        {/* Form */}
        <form className="predict-form animate-fade-in-up stagger-2" onSubmit={handleSubmit} id="prediction-form">
          {/* Step 1: Personal & Employment */}
          {step === 1 && (
            <div className="form-step animate-fade-in">
              <div className="form-header">
                <div className="form-step-badge">Step 1 of 3</div>
                <h2>Personal & Employment Information</h2>
                <p>Enter the applicant's personal and employment details</p>
              </div>

              <div className="form-grid">
                {/* Annual Income */}
                <div className="form-group">
                  <label htmlFor="annual_income">
                    <span className="label-icon">
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 1v22M17 5H9.5a3.5 3.5 0 000 7h5a3.5 3.5 0 010 7H6"/></svg>
                    </span>
                    Annual Income (₹)
                  </label>
                  <input
                    type="number" id="annual_income" name="annual_income"
                    value={formData.annual_income} onChange={handleChange}
                    placeholder="e.g., 800000" min="1" required
                  />
                  <span className="form-hint">Gross annual income in Indian Rupees</span>
                </div>

                {/* Employment Length */}
                <div className="form-group">
                  <label htmlFor="emp_length">
                    <span className="label-icon">
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="2" y="7" width="20" height="14" rx="2" ry="2"/><path d="M16 21V5a2 2 0 00-2-2h-4a2 2 0 00-2 2v16"/></svg>
                    </span>
                    Employment Length (years)
                  </label>
                  <input
                    type="number" id="emp_length" name="emp_length"
                    value={formData.emp_length} onChange={handleChange}
                    placeholder="e.g., 5" min="0" max="40" step="0.5" required
                  />
                </div>

                {/* Home Ownership */}
                <div className="form-group">
                  <label htmlFor="home_ownership">
                    <span className="label-icon">
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M3 9l9-7 9 7v11a2 2 0 01-2 2H5a2 2 0 01-2-2V9z"/><path d="M9 22V12h6v10"/></svg>
                    </span>
                    Home Ownership
                  </label>
                  <select id="home_ownership" name="home_ownership" value={formData.home_ownership} onChange={handleChange} required>
                    <option value="RENT">Rent</option>
                    <option value="OWN">Own</option>
                    <option value="MORTGAGE">Mortgage / Home Loan</option>
                    <option value="OTHER">Other</option>
                  </select>
                </div>

                {/* Mortgage Accounts */}
                <div className="form-group">
                  <label htmlFor="mort_acc">
                    <span className="label-icon">
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="1" y="4" width="22" height="16" rx="2" ry="2"/><line x1="1" y1="10" x2="23" y2="10"/></svg>
                    </span>
                    Mortgage Accounts
                  </label>
                  <input
                    type="number" id="mort_acc" name="mort_acc"
                    value={formData.mort_acc} onChange={handleChange}
                    placeholder="e.g., 0" min="0" required
                  />
                </div>
              </div>
            </div>
          )}

          {/* Step 2: Credit Profile */}
          {step === 2 && (
            <div className="form-step animate-fade-in">
              <div className="form-header">
                <div className="form-step-badge">Step 2 of 3</div>
                <h2>Credit Profile</h2>
                <p>Enter the applicant's credit information and CIBIL score</p>
              </div>

              {/* CIBIL Score Gauge */}
              <div className="cibil-gauge-section">
                <div className="cibil-input-wrapper">
                  <label htmlFor="cibil_score" className="cibil-label">CIBIL Score</label>
                  <input
                    type="number" id="cibil_score" name="cibil_score"
                    value={formData.cibil_score} onChange={handleChange}
                    placeholder="300-900" min="300" max="900" required
                    className="cibil-input"
                  />
                </div>
                <div className="cibil-gauge">
                  <svg viewBox="0 0 200 120" className="gauge-svg">
                    {/* Background arc */}
                    <path
                      d="M 20 100 A 80 80 0 0 1 180 100"
                      fill="none"
                      stroke="rgba(255,255,255,0.06)"
                      strokeWidth="12"
                      strokeLinecap="round"
                    />
                    {/* Colored fill arc */}
                    <path
                      d="M 20 100 A 80 80 0 0 1 180 100"
                      fill="none"
                      stroke={cibilColor}
                      strokeWidth="12"
                      strokeLinecap="round"
                      strokeDasharray={`${gaugeFill} ${gaugeCircumference}`}
                      className="gauge-fill-path"
                    />
                    {/* Score text */}
                    <text x="100" y="80" textAnchor="middle" className="gauge-score">
                      {formData.cibil_score || '—'}
                    </text>
                    <text x="100" y="100" textAnchor="middle" className="gauge-label" fill={cibilColor}>
                      {formData.cibil_score ? cibilCategory : 'Enter Score'}
                    </text>
                    {/* Scale labels */}
                    <text x="18" y="115" textAnchor="middle" className="gauge-scale">300</text>
                    <text x="100" y="18" textAnchor="middle" className="gauge-scale">600</text>
                    <text x="182" y="115" textAnchor="middle" className="gauge-scale">900</text>
                  </svg>
                </div>
              </div>

              <div className="form-grid">
                {/* Credit Grade */}
                <div className="form-group">
                  <label htmlFor="grade">
                    <span className="label-icon">
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/></svg>
                    </span>
                    Credit Grade
                  </label>
                  <select id="grade" name="grade" value={formData.grade} onChange={handleChange} required>
                    <option value="A">A — Excellent</option>
                    <option value="B">B — Very Good</option>
                    <option value="C">C — Good</option>
                    <option value="D">D — Fair</option>
                    <option value="E">E — Below Average</option>
                    <option value="F">F — Poor</option>
                    <option value="G">G — Very Poor</option>
                  </select>
                  <span className="form-hint">Auto-suggested from CIBIL score</span>
                </div>

                {/* Open Accounts */}
                <div className="form-group">
                  <label htmlFor="open_acc">
                    <span className="label-icon">
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0110 0v4"/></svg>
                    </span>
                    Open Credit Accounts
                  </label>
                  <input
                    type="number" id="open_acc" name="open_acc"
                    value={formData.open_acc} onChange={handleChange}
                    placeholder="e.g., 8" min="0" required
                  />
                </div>

                {/* Revolving Utilization */}
                <div className="form-group">
                  <label htmlFor="revol_util">
                    <span className="label-icon">
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>
                    </span>
                    Credit Utilization (%)
                  </label>
                  <input
                    type="number" id="revol_util" name="revol_util"
                    value={formData.revol_util} onChange={handleChange}
                    placeholder="e.g., 45" min="0" max="150" step="0.1" required
                  />
                  <span className="form-hint">Revolving credit utilization rate</span>
                </div>

                {/* Total Accounts */}
                <div className="form-group">
                  <label htmlFor="total_acc">
                    <span className="label-icon">
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M17 21v-2a4 4 0 00-4-4H5a4 4 0 00-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 00-3-3.87"/><path d="M16 3.13a4 4 0 010 7.75"/></svg>
                    </span>
                    Total Credit Accounts
                  </label>
                  <input
                    type="number" id="total_acc" name="total_acc"
                    value={formData.total_acc} onChange={handleChange}
                    placeholder="e.g., 15" min="0" required
                  />
                </div>

                {/* Public Records */}
                <div className="form-group">
                  <label htmlFor="pub_rec">
                    <span className="label-icon">
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
                    </span>
                    Public Records
                  </label>
                  <input
                    type="number" id="pub_rec" name="pub_rec"
                    value={formData.pub_rec} onChange={handleChange}
                    placeholder="e.g., 0" min="0" required
                  />
                  <span className="form-hint">Derogatory public records (bankruptcies, etc.)</span>
                </div>

                {/* Delinquencies */}
                <div className="form-group">
                  <label htmlFor="delinq_2yrs">
                    <span className="label-icon">
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
                    </span>
                    Delinquencies (last 2 years)
                  </label>
                  <input
                    type="number" id="delinq_2yrs" name="delinq_2yrs"
                    value={formData.delinq_2yrs} onChange={handleChange}
                    placeholder="e.g., 0" min="0" required
                  />
                </div>

                {/* Inquiries */}
                <div className="form-group">
                  <label htmlFor="inq_last_6mths">
                    <span className="label-icon">
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
                    </span>
                    Credit Inquiries (last 6 months)
                  </label>
                  <input
                    type="number" id="inq_last_6mths" name="inq_last_6mths"
                    value={formData.inq_last_6mths} onChange={handleChange}
                    placeholder="e.g., 1" min="0" required
                  />
                </div>
              </div>
            </div>
          )}

          {/* Step 3: Loan Details */}
          {step === 3 && (
            <div className="form-step animate-fade-in">
              <div className="form-header">
                <div className="form-step-badge">Step 3 of 3</div>
                <h2>Loan Details</h2>
                <p>Enter the loan amount, term, and purpose</p>
              </div>

              <div className="form-grid">
                {/* Loan Amount */}
                <div className="form-group">
                  <label htmlFor="loan_amount">
                    <span className="label-icon">
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 1v22M17 5H9.5a3.5 3.5 0 000 7h5a3.5 3.5 0 010 7H6"/></svg>
                    </span>
                    Loan Amount (₹)
                  </label>
                  <input
                    type="number" id="loan_amount" name="loan_amount"
                    value={formData.loan_amount} onChange={handleChange}
                    placeholder="e.g., 500000" min="1" required
                  />
                </div>

                {/* Interest Rate */}
                <div className="form-group">
                  <label htmlFor="int_rate">
                    <span className="label-icon">
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="19" y1="5" x2="5" y2="19"/><circle cx="6.5" cy="6.5" r="2.5"/><circle cx="17.5" cy="17.5" r="2.5"/></svg>
                    </span>
                    Interest Rate (%)
                  </label>
                  <input
                    type="number" id="int_rate" name="int_rate"
                    value={formData.int_rate} onChange={handleChange}
                    placeholder="e.g., 12.5" min="0.01" max="35" step="0.01" required
                  />
                </div>

                {/* Loan Term */}
                <div className="form-group">
                  <label htmlFor="term_months">
                    <span className="label-icon">
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
                    </span>
                    Loan Term
                  </label>
                  <select id="term_months" name="term_months" value={formData.term_months} onChange={handleChange} required>
                    <option value="36">36 Months (3 Years)</option>
                    <option value="60">60 Months (5 Years)</option>
                  </select>
                </div>

                {/* DTI */}
                <div className="form-group">
                  <label htmlFor="dti">
                    <span className="label-icon">
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>
                    </span>
                    Debt-to-Income Ratio (%)
                  </label>
                  <input
                    type="number" id="dti" name="dti"
                    value={formData.dti} onChange={handleChange}
                    placeholder="Auto-calculated" min="0" max="60" step="0.1" required
                  />
                  <span className="form-hint">Auto-calculated from Loan Amount / Income</span>
                </div>

                {/* Loan Purpose */}
                <div className="form-group full-width">
                  <label htmlFor="purpose">
                    <span className="label-icon">
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 16V8a2 2 0 00-1-1.73l-7-4a2 2 0 00-2 0l-7 4A2 2 0 003 8v8a2 2 0 001 1.73l7 4a2 2 0 002 0l7-4A2 2 0 0021 16z"/></svg>
                    </span>
                    Loan Purpose
                  </label>
                  <select id="purpose" name="purpose" value={formData.purpose} onChange={handleChange} required>
                    <option value="debt_consolidation">Debt Consolidation</option>
                    <option value="credit_card">Credit Card Refinancing</option>
                    <option value="home_improvement">Home Improvement</option>
                    <option value="major_purchase">Major Purchase</option>
                    <option value="small_business">Small Business</option>
                    <option value="car">Car / Vehicle Loan</option>
                    <option value="medical">Medical Expenses</option>
                    <option value="moving">Moving / Relocation</option>
                    <option value="vacation">Vacation</option>
                    <option value="house">House Purchase</option>
                    <option value="wedding">Wedding</option>
                    <option value="renewable_energy">Renewable Energy</option>
                    <option value="educational">Education</option>
                    <option value="other">Other</option>
                  </select>
                </div>
              </div>
            </div>
          )}

          {/* Error Message */}
          {error && (
            <div className="form-error animate-fade-in" id="form-error">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>
              {error}
            </div>
          )}

          {/* Navigation Buttons */}
          <div className="form-actions">
            {step > 1 && (
              <button type="button" className="btn btn-secondary" onClick={prevStep}>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="15 18 9 12 15 6"/></svg>
                Previous
              </button>
            )}

            {step < totalSteps ? (
              <button type="button" className="btn btn-primary" onClick={nextStep}>
                Next Step
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="9 18 15 12 9 6"/></svg>
              </button>
            ) : (
              <button type="submit" className="btn btn-primary btn-submit" id="submit-btn" disabled={loading}>
                {loading ? (
                  <><span className="spinner"></span> Analyzing Application...</>
                ) : (
                  <>
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
                    Get Bank Decision
                  </>
                )}
              </button>
            )}

            <button type="button" className="btn btn-ghost" onClick={resetForm} id="reset-btn">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M2.5 2v6h6M21.5 22v-6h-6"/><path d="M22 11.5A10 10 0 003.2 7.2M2 12.5a10 10 0 0018.8 4.2"/></svg>
              Reset
            </button>
          </div>
        </form>

        {/* Result Section */}
        {result && (
          <div className="result-section animate-fade-in-up" id="prediction-result">
            <div className={`result-card ${result.prediction_code === 1 ? 'approved' : 'rejected'}`}>
              {/* Credit Grade Badge */}
              <div className="credit-grade-badge-wrapper">
                <div className={`credit-grade-badge grade-${result.credit_grade?.toLowerCase()}`}>
                  <span className="grade-letter">{result.credit_grade}</span>
                  <span className="grade-label">{result.cibil_category}</span>
                </div>
              </div>

              {/* Main Result */}
              <div className="result-status">
                <div className={`status-icon ${result.prediction_code === 1 ? 'approved' : 'rejected'}`}>
                  {result.prediction_code === 1 ? (
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
                  ) : (
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>
                  )}
                </div>
                <div className="status-info">
                  <h2 className="status-label">Loan {result.prediction}</h2>
                  <p className="status-sublabel">{result.risk_level}</p>
                </div>
              </div>

              {/* Confidence Meter */}
              <div className="confidence-section">
                <div className="confidence-header">
                  <span>Model Confidence</span>
                  <span className="confidence-value">{result.confidence}%</span>
                </div>
                <div className="confidence-bar">
                  <div
                    className={`confidence-fill ${result.prediction_code === 1 ? 'approved' : 'rejected'}`}
                    style={{ width: `${result.confidence}%` }}
                  ></div>
                </div>
              </div>

              {/* Bank Decision Summary */}
              <div className="bank-decision-summary">
                <div className="decision-row">
                  <span className="decision-label">CIBIL Score</span>
                  <span className="decision-value">{formData.cibil_score} — {result.cibil_category}</span>
                </div>
                <div className="decision-row">
                  <span className="decision-label">Credit Grade</span>
                  <span className={`decision-value grade-text grade-${result.credit_grade?.toLowerCase()}`}>{result.credit_grade}</span>
                </div>
                <div className="decision-row">
                  <span className="decision-label">Risk Assessment</span>
                  <span className="decision-value">{result.risk_level}</span>
                </div>
              </div>

              {/* Key Factors */}
              <div className="factors-section">
                <h3>Banking Analysis — Key Factors</h3>
                <div className="factors-list">
                  {Object.entries(result.factors).map(([key, value]) => (
                    <div className="factor-item" key={key}>
                      <span className="factor-label">{key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}</span>
                      <span className="factor-value">{value}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default PredictionForm;
