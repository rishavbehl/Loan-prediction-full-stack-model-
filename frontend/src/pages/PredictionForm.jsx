import { useState } from 'react';
import { predictLoan } from '../api/api';
import './PredictionForm.css';

/**
 * PredictionForm Page
 * ===================
 * Beautiful loan application form with real-time prediction.
 * Users fill in their details and get instant AI-powered results.
 */
function PredictionForm() {
  const [formData, setFormData] = useState({
    age: '',
    income: '',
    home: 'RENT',
    emp_length: '',
    intent: 'PERSONAL',
    amount: '',
    rate: '',
    percent_income: '',
    default: 'N',
    cred_length: '',
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // Auto-calculate percent_income when income or amount changes
  const handleChange = (e) => {
    const { name, value } = e.target;
    const updated = { ...formData, [name]: value };

    // Auto-calculate debt-to-income ratio
    if ((name === 'income' || name === 'amount') && updated.income && updated.amount) {
      const income = parseFloat(updated.income);
      const amount = parseFloat(updated.amount);
      if (income > 0) {
        updated.percent_income = (amount / income).toFixed(2);
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
      // Build the request payload
      const payload = {
        age: parseInt(formData.age),
        income: parseFloat(formData.income),
        home: formData.home,
        emp_length: parseFloat(formData.emp_length),
        intent: formData.intent,
        amount: parseFloat(formData.amount),
        rate: parseFloat(formData.rate),
        percent_income: parseFloat(formData.percent_income),
        default: formData.default,
        cred_length: parseInt(formData.cred_length),
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
      age: '', income: '', home: 'RENT', emp_length: '',
      intent: 'PERSONAL', amount: '', rate: '',
      percent_income: '', default: 'N', cred_length: '',
    });
    setResult(null);
    setError('');
  };

  return (
    <div className="predict-page">
      {/* Hero Section */}
      <div className="predict-hero animate-fade-in-up">
        <div className="hero-badge">AI-Powered Prediction</div>
        <h1 className="hero-title">
          Loan Approval <span className="gradient-text">Predictor</span>
        </h1>
        <p className="hero-subtitle">
          Fill in the application details below and our machine learning model will 
          predict whether the loan should be approved or rejected with confidence scoring.
        </p>
      </div>

      <div className="predict-content">
        {/* Form Section */}
        <form className="predict-form animate-fade-in-up stagger-2" onSubmit={handleSubmit} id="prediction-form">
          <div className="form-header">
            <h2>Application Details</h2>
            <p>Enter the applicant's information</p>
          </div>

          <div className="form-grid">
            {/* Age */}
            <div className="form-group">
              <label htmlFor="age">
                <span className="label-icon">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2M12 11a4 4 0 100-8 4 4 0 000 8z"/></svg>
                </span>
                Age
              </label>
              <input 
                type="number" id="age" name="age"
                value={formData.age} onChange={handleChange}
                placeholder="e.g., 25" min="18" max="100" required
              />
            </div>

            {/* Annual Income */}
            <div className="form-group">
              <label htmlFor="income">
                <span className="label-icon">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 1v22M17 5H9.5a3.5 3.5 0 000 7h5a3.5 3.5 0 010 7H6"/></svg>
                </span>
                Annual Income ($)
              </label>
              <input 
                type="number" id="income" name="income"
                value={formData.income} onChange={handleChange}
                placeholder="e.g., 60000" min="1" required
              />
            </div>

            {/* Home Ownership */}
            <div className="form-group">
              <label htmlFor="home">
                <span className="label-icon">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M3 9l9-7 9 7v11a2 2 0 01-2 2H5a2 2 0 01-2-2V9z"/><path d="M9 22V12h6v10"/></svg>
                </span>
                Home Ownership
              </label>
              <select id="home" name="home" value={formData.home} onChange={handleChange} required>
                <option value="RENT">Rent</option>
                <option value="OWN">Own</option>
                <option value="MORTGAGE">Mortgage</option>
                <option value="OTHER">Other</option>
              </select>
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
                placeholder="e.g., 5" min="0" step="0.1" required
              />
            </div>

            {/* Loan Intent */}
            <div className="form-group">
              <label htmlFor="intent">
                <span className="label-icon">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 16V8a2 2 0 00-1-1.73l-7-4a2 2 0 00-2 0l-7 4A2 2 0 003 8v8a2 2 0 001 1.73l7 4a2 2 0 002 0l7-4A2 2 0 0021 16z"/></svg>
                </span>
                Loan Purpose
              </label>
              <select id="intent" name="intent" value={formData.intent} onChange={handleChange} required>
                <option value="PERSONAL">Personal</option>
                <option value="EDUCATION">Education</option>
                <option value="MEDICAL">Medical</option>
                <option value="VENTURE">Business Venture</option>
                <option value="HOMEIMPROVEMENT">Home Improvement</option>
                <option value="DEBTCONSOLIDATION">Debt Consolidation</option>
              </select>
            </div>

            {/* Loan Amount */}
            <div className="form-group">
              <label htmlFor="amount">
                <span className="label-icon">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 4H3a2 2 0 00-2 2v12a2 2 0 002 2h18a2 2 0 002-2V6a2 2 0 00-2-2z"/><circle cx="12" cy="12" r="3"/></svg>
                </span>
                Loan Amount ($)
              </label>
              <input 
                type="number" id="amount" name="amount"
                value={formData.amount} onChange={handleChange}
                placeholder="e.g., 10000" min="1" required
              />
            </div>

            {/* Interest Rate */}
            <div className="form-group">
              <label htmlFor="rate">
                <span className="label-icon">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="19" y1="5" x2="5" y2="19"/><circle cx="6.5" cy="6.5" r="2.5"/><circle cx="17.5" cy="17.5" r="2.5"/></svg>
                </span>
                Interest Rate (%)
              </label>
              <input 
                type="number" id="rate" name="rate"
                value={formData.rate} onChange={handleChange}
                placeholder="e.g., 11.5" min="0.01" max="30" step="0.01" required
              />
            </div>

            {/* Debt-to-Income Ratio */}
            <div className="form-group">
              <label htmlFor="percent_income">
                <span className="label-icon">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>
                </span>
                Debt-to-Income Ratio
              </label>
              <input 
                type="number" id="percent_income" name="percent_income"
                value={formData.percent_income} onChange={handleChange}
                placeholder="Auto-calculated" min="0" max="1" step="0.01" required
              />
              <span className="form-hint">Auto-calculated from Amount/Income</span>
            </div>

            {/* Previous Default */}
            <div className="form-group">
              <label htmlFor="default">
                <span className="label-icon">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 9v2m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/></svg>
                </span>
                Previous Default?
              </label>
              <select id="default" name="default" value={formData.default} onChange={handleChange} required>
                <option value="N">No</option>
                <option value="Y">Yes</option>
              </select>
            </div>

            {/* Credit History Length */}
            <div className="form-group">
              <label htmlFor="cred_length">
                <span className="label-icon">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
                </span>
                Credit History (years)
              </label>
              <input 
                type="number" id="cred_length" name="cred_length"
                value={formData.cred_length} onChange={handleChange}
                placeholder="e.g., 4" min="0" required
              />
            </div>
          </div>

          {/* Error Message */}
          {error && (
            <div className="form-error animate-fade-in" id="form-error">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>
              {error}
            </div>
          )}

          {/* Action Buttons */}
          <div className="form-actions">
            <button type="submit" className="btn btn-primary" id="submit-btn" disabled={loading}>
              {loading ? (
                <><span className="spinner"></span> Analyzing...</>
              ) : (
                <>
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"/></svg>
                  Get Prediction
                </>
              )}
            </button>
            <button type="button" className="btn btn-secondary" onClick={resetForm} id="reset-btn">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M2.5 2v6h6M21.5 22v-6h-6"/><path d="M22 11.5A10 10 0 003.2 7.2M2 12.5a10 10 0 0018.8 4.2"/></svg>
              Reset
            </button>
          </div>
        </form>

        {/* Result Section */}
        {result && (
          <div className="result-section animate-fade-in-up" id="prediction-result">
            <div className={`result-card ${result.prediction_code === 1 ? 'approved' : 'rejected'}`}>
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

              {/* Key Factors */}
              <div className="factors-section">
                <h3>Key Factors</h3>
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
