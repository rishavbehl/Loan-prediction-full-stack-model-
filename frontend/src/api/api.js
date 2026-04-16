/**
 * API Service - Connects React frontend to FastAPI backend
 * =========================================================
 * Base URL: http://localhost:8000
 */

const API_BASE_URL = 'http://localhost:8000';

/**
 * Submit a loan application for prediction
 * @param {Object} applicationData - Loan application form data
 * @returns {Object} Prediction result with confidence and risk level
 */
export async function predictLoan(applicationData) {
  try {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(applicationData),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Prediction failed');
    }

    return await response.json();
  } catch (error) {
    console.error('Prediction API Error:', error);
    throw error;
  }
}

/**
 * Get dataset statistics for the dashboard
 * @returns {Object} Dataset statistics (distributions, rates, etc.)
 */
export async function getStats() {
  try {
    const response = await fetch(`${API_BASE_URL}/stats`);
    if (!response.ok) throw new Error('Failed to fetch stats');
    return await response.json();
  } catch (error) {
    console.error('Stats API Error:', error);
    throw error;
  }
}

/**
 * Get model performance info and feature importance
 * @returns {Object} Model metadata (accuracy, features, etc.)
 */
export async function getModelInfo() {
  try {
    const response = await fetch(`${API_BASE_URL}/model-info`);
    if (!response.ok) throw new Error('Failed to fetch model info');
    return await response.json();
  } catch (error) {
    console.error('Model Info API Error:', error);
    throw error;
  }
}

/**
 * Health check
 * @returns {Object} API health status
 */
export async function healthCheck() {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    if (!response.ok) throw new Error('API is unhealthy');
    return await response.json();
  } catch (error) {
    console.error('Health Check Error:', error);
    throw error;
  }
}
