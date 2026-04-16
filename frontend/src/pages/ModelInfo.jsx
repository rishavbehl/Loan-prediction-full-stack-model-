import { useState, useEffect } from 'react';
import { getModelInfo } from '../api/api';
import { Bar } from 'react-chartjs-2';
import './ModelInfo.css';

/**
 * ModelInfo Page
 * ==============
 * Displays model performance metrics, feature importance,
 * confusion matrix, and classification report.
 */
function ModelInfo() {
  const [modelInfo, setModelInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    loadModelInfo();
  }, []);

  const loadModelInfo = async () => {
    try {
      setLoading(true);
      const data = await getModelInfo();
      setModelInfo(data);
    } catch (err) {
      setError('Failed to load model info. Make sure the backend is running!');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="dashboard-loading">
        <div className="loading-spinner"></div>
        <p>Loading model information...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="dashboard-error">
        <div className="error-icon">!</div>
        <h2>Connection Error</h2>
        <p>{error}</p>
        <button onClick={loadModelInfo} className="btn btn-primary">Retry</button>
      </div>
    );
  }

  // Feature Importance Bar Chart
  const featureLabels = Object.keys(modelInfo?.feature_importance || {});
  const featureValues = Object.values(modelInfo?.feature_importance || {}).map(v => (v * 100).toFixed(1));

  const featureBarData = {
    labels: featureLabels.map(l => l.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())),
    datasets: [{
      label: 'Importance (%)',
      data: featureValues,
      backgroundColor: featureLabels.map((_, i) => {
        const colors = [
          'rgba(99, 102, 241, 0.7)',
          'rgba(139, 92, 246, 0.7)',
          'rgba(168, 85, 247, 0.7)',
          'rgba(59, 130, 246, 0.7)',
          'rgba(14, 165, 233, 0.7)',
          'rgba(20, 184, 166, 0.7)',
          'rgba(34, 197, 94, 0.7)',
          'rgba(251, 191, 36, 0.7)',
          'rgba(251, 146, 60, 0.7)',
          'rgba(236, 72, 153, 0.7)',
        ];
        return colors[i % colors.length];
      }),
      borderRadius: 8,
      borderSkipped: false,
    }]
  };

  const cm = modelInfo?.confusion_matrix || [[0,0],[0,0]];

  return (
    <div className="model-page">
      {/* Hero */}
      <div className="predict-hero animate-fade-in-up">
        <div className="hero-badge">Machine Learning</div>
        <h1 className="hero-title">
          Model <span className="gradient-text">Performance</span>
        </h1>
        <p className="hero-subtitle">
          Detailed metrics and analysis of the {modelInfo?.model_type} model 
          trained on {modelInfo?.feature_columns?.length} features.
        </p>
      </div>

      {/* Model Overview Cards */}
      <div className="model-overview animate-fade-in-up stagger-1">
        <div className="model-card highlight">
          <div className="model-card-icon">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
          </div>
          <div className="model-card-value">{modelInfo?.accuracy_percent}%</div>
          <div className="model-card-label">Overall Accuracy</div>
        </div>

        <div className="model-card">
          <div className="model-card-icon blue">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="9" y1="21" x2="9" y2="9"/></svg>
          </div>
          <div className="model-card-value">{modelInfo?.model_type?.replace('Classifier', '')}</div>
          <div className="model-card-label">Algorithm</div>
        </div>

        <div className="model-card">
          <div className="model-card-icon purple">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 20V10M18 20V4M6 20v-4"/></svg>
          </div>
          <div className="model-card-value">{modelInfo?.n_estimators}</div>
          <div className="model-card-label">Decision Trees</div>
        </div>

        <div className="model-card">
          <div className="model-card-icon orange">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-2 2 2 2 0 01-2-2v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83 0 2 2 0 010-2.83l.06-.06A1.65 1.65 0 004.68 15a1.65 1.65 0 00-1.51-1H3a2 2 0 01-2-2 2 2 0 012-2h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 010-2.83 2 2 0 012.83 0l.06.06A1.65 1.65 0 009 4.68a1.65 1.65 0 001-1.51V3a2 2 0 012-2 2 2 0 012 2v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 0 2 2 0 010 2.83l-.06.06A1.65 1.65 0 0019.4 9a1.65 1.65 0 001.51 1H21a2 2 0 012 2 2 2 0 01-2 2h-.09a1.65 1.65 0 00-1.51 1z"/></svg>
          </div>
          <div className="model-card-value">{modelInfo?.feature_columns?.length}</div>
          <div className="model-card-label">Features Used</div>
        </div>
      </div>

      {/* Classification Report */}
      <div className="report-grid animate-fade-in-up stagger-2">
        <div className="report-card">
          <h3 className="report-title">Classification Report</h3>
          <p className="report-subtitle">Precision & Recall for each class</p>
          
          <div className="report-table">
            <div className="report-header">
              <span>Class</span>
              <span>Precision</span>
              <span>Recall</span>
            </div>
            <div className="report-row rejected-row">
              <span className="class-label">
                <span className="class-dot rejected"></span>
                Rejected (0)
              </span>
              <span className="metric-value">{(modelInfo?.classification_report?.precision_rejected * 100).toFixed(1)}%</span>
              <span className="metric-value">{(modelInfo?.classification_report?.recall_rejected * 100).toFixed(1)}%</span>
            </div>
            <div className="report-row approved-row">
              <span className="class-label">
                <span className="class-dot approved"></span>
                Approved (1)
              </span>
              <span className="metric-value">{(modelInfo?.classification_report?.precision_approved * 100).toFixed(1)}%</span>
              <span className="metric-value">{(modelInfo?.classification_report?.recall_approved * 100).toFixed(1)}%</span>
            </div>
          </div>
        </div>

        {/* Confusion Matrix */}
        <div className="report-card">
          <h3 className="report-title">Confusion Matrix</h3>
          <p className="report-subtitle">Predicted vs Actual classifications</p>
          
          <div className="confusion-matrix">
            <div className="cm-labels-top">
              <span></span>
              <span>Pred: Rejected</span>
              <span>Pred: Approved</span>
            </div>
            <div className="cm-row">
              <span className="cm-label">Actual: Rejected</span>
              <span className="cm-cell cm-correct">{cm[0][0]}</span>
              <span className="cm-cell cm-wrong">{cm[0][1]}</span>
            </div>
            <div className="cm-row">
              <span className="cm-label">Actual: Approved</span>
              <span className="cm-cell cm-wrong">{cm[1][0]}</span>
              <span className="cm-cell cm-correct">{cm[1][1]}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Feature Importance Chart */}
      <div className="chart-card feature-chart animate-fade-in-up stagger-3">
        <h3 className="chart-title">Feature Importance Ranking</h3>
        <p className="chart-subtitle">Which factors matter most for loan approval decisions</p>
        <div className="chart-container feature-bar-chart">
          <Bar data={featureBarData} options={{
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
              x: { 
                beginAtZero: true, 
                grid: { color: 'rgba(255,255,255,0.04)' },
                ticks: { callback: v => v + '%' },
              },
              y: { grid: { display: false } },
            },
          }} />
        </div>
      </div>

      {/* Feature List */}
      <div className="features-used animate-fade-in-up stagger-4">
        <h3>Features Used for Prediction</h3>
        <div className="features-grid">
          {modelInfo?.feature_columns?.map((feature, index) => (
            <div className="feature-chip" key={feature}>
              <span className="feature-number">{index + 1}</span>
              <span>{feature.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}</span>
              <span className="feature-importance-badge">
                {((modelInfo?.feature_importance?.[feature] || 0) * 100).toFixed(1)}%
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default ModelInfo;
