import { useState, useEffect } from 'react';
import { getModelInfo } from '../api/api';
import './ModelInfo.css';

/**
 * ModelInfo Page (v2.0)
 * =====================
 * Displays technical details of the HistGradientBoosting model.
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
      setError('Failed to load model information. Ensure the backend is running and the model is trained.');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="model-info-loading">
        <div className="loading-spinner"></div>
        <p>Loading AI model architecture...</p>
      </div>
    );
  }

  if (error || !modelInfo) {
    return (
      <div className="model-info-error">
        <div className="error-icon">!</div>
        <h2>Connection Error</h2>
        <p>{error}</p>
        <button onClick={loadModelInfo} className="btn btn-primary">Retry</button>
      </div>
    );
  }

  return (
    <div className="model-info-page">
      {/* Hero */}
      <div className="info-hero animate-fade-in-up">
        <div className="hero-badge">Model Architecture</div>
        <h1 className="hero-title">
          AI Engine <span className="gradient-text">Internals</span>
        </h1>
        <p className="hero-subtitle">
          Explore the technical specifications and evaluation metrics of the 
          {modelInfo.model_type} powering our loan decisions.
        </p>
      </div>

      <div className="info-content">
        {/* Left Column: Metrics & Config */}
        <div className="info-column animate-fade-in-up stagger-1">
          {/* Performance Metrics */}
          <div className="info-card">
            <h2 className="card-title">Performance Metrics</h2>
            <div className="metrics-grid">
              <div className="metric-item highlight">
                <span className="metric-label">Test Accuracy</span>
                <span className="metric-value">{modelInfo.accuracy_percent}%</span>
              </div>
              <div className="metric-item">
                <span className="metric-label">Train Accuracy</span>
                <span className="metric-value">{(modelInfo.train_accuracy * 100).toFixed(1)}%</span>
              </div>
              <div className="metric-item">
                <span className="metric-label">Overfit Gap</span>
                <span className={`metric-value ${modelInfo.overfit_gap < 0.02 ? 'good' : 'warning'}`}>
                  {(modelInfo.overfit_gap * 100).toFixed(2)}%
                </span>
              </div>
              <div className="metric-item">
                <span className="metric-label">AUC-ROC Score</span>
                <span className="metric-value">{modelInfo.auc_roc}</span>
              </div>
              <div className="metric-item">
                <span className="metric-label">F1 Score</span>
                <span className="metric-value">{modelInfo.f1_score}</span>
              </div>
              <div className="metric-item">
                <span className="metric-label">Cross-Validation</span>
                <span className="metric-value">{(modelInfo.cv_mean_accuracy * 100).toFixed(1)}% ±{(modelInfo.cv_std_accuracy * 100).toFixed(2)}%</span>
              </div>
            </div>
            
            {/* Precision / Recall Table */}
            {modelInfo.classification_report && (
              <div className="classification-report">
                <h3>Classification Details</h3>
                <div className="report-table-wrapper">
                  <table className="report-table">
                    <thead>
                      <tr>
                        <th>Class</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1-Score</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td><span className="class-badge approved">Approved (1)</span></td>
                        <td>{modelInfo.classification_report.precision_approved}</td>
                        <td>{modelInfo.classification_report.recall_approved}</td>
                        <td>{modelInfo.classification_report.f1_approved}</td>
                      </tr>
                      <tr>
                        <td><span className="class-badge rejected">Rejected (0)</span></td>
                        <td>{modelInfo.classification_report.precision_rejected}</td>
                        <td>{modelInfo.classification_report.recall_rejected}</td>
                        <td>{modelInfo.classification_report.f1_rejected}</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>

          {/* Configuration */}
          <div className="info-card">
            <h2 className="card-title">Model Configuration</h2>
            <div className="config-list">
              <div className="config-item">
                <span className="config-label">Algorithm</span>
                <span className="config-value badge">{modelInfo.model_type}</span>
              </div>
              <div className="config-item">
                <span className="config-label">Iterations (Trees)</span>
                <span className="config-value">{modelInfo.n_estimators_actual} (Early Stopped)</span>
              </div>
              <div className="config-item">
                <span className="config-label">Learning Rate</span>
                <span className="config-value">{modelInfo.learning_rate}</span>
              </div>
              <div className="config-item">
                <span className="config-label">Max Depth</span>
                <span className="config-value">{modelInfo.max_depth}</span>
              </div>
              <div className="config-item">
                <span className="config-label">Total Features</span>
                <span className="config-value">{modelInfo.feature_columns?.length || 17}</span>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Feature Importance */}
        <div className="info-column animate-fade-in-up stagger-2">
          <div className="info-card feature-importance-card">
            <div className="card-header-flex">
              <h2 className="card-title">Permutation Feature Importance</h2>
              <span className="feature-count">{modelInfo.feature_columns?.length} Features</span>
            </div>
            <p className="card-subtitle">
              Relative contribution of each feature to the model's predictions (Permutation Importance).
            </p>
            
            <div className="features-list">
              {Object.entries(modelInfo.feature_importance || {}).map(([key, value], index) => {
                // Find display name
                const idx = modelInfo.feature_columns?.indexOf(key);
                const displayName = idx >= 0 && modelInfo.feature_display_names 
                  ? modelInfo.feature_display_names[idx] 
                  : key;
                
                // Calculate percentage based on max value for the bar width
                const maxValue = Math.max(...Object.values(modelInfo.feature_importance || {}));
                const percentage = maxValue > 0 ? (value / maxValue) * 100 : 0;
                
                return (
                  <div className="feature-item" key={key}>
                    <div className="feature-info">
                      <span className="feature-rank">#{index + 1}</span>
                      <span className="feature-name">{displayName}</span>
                      <span className="feature-score">{(value * 100).toFixed(2)}%</span>
                    </div>
                    <div className="feature-bar-wrapper">
                      <div 
                        className="feature-bar" 
                        style={{ width: `${Math.max(percentage, 1)}%` }}
                      ></div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ModelInfo;
