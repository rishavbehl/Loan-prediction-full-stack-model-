import { useState, useEffect } from 'react';
import { getModelInfo } from '../api/api';
import './ModelInfo.css';

// Subcomponents
import MetricsCards from '../components/model/MetricsCards';
import ClassificationReport from '../components/model/ClassificationReport';
import ModelConfig from '../components/model/ModelConfig';
import FeatureImportanceChart from '../components/model/FeatureImportanceChart';

/**
 * ModelInfo Page (v2.0 Modular & Creative)
 * =====================
 * Displays technical details of the ML model with Chart.js visualizations.
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
        <p>Loading AI model architecture & telemetry...</p>
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
      {/* Hero Section */}
      <div className="info-hero animate-fade-in-up">
        <div className="hero-badge">Telemetry & Diagnostics</div>
        <h1 className="hero-title">
          Model <span className="gradient-text">Intelligence</span>
        </h1>
        <p className="hero-subtitle">
          Interactive insights, performance radar, and permutation importance for the {modelInfo.model_type} engine.
        </p>
      </div>

      <div className="info-content-grid">
        {/* Left Column: Top Metrics & Config */}
        <div className="info-column animate-fade-in-up stagger-1">
          <MetricsCards modelInfo={modelInfo} />
          <ModelConfig modelInfo={modelInfo} />
          <div className="info-card">
             <ClassificationReport classificationReport={modelInfo.classification_report} />
          </div>
        </div>

        {/* Right Column: Dynamic Feature Importance Chart */}
        <div className="info-column animate-fade-in-up stagger-2">
          <FeatureImportanceChart 
            featureImportance={modelInfo.feature_importance} 
            featureColumns={modelInfo.feature_columns}
            featureDisplayNames={modelInfo.feature_display_names}
          />
          
          <div className="info-card fairness-card">
             <h2 className="card-title">Fairness & Bias Audit</h2>
             {modelInfo.fairness_analysis ? (
               <div className="fairness-metrics">
                 <div className="fairness-grade">
                    Grade: <span className={modelInfo.fairness_grade === 'PASS' ? 'text-success' : 'text-warning'}>{modelInfo.fairness_grade}</span>
                 </div>
                 {modelInfo.fairness_analysis.map((band, idx) => (
                    <div className="fairness-band" key={idx}>
                      <span className="band-name">{band.income_band}</span>
                      <span className="band-stat">Approval Rate: {band.approval_rate_pct}%</span>
                    </div>
                 ))}
               </div>
             ) : (
               <p className="text-muted">No fairness audit data available.</p>
             )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default ModelInfo;
