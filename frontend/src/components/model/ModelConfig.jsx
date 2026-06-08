function ModelConfig({ modelInfo }) {
  return (
    <div className="info-card">
      <h2 className="card-title">Model Architecture & Config</h2>
      <div className="config-list grid-layout">
        <div className="config-item gradient-border">
          <span className="config-label">Algorithm</span>
          <span className="config-value badge-primary">{modelInfo.model_type}</span>
        </div>
        <div className="config-item">
          <span className="config-label">Trees / Estimators</span>
          <span className="config-value">{modelInfo.n_estimators_actual} <span className="text-muted">(Early Stopped)</span></span>
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
          <span className="config-label">Gini Coefficient</span>
          <span className="config-value">{modelInfo.gini_coefficient}</span>
        </div>
        <div className="config-item">
          <span className="config-label">KS Statistic</span>
          <span className="config-value">{modelInfo.ks_statistic}</span>
        </div>
      </div>
    </div>
  );
}

export default ModelConfig;
