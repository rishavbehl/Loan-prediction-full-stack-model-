import {
  Chart as ChartJS,
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend,
} from 'chart.js';
import { Radar } from 'react-chartjs-2';

ChartJS.register(
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend
);

function MetricsCards({ modelInfo }) {
  const radarData = {
    labels: ['Accuracy', 'Train Accuracy', 'AUC-ROC', 'F1 Score', 'Precision', 'Recall'],
    datasets: [
      {
        label: 'Model Performance Metrics',
        data: [
          modelInfo.accuracy || 0,
          modelInfo.train_accuracy || 0,
          modelInfo.auc_roc || 0,
          modelInfo.f1_score || 0,
          modelInfo.precision || 0,
          modelInfo.recall || 0,
        ],
        backgroundColor: 'rgba(56, 189, 248, 0.2)',
        borderColor: 'rgba(56, 189, 248, 1)',
        borderWidth: 2,
        pointBackgroundColor: 'rgba(56, 189, 248, 1)',
        pointBorderColor: '#fff',
        pointHoverBackgroundColor: '#fff',
        pointHoverBorderColor: 'rgba(56, 189, 248, 1)',
      },
    ],
  };

  const radarOptions = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      r: {
        angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
        grid: { color: 'rgba(255, 255, 255, 0.1)' },
        pointLabels: { color: '#e2e8f0', font: { size: 12 } },
        ticks: { display: false, min: 0, max: 1 },
      },
    },
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: 'rgba(15, 23, 42, 0.9)',
        titleColor: '#fff',
        bodyColor: '#cbd5e1',
        borderColor: 'rgba(56, 189, 248, 0.3)',
        borderWidth: 1,
      },
    },
  };

  return (
    <div className="info-card">
      <h2 className="card-title">Core Performance Metrics</h2>
      <div className="metrics-layout">
        <div className="radar-chart-container" style={{ height: '250px', marginBottom: '20px' }}>
          <Radar data={radarData} options={radarOptions} />
        </div>
        <div className="metrics-grid">
          <div className="metric-item highlight">
            <span className="metric-label">Test Accuracy</span>
            <span className="metric-value">{modelInfo.accuracy_percent}%</span>
          </div>
          <div className="metric-item highlight-purple">
            <span className="metric-label">AUC-ROC</span>
            <span className="metric-value">{modelInfo.auc_roc}</span>
          </div>
          <div className="metric-item">
            <span className="metric-label">Overfit Gap</span>
            <span className={`metric-value ${modelInfo.overfit_gap < 0.02 ? 'good' : 'warning'}`}>
              {(modelInfo.overfit_gap * 100).toFixed(2)}%
            </span>
          </div>
          <div className="metric-item">
            <span className="metric-label">Cross-Validation</span>
            <span className="metric-value">{(modelInfo.cv_mean_accuracy * 100).toFixed(1)}%</span>
          </div>
        </div>
      </div>
    </div>
  );
}

export default MetricsCards;
