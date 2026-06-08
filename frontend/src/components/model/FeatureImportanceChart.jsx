import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Bar } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

function FeatureImportanceChart({ featureImportance, featureColumns, featureDisplayNames }) {
  if (!featureImportance) return null;

  // Sort features by importance
  const sortedFeatures = Object.entries(featureImportance)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 10); // Top 10 features

  const labels = sortedFeatures.map(([key]) => {
    const idx = featureColumns?.indexOf(key);
    return idx >= 0 && featureDisplayNames ? featureDisplayNames[idx] : key;
  });

  const dataValues = sortedFeatures.map(([, value]) => (value * 100).toFixed(2));

  const data = {
    labels,
    datasets: [
      {
        label: 'Importance (%)',
        data: dataValues,
        backgroundColor: 'rgba(99, 102, 241, 0.8)',
        borderColor: 'rgba(99, 102, 241, 1)',
        borderWidth: 1,
        borderRadius: 4,
      },
    ],
  };

  const options = {
    indexAxis: 'y',
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false,
      },
      tooltip: {
        backgroundColor: 'rgba(15, 23, 42, 0.9)',
        titleColor: '#fff',
        bodyColor: '#cbd5e1',
        borderColor: 'rgba(99, 102, 241, 0.3)',
        borderWidth: 1,
        padding: 12,
        displayColors: false,
      },
    },
    scales: {
      x: {
        grid: {
          color: 'rgba(255, 255, 255, 0.05)',
        },
        ticks: {
          color: '#94a3b8',
        },
      },
      y: {
        grid: {
          display: false,
        },
        ticks: {
          color: '#e2e8f0',
          font: {
            size: 13,
          },
        },
      },
    },
  };

  return (
    <div className="info-card feature-importance-card">
      <div className="card-header-flex">
        <h2 className="card-title">Top 10 Feature Drivers</h2>
        <span className="feature-count">{featureColumns?.length} Total Features</span>
      </div>
      <p className="card-subtitle">
        Visualizing the most influential factors deciding loan approvals (Permutation Importance).
      </p>
      <div style={{ height: '400px', width: '100%', marginTop: '20px' }}>
        <Bar data={data} options={options} />
      </div>
    </div>
  );
}

export default FeatureImportanceChart;
