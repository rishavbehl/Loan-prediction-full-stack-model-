import { useState, useEffect } from 'react';
import { getStats, getModelInfo } from '../api/api';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  ArcElement,
  PointElement,
  LineElement,
  RadialLinearScale,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';
import { Pie, Bar, Doughnut, Radar } from 'react-chartjs-2';
import './Dashboard.css';

// Register Chart.js components
ChartJS.register(
  CategoryScale, LinearScale, BarElement, ArcElement,
  PointElement, LineElement, RadialLinearScale,
  Title, Tooltip, Legend, Filler
);

// Chart.js default styling for dark theme
ChartJS.defaults.color = '#94a3b8';
ChartJS.defaults.borderColor = 'rgba(255, 255, 255, 0.06)';
ChartJS.defaults.font.family = "'Inter', sans-serif";

/**
 * Dashboard Page (v2.0)
 * =====================
 * Updated to display CIBIL score distribution and Indian Banking metrics
 * based on the new HistGradientBoosting model metadata.
 */
function Dashboard() {
  const [stats, setStats] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      setLoading(true);
      const [statsData, modelData] = await Promise.all([
        getStats(),
        getModelInfo(),
      ]);
      setStats(statsData);
      setModelInfo(modelData);
    } catch (err) {
      setError('Failed to load dashboard data. Make sure the backend is running and model is trained!');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="dashboard-loading">
        <div className="loading-spinner"></div>
        <p>Loading banking analytics...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="dashboard-error">
        <div className="error-icon">!</div>
        <h2>Connection Error</h2>
        <p>{error}</p>
        <button onClick={loadData} className="btn btn-primary">Retry</button>
      </div>
    );
  }

  // --- Chart Data Preparation ---

  // 1. Approval Rate Doughnut
  const approvalChartData = {
    labels: ['Approved', 'Rejected'],
    datasets: [{
      data: [stats.approved_count, stats.rejected_count],
      backgroundColor: ['rgba(34, 197, 94, 0.8)', 'rgba(239, 68, 68, 0.8)'],
      borderColor: ['rgba(34, 197, 94, 1)', 'rgba(239, 68, 68, 1)'],
      borderWidth: 2,
      hoverOffset: 8,
    }]
  };

  // 2. CIBIL Score Distribution Bar Chart
  const cibilLabels = Object.keys(stats.cibil_distribution || {});
  const cibilValues = Object.values(stats.cibil_distribution || {});
  
  const cibilChartData = {
    labels: cibilLabels,
    datasets: [{
      label: 'Number of Applications',
      data: cibilValues,
      backgroundColor: 'rgba(59, 130, 246, 0.6)',
      borderColor: 'rgba(59, 130, 246, 1)',
      borderWidth: 1,
      borderRadius: 6,
      borderSkipped: false,
    }]
  };

  // 3. Home Ownership Pie Chart
  const homeLabels = Object.keys(stats.home_distribution || {});
  const homeValues = Object.values(stats.home_distribution || {});
  const homeColors = [
    'rgba(99, 102, 241, 0.8)',
    'rgba(168, 85, 247, 0.8)',
    'rgba(236, 72, 153, 0.8)',
    'rgba(251, 146, 60, 0.8)',
  ];

  const homeChartData = {
    labels: homeLabels.map(l => l.charAt(0) + l.slice(1).toLowerCase()),
    datasets: [{
      data: homeValues,
      backgroundColor: homeColors,
      borderColor: homeColors.map(c => c.replace('0.8', '1')),
      borderWidth: 2,
      hoverOffset: 8,
    }]
  };

  // 4. Approval Rate by CIBIL Band Bar Chart
  const approvalByCibilData = {
    labels: Object.keys(stats.approval_by_cibil || {}),
    datasets: [{
      label: 'Approval Rate (%)',
      data: Object.values(stats.approval_by_cibil || {}),
      backgroundColor: 'rgba(34, 197, 94, 0.6)',
      borderColor: 'rgba(34, 197, 94, 1)',
      borderWidth: 1,
      borderRadius: 6,
      borderSkipped: false,
    }]
  };

  // 5. Feature Importance Radar
  const featureLabels = Object.keys(modelInfo?.feature_importance || {}).slice(0, 8); // Top 8 features
  const featureValues = Object.values(modelInfo?.feature_importance || {}).slice(0, 8).map(v => v * 100);

  // Map API keys to Display names for radar chart
  const radarLabels = featureLabels.map(featKey => {
    const idx = modelInfo?.feature_columns?.indexOf(featKey);
    return idx >= 0 && modelInfo?.feature_display_names ? modelInfo.feature_display_names[idx] : featKey;
  });

  const featureRadarData = {
    labels: radarLabels,
    datasets: [{
      label: 'Importance (%)',
      data: featureValues,
      backgroundColor: 'rgba(99, 102, 241, 0.15)',
      borderColor: 'rgba(99, 102, 241, 0.8)',
      borderWidth: 2,
      pointBackgroundColor: 'rgba(99, 102, 241, 1)',
      pointBorderColor: '#fff',
      pointBorderWidth: 1,
      pointRadius: 4,
    }]
  };

  // 6. Loan Purpose Distribution Bar
  const purposeLabels = Object.keys(stats.purpose_distribution || {}).slice(0, 8); // Top 8 purposes
  const purposeValues = Object.values(stats.purpose_distribution || {}).slice(0, 8);

  const purposeDistData = {
    labels: purposeLabels.map(l => l.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())),
    datasets: [{
      label: 'Applications',
      data: purposeValues,
      backgroundColor: 'rgba(139, 92, 246, 0.6)',
      borderColor: 'rgba(139, 92, 246, 1)',
      borderWidth: 1,
      borderRadius: 6,
      borderSkipped: false,
    }]
  };

  // --- Chart Options ---
  const doughnutOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { position: 'bottom', labels: { padding: 20, usePointStyle: true, pointStyle: 'circle' } },
    },
    cutout: '65%',
  };

  const barOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
    },
    scales: {
      y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.04)' } },
      x: { grid: { display: false } },
    },
  };

  const radarOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
    },
    scales: {
      r: {
        beginAtZero: true,
        grid: { color: 'rgba(255,255,255,0.06)' },
        angleLines: { color: 'rgba(255,255,255,0.06)' },
        pointLabels: { font: { size: 10, color: 'rgba(255,255,255,0.7)' } },
      },
    },
  };

  return (
    <div className="dashboard-page">
      {/* Hero */}
      <div className="dashboard-hero animate-fade-in-up">
        <div className="hero-badge">Banking Analytics</div>
        <h1 className="hero-title">
          Dataset <span className="gradient-text">Insights</span>
        </h1>
        <p className="hero-subtitle">
          Comprehensive analytics from {stats.total_records?.toLocaleString()} loan applications — 
          powered by our {modelInfo?.accuracy_percent}% accurate ML model.
        </p>
      </div>

      {/* Stat Cards */}
      <div className="stat-cards animate-fade-in-up stagger-1">
        <div className="stat-card" id="stat-total">
          <div className="stat-icon total">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M17 21v-2a4 4 0 00-4-4H5a4 4 0 00-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 00-3-3.87"/><path d="M16 3.13a4 4 0 010 7.75"/></svg>
          </div>
          <div className="stat-info">
            <span className="stat-value">{stats.total_records?.toLocaleString()}</span>
            <span className="stat-label">Total Applications</span>
          </div>
        </div>

        <div className="stat-card" id="stat-approved">
          <div className="stat-icon approved">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
          </div>
          <div className="stat-info">
            <span className="stat-value">{stats.approved_count?.toLocaleString()}</span>
            <span className="stat-label">Approved ({stats.approval_rate}%)</span>
          </div>
        </div>

        <div className="stat-card" id="stat-rejected">
          <div className="stat-icon rejected">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>
          </div>
          <div className="stat-info">
            <span className="stat-value">{stats.rejected_count?.toLocaleString()}</span>
            <span className="stat-label">Rejected ({(100 - stats.approval_rate).toFixed(1)}%)</span>
          </div>
        </div>

        <div className="stat-card" id="stat-accuracy">
          <div className="stat-icon accuracy">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"/></svg>
          </div>
          <div className="stat-info">
            <span className="stat-value">{modelInfo?.accuracy_percent}%</span>
            <span className="stat-label">Model Accuracy</span>
          </div>
        </div>
      </div>

      {/* Financial Stats */}
      <div className="financial-stats animate-fade-in-up stagger-2">
        <div className="fin-card">
          <span className="fin-label">Avg CIBIL Score</span>
          <span className="fin-value">{stats.cibil_stats?.mean}</span>
        </div>
        <div className="fin-card">
          <span className="fin-label">Avg Income</span>
          <span className="fin-value">₹{stats.income_stats?.mean?.toLocaleString()}</span>
        </div>
        <div className="fin-card">
          <span className="fin-label">Avg Loan Amount</span>
          <span className="fin-value">₹{stats.amount_stats?.mean?.toLocaleString()}</span>
        </div>
        <div className="fin-card">
          <span className="fin-label">Avg Interest Rate</span>
          <span className="fin-value">{stats.rate_stats?.mean}%</span>
        </div>
      </div>

      {/* Charts Grid */}
      <div className="charts-grid">
        {/* Approval Distribution */}
        <div className="chart-card animate-fade-in-up stagger-2" id="chart-approval">
          <h3 className="chart-title">Approval Distribution</h3>
          <p className="chart-subtitle">Overall loan approval vs rejection rate</p>
          <div className="chart-container chart-small">
            <Doughnut data={approvalChartData} options={doughnutOptions} />
          </div>
        </div>

        {/* CIBIL Distribution */}
        <div className="chart-card animate-fade-in-up stagger-3" id="chart-cibil">
          <h3 className="chart-title">CIBIL Score Distribution</h3>
          <p className="chart-subtitle">Number of applicants in each CIBIL band</p>
          <div className="chart-container">
            <Bar data={cibilChartData} options={{
              ...barOptions,
              scales: {
                x: { grid: { display: false }, ticks: { maxRotation: 45, minRotation: 45 } },
                y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.04)' } }
              }
            }} />
          </div>
        </div>

        {/* Home Ownership */}
        <div className="chart-card animate-fade-in-up stagger-3" id="chart-home">
          <h3 className="chart-title">Home Ownership</h3>
          <p className="chart-subtitle">Distribution of applicant home ownership</p>
          <div className="chart-container chart-small">
            <Pie data={homeChartData} options={doughnutOptions} />
          </div>
        </div>

        {/* Approval by CIBIL */}
        <div className="chart-card animate-fade-in-up stagger-4" id="chart-approval-cibil">
          <h3 className="chart-title">Approval Rate by CIBIL Score</h3>
          <p className="chart-subtitle">How CIBIL score affects approval chances</p>
          <div className="chart-container">
            <Bar data={approvalByCibilData} options={{
              ...barOptions,
              scales: {
                x: { grid: { display: false }, ticks: { maxRotation: 45, minRotation: 45 } },
                y: { max: 100, beginAtZero: true, grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { callback: v => v + '%' } }
              }
            }} />
          </div>
        </div>

        {/* Purpose Distribution */}
        <div className="chart-card animate-fade-in-up stagger-4" id="chart-purpose">
          <h3 className="chart-title">Top Loan Purposes</h3>
          <p className="chart-subtitle">Most common reasons for taking a loan</p>
          <div className="chart-container">
            <Bar data={purposeDistData} options={{
              ...barOptions,
              indexAxis: 'y',
              scales: {
                y: { grid: { display: false } },
                x: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.04)' } }
              }
            }} />
          </div>
        </div>

        {/* Feature Importance Radar */}
        <div className="chart-card animate-fade-in-up stagger-5" id="chart-features">
          <h3 className="chart-title">Top Feature Importance</h3>
          <p className="chart-subtitle">What matters most for loan approval decisions</p>
          <div className="chart-container chart-small">
            <Radar data={featureRadarData} options={radarOptions} />
          </div>
        </div>
      </div>
    </div>
  );
}

export default Dashboard;
