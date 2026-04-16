import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import PredictionForm from './pages/PredictionForm';
import Dashboard from './pages/Dashboard';
import ModelInfo from './pages/ModelInfo';
import './App.css';

/**
 * App Component
 * =============
 * Root component with routing between:
 * - / (Prediction Form)
 * - /dashboard (Analytics Dashboard)
 * - /model (Model Performance Info)
 */
function App() {
  return (
    <Router>
      <div className="app">
        <Navbar />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<PredictionForm />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/model" element={<ModelInfo />} />
          </Routes>
        </main>
        <footer className="app-footer">
          <div className="footer-content">
            <p>
              <span className="footer-brand">LoanPredict<span className="footer-accent">AI</span></span>
              &nbsp;&mdash;&nbsp;Full Stack ML Project
            </p>
            <p className="footer-tech">
              Built with React + FastAPI + Scikit-Learn
            </p>
          </div>
        </footer>
      </div>
    </Router>
  );
}

export default App;
