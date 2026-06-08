<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=2&height=250&section=header&text=Loan%20Approval%20Prediction&fontSize=60&animation=fadeIn&fontAlignY=38&desc=Full-Stack%20Machine%20Learning%20System&descAlignY=55&descAlign=62" />

  <!-- Badges -->
  <p>
    <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange?style=for-the-badge&logo=scikit-learn" alt="Scikit-Learn" />
    <img src="https://img.shields.io/badge/Backend-FastAPI-009688?style=for-the-badge&logo=fastapi" alt="FastAPI" />
    <img src="https://img.shields.io/badge/Frontend-React%20%2B%20Vite-61DAFB?style=for-the-badge&logo=react" alt="React" />
    <img src="https://img.shields.io/badge/Status-Active-success?style=for-the-badge" alt="Status" />
  </p>
</div>

Welcome to the **Loan Approval Prediction System**! This application leverages a highly tuned, **Banking-Grade HistGradientBoosting model** to evaluate loan applicants. Featuring a blazing-fast FastAPI backend and a sleek React frontend, it acts as a realistic banking portal—complete with detailed **CIBIL Score integration**, risk level analysis, and key financial factor breakdown.

---

## 🚀 Key Features
- **Intelligent Predictions**: Real-time loan approval prediction using a model trained on 300,000 real-world records.
- **Deep Financial Analysis**: Returns granular insights including Risk Level, CIBIL category, and specific driving factors (like Debt-to-Income and Loan-to-Income ratios).
- **Interactive Dashboard**: Modern, responsive React interface to submit applications and view rich, chart-based model analytics.
- **Developer-Friendly API**: Clean, well-documented RESTful endpoints ready for integration.

---

## 🛠️ Tech Stack

### ML & Backend
* **Algorithm**: `HistGradientBoostingClassifier` (Scikit-Learn)
* **API Framework**: FastAPI & Uvicorn
* **Data Processing**: Pandas & NumPy

### Frontend
* **Core**: React 19 & Vite
* **Styling**: Vanilla CSS (Premium Dark Theme)
* **Visualizations**: Chart.js

---

## 📂 Project Architecture

```bash
loan-prediction/
├── 📊 Datasets
│   ├── accepted_2007_to_2018Q4.csv  # Lending Club Accepted Data
│   └── rejected_2007_to_2018Q4.csv  # Lending Club Rejected Data
│
├── ⚙️ backend/                      # Python + FastAPI Engine
│   ├── main.py                  # Core API server (v2.0)
│   ├── train_model.py           # ML model pipeline
│   ├── diagnose_and_refine.py   # Model tuning & analysis
│   ├── test_scenarios.py        # Edge-case & real-world testing
│   ├── model.pkl                # Serialized predictive model
│   └── model_metadata.json      # Comprehensive model telemetry
│
└── 💻 frontend/                     # React + Vite UI
    ├── package.json
    └── src/
        ├── api/api.js           # API connection service
        ├── components/          # Reusable UI components
        └── pages/               # Application Views (Form, Dashboard, Model Info)
```

---

## 📈 Model Performance & Telemetry

Our model is rigorously tested and optimized for banking standards. 

- **Engine**: Banking-Grade HistGradientBoosting v5.0
- **Scale**: Trained on **300,000** records (Balanced 1:1 Approved/Rejected)
- **Features Analyzed**: 22 diverse financial data points
- **Accuracy**: **66.0%** *(CV Mean: 65.76%)*
- **AUC-ROC**: **0.722** *(Fair Grade)*
- **F1 Score**: **65.25%**

### 🏆 Top Decision Factors
1. **Credit Grade**: `3.8%` impact
2. **Loan Term**: `1.4%` impact
3. **CIBIL Score**: `0.6%` impact
4. **Interest Rate**: `0.6%` impact
5. **Loan-to-Income (LTI) & DTI**: `0.5%` impact

---

## 🌐 API Endpoints

The FastAPI backend automatically generates interactive OpenAPI documentation at `/docs`.

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | **`/predict`** | Submit an applicant profile. Returns probability, risk level, CIBIL grade, and a factor breakdown. |
| `GET` | **`/stats`** | Fetches global dataset statistics for the frontend dashboard. |
| `GET` | **`/model-info`** | Exposes deep model metadata and performance metrics. |
| `GET` | **`/health`** | Simple ping to check if the API and model are active. |

---

## ⚡ Getting Started

Ready to run the project locally? Follow these simple steps:

### 1️⃣ Prepare the Environment (Optional)
The model (`model.pkl`) is already trained. However, to retrain or test the model:
```bash
cd backend
pip install -r requirements.txt
python train_model.py
```

### 2️⃣ Ignite the Backend
```bash
cd backend
python main.py
```
> **Backend runs at:** `http://localhost:8000`  
> **Swagger API Docs:** `http://localhost:8000/docs`

### 3️⃣ Launch the Frontend
Open a new terminal window:
```bash
cd frontend
npm install
npm run dev
```
> **Frontend runs at:** `http://localhost:5173`

---
*Built with ❤️ for advanced machine learning analytics.*
