# Loan Approval Prediction System - Full Stack ML Project

A full-stack machine learning project that predicts loan approval using a trained Random Forest model, FastAPI backend, and React frontend.

## Project Structure

```
loan-prediction/
├── credit_risk.csv              # Dataset (32,581 records)
├── backend/                     # Python + FastAPI
│   ├── main.py                  # FastAPI API server
│   ├── train_model.py           # ML model training script
│   ├── model.pkl                # Trained model (generated)
│   ├── model_metadata.json      # Model stats & encodings (generated)
│   └── requirements.txt         # Python dependencies
└── frontend/                    # React + Vite
    ├── index.html               # HTML entry point
    ├── package.json             # Node.js dependencies
    ├── vite.config.js           # Vite configuration
    └── src/
        ├── main.jsx             # React entry point
        ├── App.jsx              # Root component with routing
        ├── App.css              # App-level styles
        ├── index.css            # Global design system
        ├── api/
        │   └── api.js           # API service (connects to FastAPI)
        ├── components/
        │   ├── Navbar.jsx       # Navigation bar
        │   └── Navbar.css
        └── pages/
            ├── PredictionForm.jsx   # Loan application form
            ├── PredictionForm.css
            ├── Dashboard.jsx        # Analytics dashboard
            ├── Dashboard.css
            ├── ModelInfo.jsx        # Model performance page
            └── ModelInfo.css
```

## Tech Stack

| Layer      | Technology        |
|------------|-------------------|
| ML Model   | Scikit-Learn (Random Forest) |
| Backend    | Python, FastAPI, Uvicorn |
| Frontend   | HTML, CSS, JavaScript, React, Vite, Chart.js |
| Styling    | CSS (Dark Theme) |

## How to Run

### Step 1: Train the Model (already done)
```bash
cd backend
pip install -r requirements.txt
python train_model.py
```

### Step 2: Start the Backend
```bash
cd backend
python main.py
```
Backend runs at: **http://localhost:8000**
API Docs at: **http://localhost:8000/docs**

### Step 3: Start the Frontend
```bash
cd frontend
npm install
npm run dev
```
Frontend runs at: **http://localhost:5173**

## API Endpoints

| Method | Endpoint      | Description                          |
|--------|---------------|--------------------------------------|
| POST   | `/predict`    | Submit loan application for prediction |
| GET    | `/stats`      | Get dataset statistics for dashboard |
| GET    | `/model-info` | Get model performance & feature importance |
| GET    | `/health`     | Health check                         |

## Model Performance

- **Accuracy**: 91.3%
- **Algorithm**: Random Forest (100 trees)
- **Top Features**: Percent Income (27.2%), Interest Rate (20%), Income (15.5%)

## Dataset

- **Source**: credit_risk.csv
- **Records**: 32,581
- **Features**: Age, Income, Home, Employment Length, Intent, Amount, Rate, Percent Income, Default History, Credit Length
- **Target**: Status (0 = Rejected, 1 = Approved)
