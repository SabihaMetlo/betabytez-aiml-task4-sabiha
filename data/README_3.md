# House Price Prediction — BetaBytez AI/ML Internship, Task 3

**Author:** Sabiha Metlo
**Track:** AI/ML — Full Stack AI Engineering

## Screenshot

![House Price Predictor](screenshots/app-demo.png.jpeg)


## Overview

An end-to-end machine learning system that predicts house sale prices. A trained
Gradient Boosting Regressor model is served through a FastAPI backend, and a React
frontend lets users enter house details and receive a real-time price prediction.

**Domain:** Real Estate
**Problem type:** Regression (predicting a continuous price, not a category)
**Dataset:** Ames Housing (Kaggle — House Prices: Advanced Regression Techniques),
1460 rows, 81 features

## Project Structure
betabytez-aiml-task3-sabiha/
├── model/ # Data cleaning, EDA, model training notebook + saved model
├── api/ # FastAPI backend serving the /predict endpoint
├── frontend/ # React frontend (form + prediction display)
└── README.md

## Approach

### Data Preparation
- Handled missing values using two different strategies depending on meaning:
  - Documented "no feature" columns (e.g., PoolQC, GarageType) — confirmed via
    `data_description.txt` that NaN represents a real category, not missing data.
    Filled with "None" (or 0 for numeric equivalents like GarageYrBlt).
  - Genuinely missing values (LotFrontage, Electrical) — filled using median/mode.
- Removed 2 documented outliers (GrLivArea > 4000 sq ft, SalePrice < $300,000).
- Applied log transformation (`np.log1p`) to SalePrice to correct right-skew,
  training models on the transformed target and converting predictions back
  with `np.expm1`.
- Encoded categorical features: ordinal encoding for quality/ranking columns
  (e.g., KitchenQual), one-hot encoding for nominal columns (e.g., Neighborhood).
- Final dataset: 1458 rows × 207 columns, fully numeric.

### Model Comparison

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | $15,557 | $22,321 | 0.9098 |
| Random Forest | $16,340 | $23,900 | 0.8966 |
| **Gradient Boosting (selected)** | **$15,028** | **$21,173** | **0.9188** |

Gradient Boosting was selected as the final model — lowest error, highest R² across
all three metrics. Notably, Random Forest underperformed simple Linear Regression,
suggesting the underlying relationships in this data are largely linear after
preprocessing; Gradient Boosting's sequential error-correction still improved on
both, capturing residual patterns the others missed.

## API Documentation

### Base URL
http://127.0.0.1:8000

### `POST /predict`

Accepts house features as JSON and returns a predicted sale price.

**Request body:**
```json
{
  "OverallQual": 7,
  "GrLivArea": 1800,
  "GarageCars": 2,
  "TotalBsmtSF": 900,
  "FullBath": 2,
  "YearBuilt": 2005,
  "Neighborhood": "CollgCr"
}
```

**Response (200 OK):**
```json
{
  "predicted_price": 182969.63,
  "currency": "USD"
}
```

**Response (422 Unprocessable Content)** — returned automatically when input data
is invalid (e.g., text sent for a numeric field), with a descriptive error message
instead of a server crash.

Interactive API docs (test endpoints directly in the browser) are available at:
http://127.0.0.1:8000/docs

## Setup Instructions

### 1. Backend (FastAPI)

```bash
cd api
python -m venv venv
venv\Scripts\activate          # Windows
pip install fastapi uvicorn scikit-learn joblib pandas
uvicorn main:app --reload
```
Backend runs at `http://127.0.0.1:8000`

### 2. Frontend (React)

```bash
cd frontend
npm install
npm run dev
```
Frontend runs at `http://localhost:5173`



**Note:** both the backend and frontend must be running simultaneously (in
separate terminals) for predictions to work.

## Tech Stack

- **Model:** Python, scikit-learn, pandas, joblib
- **Backend:** FastAPI, Pydantic, Uvicorn
- **Frontend:** React, Vite, Fetch API