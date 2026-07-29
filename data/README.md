# ❤️ Heart Disease Prediction App
**BetaBytez Summer Internship 2026 — AI/ML Track — Task 1**  
**Author:** Sabiha Metlo

---

## 📌 Project Overview
A machine learning web application that predicts whether a 
patient has heart disease based on 13 clinical features.
Built using the Cleveland Heart Disease dataset (UCI/Kaggle).

---

## 📊 Dataset
- **Source:** Heart Disease UCI Dataset (Kaggle)
- **Size:** 303 patients, 13 features + 1 target
- **Target:** 1 = Has Disease, 0 = No Disease
- **Class Balance:** 165 disease (54.5%) vs 138 no disease (45.5%)

---

## 🤖 Models Trained & Compared

| Metric    | Logistic Regression | Random Forest |
|-----------|--------------------:|-------------:|
| Accuracy  | 85.25%             | 83.61%       |
| Precision | 87.10%             | 84.38%       |
| Recall    | 84.38%             | 84.38%       |
| F1-Score  | 85.71%             | 84.38%       |

**Winner: Logistic Regression** — outperformed Random Forest 
on 3 out of 4 metrics on this small, clean dataset.

---

## 🏗️ Project Structure
betabytez-aiml-task1-sabiha/  
├── heart_disease_model.ipynb  
├── heart.csv                  
├── app.py                     
├── requirements.txt          
└── model/   
    ├── heart_disease_model.pkl     
    └── scaler.pkl             

---

## 🚀 How to Run the App

**Step 1 — Install required libraries:**
pip install -r requirements.txt

**Step 2 — Run the Streamlit app:**
streamlit run app.py

**Step 3 — Open your browser at:**
http://localhost:8501

---

## 🔍 How It Works
1. User enters 13 patient clinical features in the web form
2. App scales the input using the saved StandardScaler
3. Scaled input is passed to the saved Logistic Regression model
4. Model returns prediction — High Risk or Low Risk
5. App displays result with probability percentage

---

## ⚠️ Disclaimer
This app is built for educational purposes as part of the 
BetaBytez Summer Internship 2026. It is not a substitute 
for professional medical advice.