# ❤️ Heart Disease Prediction System

 ## Overview:-
This project is a Machine Learning-based web application that predicts the risk of heart disease using patient health data. It helps in early detection by analyzing various medical parameters.

## Features:-
- Predicts heart disease risk (High / Low)
- Interactive user interface using Streamlit
- Real-time prediction
- Confidence score display
- Clean and user-friendly design

##  Tech Stack:-
- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit
- Joblib

## Machine Learning Workflow:-
1. Data Collection (Kaggle Dataset)
2. Data Cleaning & Preprocessing
3. Feature Selection
4. Encoding (One-Hot Encoding)
5. Feature Scaling (StandardScaler)
6. Model Training (Logistic Regression)
7. Model Evaluation (Accuracy & F1-score)

##  Model Performance:-
- Logistic Regression gave the best performance
- Accuracy: ~86%
- F1 Score: ~88%

##  Project Structure:-
Heart-Disease-Prediction/
│
├── app.py # Streamlit web app
├── heart.ipynb # Model training notebook
├── heart.csv # Dataset
├── Log_Reg_heart.pkl # Trained model
├── scaler.pkl # Scaler
├── columns.pkl # Feature columns
│
├── frontpage.png # App UI screenshot
├── input.png # Input example
├── OP-1.png # Output (Low Risk)
├── OP-2.png # Output (High Risk)
│
└── README.md # Documentation


