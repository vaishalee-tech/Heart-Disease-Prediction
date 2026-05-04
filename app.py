import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load files
model = joblib.load('Log_Reg_heart.pkl')
scaler = joblib.load('scaler.pkl')
columns = joblib.load('columns.pkl')

# Page config
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

# Title
st.markdown("<h1 style='text-align: center; color: green;'>❤️ Heart Disease Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; '>AI-powered health risk detection system</p>", unsafe_allow_html=True)

st.write("---")

# Input section
st.subheader("📝 Enter Patient Details")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 100, 40)
    sex = st.selectbox("Sex", ['M', 'F'])
    chest_pain = st.selectbox("Chest Pain Type", ['ATA','NAP','TA','ASY'])
    resting_BP = st.number_input("Resting BP", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol", 100, 600, 200)

with col2:
    fasting_bs = st.selectbox("Fasting BS > 120", [0,1])
    resting_ecg = st.selectbox("Resting ECG", ["Normal","ST","LVH"])
    max_hr = st.slider("Max Heart Rate", 60, 220, 150)
    exercise_angina = st.selectbox("Exercise Angina", ['Y','N'])
    oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)
    st_slope = st.selectbox("ST Slope", ['Up','Flat','Down'])

st.write("---")

# Prediction button
if st.button("🔍 Predict Risk"):

    raw_input = {
        'Age': age,
        'RestingBP': resting_BP,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,

        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    for col in columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[columns]

    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    prob = model.predict_proba(scaled_input)[0]

    st.write("### 🔎 Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Disease\nConfidence: {round(prob[1]*100,2)}%")
    else:
        st.success(f"✅ Low Risk of Heart Disease\nConfidence: {round(prob[0]*100,2)}%")

    with st.expander("📊 View Input Data"):
        st.write(input_df)

st.write("---")

# Footer
st.markdown("<p style='text-align:center; color:blue;'>Developed by Sanu </p>", unsafe_allow_html=True)