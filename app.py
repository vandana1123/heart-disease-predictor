import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Load the "Brain" and the "Ruler"
# Change this:
model = joblib.load('Heart Model.pkl')
scaler = joblib.load('Scaler Model.pkl')
# 2. Page Setup
st.set_page_config(page_title="CardioGuard AI", page_icon="❤️")
st.title("🩺 CardioGuard: Heart Disease Risk Assessment")
st.write("This AI model predicts the likelihood of heart disease based on clinical parameters.")

# 3. Sidebar for User Input
st.sidebar.header("Patient Data Input")

def user_input_features():
    age = st.sidebar.number_input("Age", 1, 100, 50)
    sex = st.sidebar.selectbox("Sex (1=Male, 0=Female)", [1, 0])
    cp = st.sidebar.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
    trestbps = st.sidebar.number_input("Resting Blood Pressure", 80, 200, 120)
    chol = st.sidebar.number_input("Serum Cholestoral (mg/dl)", 100, 600, 200)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)", [0, 1])
    restecg = st.sidebar.selectbox("Resting ECG Results (0-2)", [0, 1, 2])
    thalach = st.sidebar.number_input("Max Heart Rate Achieved", 60, 220, 150)
    exang = st.sidebar.selectbox("Exercise Induced Angina (1=Yes, 0=No)", [0, 1])
    oldpeak = st.sidebar.number_input("ST Depression (Oldpeak)", 0.0, 6.0, 1.0)
    slope = st.sidebar.selectbox("Slope of Peak Exercise ST Segment (0-2)", [0, 1, 2])
    ca = st.sidebar.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
    thal = st.sidebar.selectbox("Thalassemia (0-3)", [0, 1, 2, 3])

    data = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# 4. Prediction Logic
if st.button("Analyze Risk"):
    # Scale the input just like we did in training
    input_scaled = scaler.transform(input_df)
    
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)

    # 5. Show Results
    st.subheader("Results")
    if prediction[0] == 1:
        st.error(f"⚠️ High Risk Detected! (Probability: {prediction_proba[0][1]:.2%})")
        st.write("The model suggests a high likelihood of heart disease. Please consult a professional.")
    else:
        st.success(f"✅ Low Risk Detected. (Probability of Disease: {prediction_proba[0][1]:.2%})")
        st.write("The model suggests the heart parameters are within a healthy range.")
        