import streamlit as st
import joblib
import numpy as np
import pandas as pd

# --- Load Models and Column Lists ---
# We load 4 files: two models and their respective column orders
lin_model = joblib.load('insurance_linear.sav')
log_model = joblib.load('insurance_logistic.sav')
linear_columns = joblib.load('linear_columns.sav')
logistic_columns = joblib.load('logistic_columns.sav')

# --- Page Setup ---
st.set_page_config(page_title="Medical Cost Predictor", layout="wide")
st.title('🏥 Medical Cost & High-Risk Prediction Tool')
st.write('Enter the patient details below to predict medical charges or catastrophic risk.')

# --- Sidebar for Model Selection ---
prediction_type = st.sidebar.radio(
    "What would you like to predict?",
    ("Medical Charges (Linear)", "Catastrophic High-Cost Risk (Logistic)")
)

st.header("Patient Details")

# --- Input Fields ---
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)

with col2:
    sex_input = st.selectbox("Sex", ['male', 'female'])
    region_input = st.selectbox("Region", ['northeast', 'northwest', 'southeast', 'southwest'])
    smoker_input = st.selectbox("Smoker", ['no', 'yes'])

# --- Prediction Logic ---
if st.button('Predict', type="primary"):
    
    # 1. BASE PREPROCESSING (Convert text to numbers exactly as in Colab)
    sex = 0 if sex_input == 'male' else 1
    smoker = 1 if smoker_input == 'yes' else 0
    region_map = {'northeast': 0, 'northwest': 1, 'southeast': 2, 'southwest': 3}
    region = region_map[region_input]

    # 2. BUILD BASE INPUT DICTIONARY (Common for both models)
    base_data = {
        'age': [age], 
        'sex': [sex], 
        'bmi': [bmi], 
        'children': [children], 
        'smoker': [smoker], 
        'region': [region]
    }

    # --- LINEAR REGRESSION BRANCH ---
    if prediction_type == "Medical Charges (Linear)":
        # Add engineered features ONLY for the linear model
        base_data['smoker_bmi_interaction'] = [smoker * bmi]
        base_data['smoker_age'] = [smoker * age]
        base_data['smoker_children'] = [smoker * children]
        base_data['bmi_age'] = [bmi * age]
        base_data['is_obese'] = [1 if bmi >= 30 else 0]
        
        # Calculate Age Group
        if age <= 30: 
            age_group = 0
        elif age <= 45: 
            age_group = 1
        elif age <= 60: 
            age_group = 2
        else: 
            age_group = 3
        base_data['age_group'] = [age_group]
        
        # Calculate Lifestyle Risk Score
        base_data['lifestyle_risk_score'] = [(smoker * 3) + (base_data['is_obese'][0] * 2)]

        # Create DataFrame and match Linear columns exactly
        input_df = pd.DataFrame(base_data)
        input_df = input_df[linear_columns]
        
        # Predict
        prediction = lin_model.predict(input_df)[0]
        st.success(f'### Predicted Medical Charges: **${prediction:,.2f}**')

    # --- LOGISTIC REGRESSION BRANCH ---
    else:
        # For Logistic Regression, we ONLY use the base features
        # (No engineered features needed here because the $30k threshold makes it naturally robust)
        input_df = pd.DataFrame(base_data)
        input_df = input_df[logistic_columns]
        
        # Predict Probability and Class
        proba = log_model.predict_proba(input_df)[0][1]
        prediction = log_model.predict(input_df)[0]
        
        # Display Results
        if prediction == 1:
            status = "🔴 HIGH RISK"
            st.error(f"### Catastrophic High-Cost Risk: {status}")
        else:
            status = "🟢 LOW RISK"
            st.success(f"### Catastrophic High-Cost Risk: {status}")
            
        st.info(f"Probability of exceeding **$30,000** in annual medical charges: **{proba:.2%}**")
        
        # Add a contextual warning for managers
        if proba > 0.6:
            st.warning("⚠️ Recommendation: Consider assigning a case manager for preventive care.")
