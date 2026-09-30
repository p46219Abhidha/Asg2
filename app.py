import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load models and separate column lists
lin_model = joblib.load('insurance_linear.sav')
log_model = joblib.load('insurance_logistic.sav')
linear_columns = joblib.load('linear_columns.sav')
logistic_columns = joblib.load('logistic_columns.sav')

st.title('Medical Cost & High-Risk Prediction Tool')
st.write('Enter the patient details below to predict medical charges or catastrophic risk.')

prediction_type = st.sidebar.radio(
    "What would you like to predict?",
    ("Medical Charges (Linear)", "Catastrophic High-Cost Risk (Logistic)")
)

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)

with col2:
    sex_input = st.selectbox("Sex", ['male', 'female'])
    region_input = st.selectbox("Region", ['northeast', 'northwest', 'southeast', 'southwest'])
    smoker_input = st.selectbox("Smoker", ['no', 'yes'])

if st.button('Predict', type="primary"):
    # --- 1. BASE PREPROCESSING ---
    sex = 0 if sex_input == 'male' else 1
    smoker = 1 if smoker_input == 'yes' else 0
    region_map = {'northeast': 0, 'northwest': 1, 'southeast': 2, 'southwest': 3}
    region = region_map[region_input]

    # --- 2. BUILD BASE DICTIONARY ---
    input_data = {
        'age': [age], 'sex': [sex], 'bmi': [bmi], 
        'children': [children], 'smoker': [smoker], 'region': [region]
    }

    if prediction_type == "Medical Charges (Linear)":
        # Add engineered features ONLY for the linear model
        input_data['smoker_bmi_interaction'] = [smoker * bmi]
        input_data['smoker_age'] = [smoker * age]
        input_data['smoker_children'] = [smoker * children]
        input_data['bmi_age'] = [bmi * age]
        input_data['is_obese'] = [1 if bmi >= 30 else 0]
        
        if age <= 30: age_group = 0
        elif age <= 45: age_group = 1
        elif age <= 60: age_group = 2
        else: age_group = 3
        
        input_data['age_group'] = [age_group]
        input_data['lifestyle_risk_score'] = [(smoker * 3) + (input_data['is_obese'][0] * 2)]

        # Create DataFrame and match Linear columns
        input_df = pd.DataFrame(input_data)
        input_df = input_df[linear_columns]
        
        # Predict
        prediction = lin_model.predict(input_df)[0]
        st.success(f'Predicted Medical Charges: **${prediction:,.2f}**')

    else:
        # Logistic Regression ONLY uses the base features
        input_df = pd.DataFrame(input_data)
        input_df = input_df[logistic_columns]
        
        # Predict
        proba = log_model.predict_proba(input_df)[0][1]
        prediction = log_model.predict(input_df)[0]
        
        status = "🔴 HIGH RISK" if prediction == 1 else "🟢 LOW RISK"
        st.success(f"### Catastrophic High-Cost Risk: {status}")
        st.info(f"Probability of exceeding $20,000 in medical charges: {proba:.2%}")
