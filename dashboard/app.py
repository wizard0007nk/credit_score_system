import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import pandas as pd
import joblib
from src.scoring import generate_credit_score

model = joblib.load("C:/Users/nisha/Desktop/credit_score_system/models/credit_model.pkl")
scaler = joblib.load( "C:/Users/nisha/Desktop/credit_score_system/models/scaler.pkl")

st.title("📊 Credit Score Prediction Dashboard")

age = st.number_input("Age", 18, 100, 30)
monthly_income = st.number_input("Monthly Income ($)", 0.0, step=100.0)
open_credit_lines = st.number_input("Open Credit Lines", 0, 50, 5)
credit_history = st.number_input("Credit History (Years)", 0, 30, 5)
loan_amount = st.number_input("Loan Amount ($)", 0.0, step=100.0)

if st.button("Predict Credit Score"):
    input_data = pd.DataFrame([{
        "age": age,
        "monthly_income": monthly_income,
        "open_credit_lines": open_credit_lines,
        "credit_history": credit_history,
        "loan_amount": loan_amount
    }])

    features = scaler.transform(input_data)
    prob = model.predict_proba(features)[0][1]
    score = generate_credit_score(prob)

    st.metric("Predicted Credit Score", score)

    if score >= 750:
        st.success(f"Your credit score is **{score}** – *Excellent*")
    elif score >= 700:
        st.success(f"Your credit score is **{score}** – *Good*")
    elif score >= 650:
        st.warning(f"Your credit score is **{score}** – *Fair*")
    else:
        st.error(f"Your credit score is **{score}** – *Poor*")
