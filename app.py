import streamlit as st
import pandas as pd
import pickle

# Load model and scaler
model = pickle.load(open("machine_failure_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Machine Failure Prediction", page_icon="🏭")

st.title("🏭 Machine Failure Prediction System")
st.write("Enter machine sensor values below")

footfall = st.number_input("Footfall", min_value=0)
tempMode = st.number_input("Temp Mode", min_value=0)
AQ = st.number_input("Air Quality (AQ)", min_value=0)
USS = st.number_input("USS", min_value=0)
CS = st.number_input("Current Sensor (CS)", min_value=0)
VOC = st.number_input("VOC", min_value=0)
RP = st.number_input("RP", min_value=0)
IP = st.number_input("IP", min_value=0)
Temperature = st.number_input("Temperature", min_value=0)

if st.button("Predict Failure"):

    data = pd.DataFrame([[footfall,tempMode,AQ,USS,CS,VOC,RP,IP,Temperature]],
                        columns=['footfall','tempMode','AQ','USS','CS','VOC','RP','IP','Temperature'])

    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)[0]
    probability = model.predict_proba(data_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Machine Failure Risk Detected ({probability:.2%})")
    else:
        st.success(f"✅ Machine Healthy ({1-probability:.2%})")