import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Page config
st.set_page_config(
    page_title="Credit Card Fraud Detector",
    page_icon=None,
    layout="centered"
)

# Load models
@st.cache_resource
def load_models():
    model            = joblib.load('models/xgb_model.pkl')
    scaler           = joblib.load('models/scaler.pkl')
    selector         = joblib.load('models/feature_selector.pkl')
    selected_features= joblib.load('models/selected_features.pkl')
    return model, scaler, selector, selected_features

model, scaler, selector, selected_features = load_models()

# All original features before selection
ALL_FEATURES = ['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10',
                'V11','V12','V13','V14','V15','V16','V17','V18','V19',
                'V20','V21','V22','V23','V24','V25','V26','V27','V28',
                'Amount_log','Hour','is_night']

# Header
st.title("Credit Card Fraud Detection System")
st.markdown("Enter transaction details below to predict whether it is legitimate or fraudulent.")
st.markdown("---")

# Input form
st.subheader("Transaction Input")

col1, col2 = st.columns(2)

with col1:
    amount  = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0, step=0.01)
    hour    = st.slider("Transaction Hour (0-23)", min_value=0, max_value=23, value=12)

with col2:
    is_night = 1 if (hour >= 22 or hour <= 6) else 0
    st.metric("Is Night Transaction", "Yes" if is_night else "No")
    st.metric("Amount (log-transformed)", f"{np.log1p(amount):.4f}")

st.markdown("---")
st.subheader("PCA Feature Values (V1 - V28)")
st.caption("These are the anonymized PCA components from the original transaction data.")

# V1-V28 sliders in 4 columns
v_values = {}
cols = st.columns(4)
for i in range(1, 29):
    col_idx = (i - 1) % 4
    with cols[col_idx]:
        v_values[f'V{i}'] = st.number_input(
            f"V{i}", value=0.0, format="%.4f", key=f"v{i}"
        )

st.markdown("---")

# Predict button
if st.button("Predict Transaction", use_container_width=True):
    
    # Build full feature vector
    input_dict = {f'V{i}': v_values[f'V{i}'] for i in range(1, 29)}
    input_dict['Amount_log'] = np.log1p(amount)
    input_dict['Hour']       = hour
    input_dict['is_night']   = is_night
    
    input_df = pd.DataFrame([input_dict], columns=ALL_FEATURES)
    
    # Scale
    input_scaled = scaler.transform(input_df)
    
    # Select features
    input_selected = selector.transform(input_scaled)
    
    # Predict
    prediction = model.predict(input_selected)[0]
    probability = model.predict_proba(input_selected)[0]
    
    fraud_prob = probability[1] * 100
    legit_prob = probability[0] * 100
    
    st.markdown("---")
    st.subheader("Prediction Result")
    
    if prediction == 1:
        st.error("FRAUDULENT TRANSACTION DETECTED")
        st.markdown(f"**Fraud Probability   : {fraud_prob:.2f}%**")
        st.markdown(f"Legitimate Probability : {legit_prob:.2f}%")
    else:
        st.success("LEGITIMATE TRANSACTION")
        st.markdown(f"**Legitimate Probability : {legit_prob:.2f}%**")
        st.markdown(f"Fraud Probability        : {fraud_prob:.2f}%")
    
    st.progress(int(fraud_prob))
    st.caption(f"Fraud risk score: {fraud_prob:.2f}%")

st.markdown("---")
st.caption("ML InnovateX Hackathon | Credit Card Fraud Detection | XGBoost + ANN")