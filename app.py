import streamlit as st
import joblib
import numpy as np

# Load the trained fraud detection model
model = joblib.load("fraud_model.pkl")

# App heading
st.title("💳 Credit Card Fraud Checker")

st.markdown("Provide transaction details below to get a prediction.")

# Transaction-related inputs
scaled_amount = st.number_input("Normalized Transaction Amount", value=0.5)
transaction_index = st.number_input("Transaction Sequence Number", value=10000)
average_spending = st.number_input("Rolling Average of Recent Amounts", value=0.5)

# Input fields for anonymized PCA features (V1 to V28)
st.markdown("You can tweak the underlying feature values (V1 to V28) here:")
pca_components = [st.number_input(f"Feature V{i}", value=0.0) for i in range(1, 29)]

# Combine all user inputs into a single array
features = np.array([*pca_components, scaled_amount, transaction_index, average_spending]).reshape(1, -1)

# Predict fraud when button is clicked
if st.button("Analyze Transaction"):
    is_fraud = model.predict(features)[0]
    fraud_probability = model.predict_proba(features)[0][1]

    if is_fraud == 1:
        st.error("⚠️ Warning: This transaction may be fraudulent.")
    else:
        st.success("✅ Transaction appears to be legitimate.")

    st.caption(f"Model confidence: {fraud_probability:.2%}")
