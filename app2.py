import streamlit as st
import pandas as pd
from xgboost import XGBClassifier
import joblib

# Load the trained model
classifier = joblib.load('xgboost_model.pkl')

# Load the saved scaler
scaler = joblib.load('standard_scaler.pkl')

def predict_single_observation(observation):
    # Predict on the single observation
    prediction = classifier.predict(scaler.transform([observation]))
    return prediction[0]

def main():
    st.title('Fraud Detection App')

    # Input fields for each column
    st.write('Enter the values for each feature:')
    time = st.number_input('Time')
    v1 = st.number_input('V1')
    v2 = st.number_input('V2')
    v3 = st.number_input('V3')
    v4 = st.number_input('V4')
    v5 = st.number_input('V5')
    v6 = st.number_input('V6')
    v7 = st.number_input('V7')
    v8 = st.number_input('V8')
    v9 = st.number_input('V9')
    v10 = st.number_input('V10')
    v11 = st.number_input('V11')
    v12 = st.number_input('V12')
    v13 = st.number_input('V13')
    v14 = st.number_input('V14')
    v15 = st.number_input('V15')
    v16 = st.number_input('V16')
    v17 = st.number_input('V17')
    v18 = st.number_input('V18')
    v19 = st.number_input('V19')
    v20 = st.number_input('V20')
    v21 = st.number_input('V21')
    v22 = st.number_input('V22')
    v23 = st.number_input('V23')
    v24 = st.number_input('V24')
    v25 = st.number_input('V25')
    v26 = st.number_input('V26')
    v27 = st.number_input('V27')
    v28 = st.number_input('V28')
    amount = st.number_input('Amount')

    # Create a button to make predictions
    if st.button('Predict'):
        # Create a single observation from the inputs
        single_observation = [time, v1, v2, v3, v4, v5, v6, v7, v8, v9,
                              v10, v11, v12, v13, v14, v15, v16, v17, v18, v19,
                              v20, v21, v22, v23, v24, v25, v26, v27, v28, amount]
        prediction = predict_single_observation(single_observation)

        if prediction == 0:
            st.write('Prediction: Non-fraud transaction')
        elif prediction == 1:
            st.write('Prediction: Fraud transaction')
        else:
            st.write('Unknown prediction')

if __name__ == '__main__':
    main()
