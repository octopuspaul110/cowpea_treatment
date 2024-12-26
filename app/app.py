import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# --- Title ---
st.title("Classification Model for Treatments Prediction")

# --- Feature Inputs ---
st.sidebar.header("Input Features")

# Collect user inputs
pH = st.sidebar.number_input("pH", min_value=0.0, max_value=14.0, step=0.1)
oc = st.sidebar.number_input("% OC", min_value=0.0, step=0.1)
tn = st.sidebar.number_input("%TN", min_value=0.0, step=0.1)
p_mgkg = st.sidebar.number_input("P (mg/kg)", min_value=0.0, step=0.1)
k_cmolkg = st.sidebar.number_input("K (cmol/kg)", min_value=0.0, step=0.1)
sand = st.sidebar.number_input("% Sand", min_value=0, max_value=100, step=1)
silt = st.sidebar.number_input("% Silt", min_value=0, max_value=100, step=1)
clay = st.sidebar.number_input("% Clay ", min_value=0, max_value=100, step=1)


dir = path.Path(__file__)
sys.path.append(dir.parent.parent)

# load model
path_to_model = './models/soil_model.joblib'

with open(path_to_model, 'rb') as file:
    model = joblib.load(file)

# --- Process Data ---
def preprocess_and_infer():
    """
    Preprocess fake data and perform inference with a pre-trained model.

    Returns:
        str: Predicted treatment label.
    """
    # --- Fake Data ---
    input_data = {
        'pH': [pH],
        '% OC': [oc],
        '%TN': [tn],
        'P (mg/kg)': [p_mgkg],
        'K (cmol/kg)': [k_cmolkg],
        '% Sand': [sand],
        '% Silt': [silt],
        '% Clay ': [clay]
    }
    user_df = pd.DataFrame(input_data)

    # --- Feature Engineering ---
    user_df['Clay_Sand_Interaction'] = user_df['% Clay '] * user_df['% Sand']
    user_df['pH_K_Interaction'] = user_df['pH'] * user_df['K (cmol/kg)']
    user_df['OC_TN_Interaction'] = user_df['% OC'] * user_df['%TN']

    # --- Scaling Numeric Features ---
    scaler = StandardScaler()
    numeric_cols = user_df.select_dtypes(include=['float64', 'int64']).columns
    user_df[numeric_cols] = scaler.fit_transform(user_df[numeric_cols])

    # --- Load Pre-trained Model ---
    #model_file = "\cowpea_treatment\models\soil_model.joblib"
    #model = joblib.load(model_file)

    # --- Perform Prediction ---
    prediction = model.predict(user_df)[0]
    
    if isinstance(prediction,str):
        prediction_label = prediction
    else:
        # --- Convert Prediction to Label ---
        treatment_labels = [
            'BUFFER+NO PGPR', 'BUFFER+PGPR', 'CABMV+NO PGPR', 'CABMV+PGPR',
            'CMV+CABMV+NO PGPR', 'CMV+PGPR'
        ]
        prediction_label = treatment_labels[prediction]

    return prediction_label

# --- Prediction ---
if st.button("Predict Treatment"):
    try:
        prediction = preprocess_and_infer()
        st.success(f"The predicted treatment is: {prediction}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
