
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("hospital_readmission_model.pkl")

# Page configuration
st.set_page_config(
    page_title="Hospital Readmission Prediction",
    page_icon="🏥",
    layout="wide"
)

# Application heading
st.title("🏥 Hospital Readmission Prediction System")
st.write(
    "Machine Learning-Based Prediction of Hospital Readmission Risk"
)

st.info(
    "This application predicts the possibility of "
    "readmission within 30 days using a Random Forest model."
)

st.divider()

# Patient information
st.header("Enter Patient Information")

col1, col2 = st.columns(2)

with col1:
    race = st.selectbox(
        "Race",
        ["Caucasian", "AfricanAmerican", "Asian",
         "Hispanic", "Other", "Unknown"]
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female", "Unknown/Invalid"]
    )

    age = st.selectbox(
        "Age Group",
        ["[0-10)", "[10-20)", "[20-30)", "[30-40)",
         "[40-50)", "[50-60)", "[60-70)", "[70-80)",
         "[80-90)", "[90-100)"]
    )

    admission_type_id = st.selectbox(
        "Admission Type ID", list(range(1, 9))
    )

    discharge_disposition_id = st.selectbox(
        "Discharge Disposition ID", list(range(1, 30))
    )

    admission_source_id = st.selectbox(
        "Admission Source ID", list(range(1, 26))
    )

with col2:
    time_in_hospital = st.number_input(
        "Days in Hospital", 1, 14, 3
    )

    num_lab_procedures = st.number_input(
        "Number of Lab Procedures", 0, 150, 40
    )

    num_procedures = st.number_input(
        "Number of Procedures", 0, 10, 1
    )

    num_medications = st.number_input(
        "Number of Medications", 0, 100, 10
    )

    number_outpatient = st.number_input(
        "Outpatient Visits", 0, 50, 0
    )

    number_emergency = st.number_input(
        "Emergency Visits", 0, 50, 0
    )

    number_inpatient = st.number_input(
        "Previous Inpatient Visits", 0, 50, 0
    )

    number_diagnoses = st.number_input(
        "Number of Diagnoses", 1, 16, 5
    )

st.divider()

# Prediction button
if st.button("Predict Readmission Risk", type="primary"):

    # Prepare all original input features
    input_data = {}

    for col in model.feature_names_in_:

        if col == "race":
            input_data[col] = race

        elif col == "gender":
            input_data[col] = gender

        elif col == "age":
            input_data[col] = age

        elif col == "admission_type_id":
            input_data[col] = admission_type_id

        elif col == "discharge_disposition_id":
            input_data[col] = discharge_disposition_id

        elif col == "admission_source_id":
            input_data[col] = admission_source_id

        elif col == "time_in_hospital":
            input_data[col] = time_in_hospital

        elif col == "num_lab_procedures":
            input_data[col] = num_lab_procedures

        elif col == "num_procedures":
            input_data[col] = num_procedures

        elif col == "num_medications":
            input_data[col] = num_medications

        elif col == "number_outpatient":
            input_data[col] = number_outpatient

        elif col == "number_emergency":
            input_data[col] = number_emergency

        elif col == "number_inpatient":
            input_data[col] = number_inpatient

        elif col == "number_diagnoses":
            input_data[col] = number_diagnoses

        else:
            input_data[col] = "Unknown"

    # Convert input into DataFrame
    patient_data = pd.DataFrame([input_data])

    # Predict probability
    probability = model.predict_proba(patient_data)[0][1]

    # Apply threshold
    threshold = 0.35

    st.subheader("Prediction Result")

    st.metric(
        "Estimated Readmission Probability",
        f"{probability * 100:.2f}%"
    )

    if probability >= threshold:
        st.warning(
            "The model flags this encounter as higher readmission risk."
        )
    else:
        st.success(
            "The model does not flag this encounter as higher readmission risk."
        )

    st.caption(
        "Educational demonstration only. This is not a medical diagnosis "
        "or a clinically validated risk assessment."
    )
