
import streamlit as st
import pandas as pd
import joblib

# Load trained model and threshold
model = joblib.load("logistic_regression_model.pkl")
threshold = joblib.load("logistic_regression_threshold.pkl")
# Page configuration
st.set_page_config(
    page_title="Hospital Readmission Prediction",
    page_icon="🏥",
    layout="wide"
)

# Main heading
st.title("🏥 Hospital Readmission Prediction System")

st.write(
    "Machine Learning-Based Prediction of Hospital Readmission Risk"
)

st.info(
    "This educational application uses a Random Forest model "
    "to classify possible readmission within 30 days."
)

st.divider()

# Patient information
st.header("Enter Patient Information")

col1, col2 = st.columns(2)

with col1:

    race = st.selectbox(
        "Race",
        [
            "Caucasian",
            "AfricanAmerican",
            "Asian",
            "Hispanic",
            "Other",
            "Unknown"
        ]
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female", "Unknown/Invalid"]
    )

    age = st.selectbox(
        "Age Group",
        [
            "[0-10)", "[10-20)", "[20-30)",
            "[30-40)", "[40-50)", "[50-60)",
            "[60-70)", "[70-80)", "[80-90)",
            "[90-100)"
        ]
    )

    admission_type_id = st.selectbox(
        "Admission Type ID",
        list(range(1, 9))
    )

    discharge_disposition_id = st.selectbox(
        "Discharge Disposition ID",
        list(range(1, 31))
    )

    admission_source_id = st.selectbox(
        "Admission Source ID",
        list(range(1, 26))
    )

with col2:

    time_in_hospital = st.number_input(
        "Days in Hospital",
        min_value=1,
        max_value=14,
        value=3
    )

    num_lab_procedures = st.number_input(
        "Number of Lab Procedures",
        min_value=0,
        max_value=150,
        value=40
    )

    num_procedures = st.number_input(
        "Number of Procedures",
        min_value=0,
        max_value=10,
        value=1
    )

    num_medications = st.number_input(
        "Number of Medications",
        min_value=0,
        max_value=100,
        value=10
    )

    number_outpatient = st.number_input(
        "Outpatient Visits",
        min_value=0,
        max_value=50,
        value=0
    )

    number_emergency = st.number_input(
        "Emergency Visits",
        min_value=0,
        max_value=50,
        value=0
    )

    number_inpatient = st.number_input(
        "Previous Inpatient Visits",
        min_value=0,
        max_value=50,
        value=0
    )

    number_diagnoses = st.number_input(
        "Number of Diagnoses",
        min_value=1,
        max_value=16,
        value=5
    )

st.divider()

# Prediction
if st.button("Predict Readmission Risk", type="primary"):

    # Prepare patient data
    patient_data = pd.DataFrame([{
        "race": race,
        "gender": gender,
        "age": age,
        "admission_type_id": admission_type_id,
        "discharge_disposition_id": discharge_disposition_id,
        "admission_source_id": admission_source_id,
        "time_in_hospital": time_in_hospital,
        "num_lab_procedures": num_lab_procedures,
        "num_procedures": num_procedures,
        "num_medications": num_medications,
        "number_outpatient": number_outpatient,
        "number_emergency": number_emergency,
        "number_inpatient": number_inpatient,
        "number_diagnoses": number_diagnoses
    }])

    # Predict readmission score
    probability = model.predict_proba(patient_data)[0][1]

    st.header("Prediction Result")

    st.metric(
        "Model Readmission Score",
        f"{probability * 100:.2f}%"
    )

    st.write(
        f"Classification threshold: {threshold:.2f}"
    )

    if probability >= threshold:

        st.warning(
            "The model flags this encounter as "
            "higher readmission risk."
        )

    else:

        st.success(
            "The model does not flag this encounter "
            "as higher readmission risk."
        )

    st.caption(
        "This score is an experimental model output, "
        "not a calibrated clinical probability or medical diagnosis."
    )
