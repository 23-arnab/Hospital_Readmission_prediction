
import os
import joblib
import pandas as pd
import numpy as np

print("\n====================================")
print(" FINAL MODEL TESTING")
print("====================================\n")


# -----------------------------------
# 1. Check Model Files
# -----------------------------------

model_file = "logistic_regression_model.pkl"
threshold_file = "logistic_regression_threshold.pkl"

assert os.path.exists(model_file), "Model file not found!"
print("PASS: Model file exists")

assert os.path.exists(threshold_file), "Threshold file not found!"
print("PASS: Threshold file exists")


# -----------------------------------
# 2. Load Model and Threshold
# -----------------------------------

model = joblib.load(model_file)
threshold = joblib.load(threshold_file)

assert 0 <= threshold <= 1

print("PASS: Model loaded successfully")
print("PASS: Threshold loaded successfully")
print("Classification threshold:", threshold)


# -----------------------------------
# 3. Prepare Sample Patients
# -----------------------------------

features = [
    "race",
    "gender",
    "age",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses"
]

patients = pd.DataFrame([

    {
        "race": "Caucasian",
        "gender": "Female",
        "age": "[50-60)",
        "admission_type_id": 1,
        "discharge_disposition_id": 1,
        "admission_source_id": 7,
        "time_in_hospital": 3,
        "num_lab_procedures": 40,
        "num_procedures": 1,
        "num_medications": 15,
        "number_outpatient": 0,
        "number_emergency": 0,
        "number_inpatient": 0,
        "number_diagnoses": 5
    },

    {
        "race": "Caucasian",
        "gender": "Male",
        "age": "[70-80)",
        "admission_type_id": 1,
        "discharge_disposition_id": 1,
        "admission_source_id": 7,
        "time_in_hospital": 8,
        "num_lab_procedures": 65,
        "num_procedures": 2,
        "num_medications": 25,
        "number_outpatient": 2,
        "number_emergency": 1,
        "number_inpatient": 3,
        "number_diagnoses": 8
    },

    {
        "race": "AfricanAmerican",
        "gender": "Female",
        "age": "[40-50)",
        "admission_type_id": 2,
        "discharge_disposition_id": 1,
        "admission_source_id": 1,
        "time_in_hospital": 2,
        "num_lab_procedures": 30,
        "num_procedures": 0,
        "num_medications": 8,
        "number_outpatient": 0,
        "number_emergency": 0,
        "number_inpatient": 0,
        "number_diagnoses": 3
    }

])

# Ensure correct feature order
patients = patients[features]

print("\nPASS: Sample patient data prepared")


# -----------------------------------
# 4. Test Model Predictions
# -----------------------------------

scores = model.predict_proba(patients)[:, 1]

assert len(scores) == 3
assert np.all(np.isfinite(scores))
assert np.all((scores >= 0) & (scores <= 1))

print("PASS: Predictions generated")
print("PASS: All prediction scores are valid")


# -----------------------------------
# 5. Test Classification
# -----------------------------------

predictions = (scores >= threshold).astype(int)

assert len(predictions) == 3
assert set(predictions).issubset({0, 1})

print("PASS: Classification generated")


# -----------------------------------
# 6. Display Final Results
# -----------------------------------

print("\n====================================")
print("       PATIENT TEST RESULTS")
print("====================================")

for i in range(len(patients)):

    print("\nPatient:", i + 1)

    print("Readmission Score:", round(scores[i] * 100, 2), "%")

    if predictions[i] == 1:
        print("Prediction: Flagged by model")
    else:
        print("Prediction: Not flagged by model")


# -----------------------------------
# 7. Final Testing Status
# -----------------------------------

print("\n====================================")
print(" ALL FINAL MODEL TESTS PASSED")
print("====================================")
