import os
import joblib
import pandas as pd

print("========== FINAL MODEL TESTING ==========\n")

# Test 1: Check required files
print("TEST 1: Checking project files")

required_files = [
    "hospital_readmission_model.pkl",
    "readmission_threshold.pkl"
]

for file in required_files:
    if os.path.exists(file):
        print(f"PASS: {file} found")
    else:
        print(f"FAIL: {file} not found")
        raise FileNotFoundError(file)

# Test 2: Load model and threshold
print("\nTEST 2: Loading saved model")

model = joblib.load("hospital_readmission_model.pkl")
threshold = joblib.load("readmission_threshold.pkl")

print("PASS: Model loaded successfully")
print("PASS: Threshold loaded successfully")
print("Threshold:", threshold)

# Test 3: Prepare sample patient information
print("\nTEST 3: Testing patient input")

patient = pd.DataFrame([{
    "race": "Caucasian",
    "gender": "Female",
    "age": "[50-60)",
    "admission_type_id": 1,
    "discharge_disposition_id": 1,
    "admission_source_id": 7,
    "time_in_hospital": 3,
    "num_lab_procedures": 40,
    "num_procedures": 1,
    "num_medications": 10,
    "number_outpatient": 0,
    "number_emergency": 0,
    "number_inpatient": 0,
    "number_diagnoses": 5
}])

print("PASS: Sample patient information created")

# Test 4: Check prediction
print("\nTEST 4: Generating prediction")

probability = model.predict_proba(patient)[0][1]
prediction = int(probability >= threshold)

print("PASS: Prediction generated")
print("Readmission score:", round(probability * 100, 2), "%")
print("Classification threshold:", threshold)

if prediction == 1:
    print("Result: Higher readmission risk flagged")
else:
    print("Result: Higher readmission risk not flagged")

# Test 5: Check probability
print("\nTEST 5: Checking prediction validity")

assert 0 <= probability <= 1, "Invalid prediction score"

print("PASS: Prediction score is within valid range")

# Test 6: Test multiple patients
print("\nTEST 6: Testing multiple patient records")

patients = pd.concat([patient, patient, patient], ignore_index=True)

probabilities = model.predict_proba(patients)[:, 1]

assert len(probabilities) == 3
assert all(0 <= p <= 1 for p in probabilities)

print("PASS: Model successfully predicted 3 patient records")

print("\n========== TESTING COMPLETED ==========")
print("All model tests passed!")