import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    brier_score_loss,
    RocCurveDisplay,
    PrecisionRecallDisplay
)

print("========== INDEPENDENT MODEL VALIDATION ==========\n")

# 1. Load dataset, saved model and threshold
df = pd.read_csv("data/diabetic_data.csv")

model = joblib.load("hospital_readmission_model.pkl")
threshold = joblib.load("readmission_threshold.pkl")

df["target"] = (df["readmitted"] == "<30").astype(int)

print("Dataset loaded successfully.")
print("Saved model loaded successfully.")
print("Threshold:", threshold)

# 2. Select the same 14 features used during training
selected_features = [
    "race", "gender", "age",
    "admission_type_id", "discharge_disposition_id",
    "admission_source_id", "time_in_hospital",
    "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient",
    "number_emergency", "number_inpatient",
    "number_diagnoses"
]

X = df[selected_features].replace("?", np.nan)
y = df["target"]
groups = df["patient_nbr"]

# 3. Reproduce the original patient-level test split
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_index, test_index = next(
    splitter.split(X, y, groups=groups)
)

X_test = X.iloc[test_index]
y_test = y.iloc[test_index]

# 4. Verify that test patients were not in training
train_patients = set(groups.iloc[train_index])
test_patients = set(groups.iloc[test_index])

assert len(train_patients.intersection(test_patients)) == 0

print("\nPASS: No patient appears in both training and test sets.")
print("Test records:", len(X_test))
print("Test patients:", len(test_patients))

# 5. Generate predictions on the untouched test set
probabilities = model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= threshold).astype(int)

# 6. Calculate evaluation metrics
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, zero_division=0)
recall = recall_score(y_test, predictions, zero_division=0)
f1 = f1_score(y_test, predictions, zero_division=0)
balanced_acc = balanced_accuracy_score(y_test, predictions)
mcc = matthews_corrcoef(y_test, predictions)
roc_auc = roc_auc_score(y_test, probabilities)
pr_auc = average_precision_score(y_test, probabilities)
brier = brier_score_loss(y_test, probabilities)

cm = confusion_matrix(y_test, predictions)
tn, fp, fn, tp = cm.ravel()

specificity = tn / (tn + fp) if (tn + fp) else 0

print("\n========== PERFORMANCE RESULTS ==========")

print(f"Accuracy:             {accuracy:.4f}")
print(f"Precision:            {precision:.4f}")
print(f"Recall (Sensitivity): {recall:.4f}")
print(f"Specificity:          {specificity:.4f}")
print(f"F1 Score:             {f1:.4f}")
print(f"Balanced Accuracy:    {balanced_acc:.4f}")
print(f"ROC-AUC:              {roc_auc:.4f}")
print(f"PR-AUC:               {pr_auc:.4f}")
print(f"MCC:                  {mcc:.4f}")
print(f"Brier Score:          {brier:.4f}")

print("\n========== CONFUSION MATRIX ==========")
print(cm)

print("\n========== CLASSIFICATION REPORT ==========")
print(classification_report(
    y_test,
    predictions,
    target_names=["Not readmitted within 30 days", "Readmitted within 30 days"],
    zero_division=0
))

# 7. Save metrics
results = pd.DataFrame([{
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "Specificity": specificity,
    "F1_Score": f1,
    "Balanced_Accuracy": balanced_acc,
    "ROC_AUC": roc_auc,
    "PR_AUC": pr_auc,
    "MCC": mcc,
    "Brier_Score": brier
}])

results.to_csv("independent_validation_results.csv", index=False)

print("Results saved to independent_validation_results.csv")

# 8. Generate ROC curve
RocCurveDisplay.from_predictions(y_test, probabilities)
plt.title("Independent Validation - ROC Curve")
plt.savefig("validation_roc_curve.png", bbox_inches="tight")
plt.show()

# 9. Generate Precision-Recall curve
PrecisionRecallDisplay.from_predictions(y_test, probabilities)
plt.title("Independent Validation - Precision Recall Curve")
plt.savefig("validation_pr_curve.png", bbox_inches="tight")
plt.show()

# 10. Generate confusion matrix graph
plt.figure(figsize=(6, 5))
plt.imshow(cm)
plt.title("Independent Validation - Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.xticks([0, 1], ["Not Readmitted", "Readmitted"])
plt.yticks([0, 1], ["Not Readmitted", "Readmitted"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")

plt.savefig("validation_confusion_matrix.png", bbox_inches="tight")
plt.show()

print("\n========== VALIDATION COMPLETED ==========")