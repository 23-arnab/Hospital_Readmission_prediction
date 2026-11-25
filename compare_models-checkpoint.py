import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score
)

print("========== MODEL COMPARISON ==========\n")

# 1. Load dataset
df = pd.read_csv("data/diabetic_data.csv")

df["target"] = (df["readmitted"] == "<30").astype(int)

# 2. Select the same 14 features
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

categorical_cols = [
    "race", "gender", "age",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id"
]

numerical_cols = [
    col for col in selected_features
    if col not in categorical_cols
]

# 3. Preprocessing
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ("cat", categorical_transformer, categorical_cols),
    ("num", numerical_transformer, numerical_cols)
])

# 4. Same patient-level split for all models
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_index, test_index = next(
    splitter.split(X, y, groups=df["patient_nbr"])
)

X_train = X.iloc[train_index]
X_test = X.iloc[test_index]

y_train = y.iloc[train_index]
y_test = y.iloc[test_index]

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# 5. Create three models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}

# 6. Train and evaluate
results = []

for name, classifier in models.items():

    print("\nTraining:", name)

    model = Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("classifier", classifier)
    ])

    model.fit(X_train, y_train)

    # Use the same 0.50 threshold for all three models
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )
    roc_auc = roc_auc_score(y_test, probabilities)
    pr_auc = average_precision_score(y_test, probabilities)
    balanced_acc = balanced_accuracy_score(
        y_test, predictions
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        "Balanced Accuracy": balanced_acc
    })

    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("ROC-AUC:", round(roc_auc, 4))
    print("PR-AUC:", round(pr_auc, 4))
    print("Balanced Accuracy:", round(balanced_acc, 4))

# 7. Display comparison table
comparison = pd.DataFrame(results)

print("\n========== FINAL COMPARISON ==========")
print(comparison.round(4).to_string(index=False))

# 8. Save comparison results
comparison.to_csv("model_comparison_results.csv", index=False)

print("\nComparison saved successfully!")