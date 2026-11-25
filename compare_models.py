import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

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
    balanced_accuracy_score,
    confusion_matrix       
)


# -----------------------------------
# 1. Load Dataset
# -----------------------------------

df = pd.read_csv("data/diabetic_data.csv")

# Replace missing value symbols
df.replace("?", np.nan, inplace=True)

# Create target column
# 1 = Readmitted within 30 days
# 0 = Not readmitted within 30 days

df["target"] = (df["readmitted"] == "<30").astype(int)


# -----------------------------------
# 2. Select 14 Features
# -----------------------------------

selected_features = [
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

X = df[selected_features]
y = df["target"]

# Patient IDs are used only for splitting
groups = df["patient_nbr"]


# -----------------------------------
# 3. Categorical and Numeric Features
# -----------------------------------

categorical_features = [
    "race",
    "gender",
    "age",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id"
]

numeric_features = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses"
]


# -----------------------------------
# 4. Preprocessing
# -----------------------------------
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

from sklearn.preprocessing import StandardScaler

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ("cat", categorical_transformer, categorical_features),
    ("num", numeric_transformer, numeric_features)
])


# -----------------------------------
# 5. Split into Train, Validation, Test
# -----------------------------------

# First: Separate test data (20%)

split1 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

trainval_idx, test_idx = next(
    split1.split(X, y, groups=groups)
)

X_trainval = X.iloc[trainval_idx]
y_trainval = y.iloc[trainval_idx]
groups_trainval = groups.iloc[trainval_idx]

X_test = X.iloc[test_idx]
y_test = y.iloc[test_idx]


# Second: Separate validation data from training data

split2 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, val_idx = next(
    split2.split(
        X_trainval,
        y_trainval,
        groups=groups_trainval
    )
)

X_train = X_trainval.iloc[train_idx]
y_train = y_trainval.iloc[train_idx]

X_val = X_trainval.iloc[val_idx]
y_val = y_trainval.iloc[val_idx]


print("Dataset splitting completed.")
print("Training records:", len(X_train))
print("Validation records:", len(X_val))
print("Testing records:", len(X_test))


# -----------------------------------
# 6. Define Models
# -----------------------------------
models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced",
        max_iter=5000,
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

# -----------------------------------
# 7. Train, Tune Threshold, Evaluate
# -----------------------------------

results = []

for name, classifier in models.items():

    print("\nTraining:", name)

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    # Train model
    model.fit(X_train, y_train)

    # Get validation scores
    val_scores = model.predict_proba(X_val)[:, 1]

    # Find best threshold using validation F1-score
    thresholds = np.arange(0.05, 0.96, 0.01)

    best_threshold = 0.50
    best_f1 = -1

    for threshold in thresholds:

        val_predictions = (
            val_scores >= threshold
        ).astype(int)

        current_f1 = f1_score(
            y_val,
            val_predictions,
            zero_division=0
        )

        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = threshold

    # SAVE LOGISTIC REGRESSION MODEL
    if name == "Logistic Regression":

        joblib.dump(
            model,
            "logistic_regression_model.pkl"
        )

        joblib.dump(
            best_threshold,
            "logistic_regression_threshold.pkl"
        )

        print("Logistic Regression model saved!")
        print("Threshold saved!")

    # Evaluate on untouched test data
    test_scores = model.predict_proba(X_test)[:, 1]

    test_predictions = (
        test_scores >= best_threshold
    ).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(y_test, test_predictions)

    precision = precision_score(
        y_test,
        test_predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        test_predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        test_predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(y_test, test_scores)

    pr_auc = average_precision_score(
        y_test,
        test_scores
    )

    balanced_acc = balanced_accuracy_score(
        y_test,
        test_predictions
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        test_predictions,
        labels=[0, 1]
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) else 0

    # Store results
    results.append({
        "Model": name,
        "Best Threshold": round(best_threshold, 2),
        "Validation F1": round(best_f1, 4),
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1 Score": round(f1, 4),
        "Specificity": round(specificity, 4),
        "ROC-AUC": round(roc_auc, 4),
        "PR-AUC": round(pr_auc, 4),
        "Balanced Accuracy": round(balanced_acc, 4)
    })

    print("Best threshold:", round(best_threshold, 2))
    print("Test F1 Score:", round(f1, 4))
    print("Test Recall:", round(recall, 4))


# -----------------------------------
# 8. Display Final Comparison
# -----------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
)

print("\n========== IMPROVED MODEL COMPARISON ==========\n")

print(results_df.to_string(index=False))


# -----------------------------------
# 9. Save Results
# -----------------------------------

results_df.to_csv(
    "improved_model_comparison_results.csv",
    index=False
)

print("\nResults saved successfully!")

print("File: improved_model_comparison_results.csv")
