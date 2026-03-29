# ============================================================
# HEART DISEASE BASELINE MODEL PIPELINE
# Load Train/Test → Scale → Train → Evaluate → Save Outputs
# ============================================================

import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

# ============================================================
# Step 1 — Load Train and Test Data
# ============================================================

train_file = "data/train_data.csv"
test_file = "data/test_data.csv"

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)


# ============================================================
# Step 2 — Separate Features and Target
# ============================================================

X_train = train_df.drop("target", axis=1)
y_train = train_df["target"]

X_test = test_df.drop("target", axis=1)
y_test = test_df["target"]

print("\nFeature columns:")
print(X_train.columns)


# ============================================================
# Step 3 — Feature Scaling
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

print("\nScaling completed")


# ============================================================
# Step 4 — Train Logistic Regression Model
# ============================================================

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(
    X_train_scaled,
    y_train
)

print("\nModel training completed")


# ============================================================
# Step 5 — Make Predictions
# ============================================================

y_pred = model.predict(X_test_scaled)

y_prob = model.predict_proba(X_test_scaled)[:, 1]


# ============================================================
# Step 6 — Evaluate Model
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

roc_auc = roc_auc_score(y_test, y_prob)

print("\n===== MODEL PERFORMANCE =====")

print("Accuracy:", accuracy)

print("Precision:", precision)

print("Recall:", recall)

print("F1 Score:", f1)

print("ROC-AUC:", roc_auc)

print("\nConfusion Matrix:")

print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")

print(classification_report(y_test, y_pred))


# ============================================================
# Step 7 — Save Model and Scaler
# ============================================================

joblib.dump(
    model,
    "models/ogistic_regression.pkl"
)

joblib.dump(
    scaler,
    "scaler/lr_scaler.pkl"
)

print("\nModel saved: baseline_model.pkl")

print("Scaler saved: scaler.pkl")


# ============================================================
# Step 8 — Save Predictions File
# ============================================================

results_df = pd.DataFrame({

    "Actual": y_test,
    "Predicted": y_pred,
    "Probability": y_prob

})

results_df.to_csv(
    "predictions/lr.csv",
    index=False
)

print("\nPredictions saved: test_predictions.csv")


# ============================================================
# Step 9 — Feature Importance (Logistic Regression)
# ============================================================

feature_importance = pd.DataFrame({

    "Feature": X_train.columns,
    "Coefficient": model.coef_[0]

})

feature_importance = feature_importance.sort_values(
    by="Coefficient",
    ascending=False
)

feature_importance.to_csv(
    "feature_importance/lr.csv",
    index=False
)

print("\nFeature importance saved: feature_importance.csv")


# ============================================================
# Step 10 — Final Status
# ============================================================

print("\nPipeline completed successfully")