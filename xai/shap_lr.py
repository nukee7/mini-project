# ==========================================================
# SHAP EXPLAINABILITY FOR LOGISTIC REGRESSION
# ==========================================================

import os
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# ==========================================================
# Create output directory
# ==========================================================

os.makedirs("explainability_outputs", exist_ok=True)

# ==========================================================
# Load Model and Scaler
# ==========================================================

model = joblib.load("models/logistic_regression.pkl")

scaler = joblib.load("scaler.pkl")

# ==========================================================
# Load Test Data
# ==========================================================

test_df = pd.read_csv("test_data.csv")

X_test = test_df.drop("target", axis=1)

y_test = test_df["target"]

# ==========================================================
# Scale Data
# ==========================================================

X_test_scaled = scaler.transform(X_test)

# ==========================================================
# Create SHAP Explainer
# ==========================================================

explainer = shap.Explainer(
    model,
    X_test_scaled
)

shap_values = explainer(
    X_test_scaled
)

print("SHAP values calculated")

# ==========================================================
# 1) Global Explanation — Summary Plot
# ==========================================================

plt.figure()

shap.summary_plot(
    shap_values,
    X_test,
    show=False
)

plt.savefig(
    "explainability_outputs/shap_summary_plot.png",
    bbox_inches="tight"
)

print("Saved: shap_summary_plot.png")

# ==========================================================
# 2) Global Explanation — Bar Plot
# ==========================================================

plt.figure()

shap.plots.bar(
    shap_values,
    show=False
)

plt.savefig(
    "explainability_outputs/shap_bar_plot.png",
    bbox_inches="tight"
)

print("Saved: shap_bar_plot.png")

# ==========================================================
# 3) Local Explanation — Single Prediction
# ==========================================================

index = 0

plt.figure()

shap.plots.waterfall(
    shap_values[index],
    show=False
)

plt.savefig(
    "explainability_outputs/shap_local_explanation.png",
    bbox_inches="tight"
)

print("Saved: shap_local_explanation.png")

print("\nExplainability pipeline completed")