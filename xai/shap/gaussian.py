# ==========================================================
# SHAP FOR GAUSSIAN NAIVE BAYES (ROBUST VERSION)
# ==========================================================

import os
import pandas as pd
import shap
import joblib
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# Create output directory
# ==========================================================

output_dir = "explainability_outputs/gaussian_nb"

os.makedirs(output_dir, exist_ok=True)

# ==========================================================
# Load model and scaler
# ==========================================================

model = joblib.load("models/gaussian_nb.pkl")

scaler = joblib.load("scaler/gaussian_scaler.pkl")

# ==========================================================
# Load test data
# ==========================================================

test_df = pd.read_csv("data/test_data.csv")

X_test = test_df.drop("target", axis=1)

feature_names = X_test.columns

# ==========================================================
# Scale data
# ==========================================================

X_test_scaled = scaler.transform(X_test)

# ==========================================================
# Background data (for KernelExplainer)
# ==========================================================

background = X_test_scaled[:100]

# ==========================================================
# Create SHAP explainer
# ==========================================================

explainer = shap.KernelExplainer(

    model.predict_proba,
    background
)

print("Computing SHAP values...")

# ==========================================================
# Compute SHAP values
# ==========================================================

shap_values = explainer.shap_values(

    X_test_scaled[:100]
)

# ==========================================================
# Normalize SHAP output shape
# ==========================================================

if isinstance(shap_values, list):

    shap_array = shap_values[1]

else:

    shap_array = np.array(shap_values)

    if shap_array.ndim == 3:

        shap_array = shap_array[:, :, 1]

# ==========================================================
# Diagnostic check (optional)
# ==========================================================

print("SHAP shape:", shap_array.shape)

print("Data shape:", X_test.iloc[:100].shape)

# ==========================================================
# Summary plot (global explanation)
# ==========================================================

plt.figure()

shap.summary_plot(

    shap_array,
    X_test.iloc[:100],

    show=False
)

plt.savefig(

    f"{output_dir}/shap_summary.png",

    bbox_inches="tight"
)

plt.close()

print("Saved: shap_summary.png")

# ==========================================================
# Bar plot (feature importance ranking)
# ==========================================================

plt.figure()

shap.summary_plot(

    shap_array,
    X_test.iloc[:100],

    plot_type="bar",

    show=False
)

plt.savefig(

    f"{output_dir}/shap_bar.png",

    bbox_inches="tight"
)

plt.close()

print("Saved: shap_bar.png")

print("\nGaussian Naive Bayes SHAP completed successfully")