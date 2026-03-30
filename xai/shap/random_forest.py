# ==========================================================
# SHAP FOR RANDOM FOREST — FULL WORKING VERSION
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

output_dir = "explainability_outputs/random_forest"

os.makedirs(output_dir, exist_ok=True)

# ==========================================================
# Load trained model
# ==========================================================

model = joblib.load("models/random_forest.pkl")

# ==========================================================
# Load test data
# ==========================================================

test_df = pd.read_csv("data/test_data.csv")

X_test = test_df.drop("target", axis=1)

feature_names = X_test.columns

# ==========================================================
# Create SHAP Tree Explainer
# ==========================================================

explainer = shap.TreeExplainer(model)

print("Computing SHAP values...")

shap_values = explainer.shap_values(X_test)

# ==========================================================
# Handle binary classification output
# ==========================================================

if isinstance(shap_values, list):

    shap_array = shap_values[1]

else:

    shap_array = np.array(shap_values)

    if shap_array.ndim == 3:

        shap_array = shap_array[:, :, 1]

# ==========================================================
# Diagnostic check
# ==========================================================

print("SHAP shape:", shap_array.shape)

print("Data shape:", X_test.shape)

# ==========================================================
# 1) SUMMARY PLOT (GLOBAL EXPLANATION)
# ==========================================================

plt.figure()

shap.summary_plot(

    shap_array,
    X_test,

    show=False
)

plt.savefig(

    f"{output_dir}/shap_summary.png",

    bbox_inches="tight"
)

plt.close()

print("Saved: shap_summary.png")

# ==========================================================
# 2) BAR PLOT (FEATURE IMPORTANCE)
# ==========================================================

plt.figure()

shap.summary_plot(

    shap_array,
    X_test,

    plot_type="bar",

    show=False
)

plt.savefig(

    f"{output_dir}/shap_bar.png",

    bbox_inches="tight"
)

plt.close()

print("Saved: shap_bar.png")

# ==========================================================
# 3) LOCAL EXPLANATION (WATERFALL)
# ==========================================================

index = 0

# Ensure scalar base value

if isinstance(explainer.expected_value, (list, np.ndarray)):

    base_value = explainer.expected_value[1]

else:

    base_value = explainer.expected_value

# Create SHAP explanation object

explanation = shap.Explanation(

    values=shap_array[index],

    base_values=base_value,

    data=X_test.iloc[index],

    feature_names=feature_names
)

plt.figure()

shap.plots.waterfall(

    explanation,

    show=False
)

plt.savefig(

    f"{output_dir}/shap_local_explanation.png",

    bbox_inches="tight"
)

plt.close()

print("Saved: shap_local_explanation.png")

print("\nRandom Forest SHAP completed successfully")