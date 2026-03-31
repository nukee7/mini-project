# ==========================================================
# SHAP FOR LOGISTIC REGRESSION — PNG + CSV OUTPUTS
# ==========================================================

import os
import pandas as pd
import joblib
import shap
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================

output_dir = "explainability_outputs/lr"

os.makedirs(output_dir, exist_ok=True)

# ==========================================================
# Load model and scaler
# ==========================================================

model = joblib.load("models/logistic_regression.pkl")

scaler = joblib.load("scaler/lr_scaler.pkl")

# ==========================================================
# Load data
# ==========================================================

test_df = pd.read_csv("data/test_data.csv")

X_test = test_df.drop("target", axis=1)

X_test_scaled = scaler.transform(X_test)

# ==========================================================
# SHAP
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
# Convert SHAP to array
# ==========================================================

shap_array = shap_values.values

# ==========================================================
# SAVE SHAP VALUES CSV
# ==========================================================

shap_df = pd.DataFrame(

    shap_array,
    columns=X_test.columns
)

shap_df.to_csv(

    f"{output_dir}/lr_shap_values.csv",
    index=False
)

print("Saved: lr_shap_values.csv")

# ==========================================================
# SAVE FEATURE IMPORTANCE CSV
# ==========================================================

importance = np.abs(shap_array).mean(axis=0)

importance_df = pd.DataFrame({

    "Feature": X_test.columns,
    "Mean_SHAP_Importance": importance

}).sort_values(

    by="Mean_SHAP_Importance",
    ascending=False
)

importance_df.to_csv(

    f"{output_dir}/lr_feature_importance.csv",
    index=False
)

print("Saved: feature_importance.csv")

# ==========================================================
# SUMMARY PLOT
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

# ==========================================================
# BAR PLOT
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

print("PNG plots saved")

print("\nLogistic Regression SHAP completed")