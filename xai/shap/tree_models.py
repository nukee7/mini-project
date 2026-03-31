# ==========================================================
# SHAP FOR ALL TREE MODELS — LOOP VERSION
# ==========================================================

import os
import pandas as pd
import shap
import joblib
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# List of tree models
# ==========================================================

TREE_MODELS = [

    "decision_tree",
    "random_forest",
    "xgboost"

]

# ==========================================================
# Load test data once
# ==========================================================

test_df = pd.read_csv("data/test_data.csv")

X_test = test_df.drop("target", axis=1)

feature_names = X_test.columns

# ==========================================================
# Loop through models
# ==========================================================

for MODEL_NAME in TREE_MODELS:

    print("\n==============================")
    print("Processing model:", MODEL_NAME)

    model_path = f"models/{MODEL_NAME}.pkl"

    output_dir = f"explainability_outputs/{MODEL_NAME}"

    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------
    # Load model
    # ------------------------------------------------------

    model = joblib.load(model_path)

    # ------------------------------------------------------
    # SHAP explainer
    # ------------------------------------------------------

    explainer = shap.TreeExplainer(model)

    print("Computing SHAP values...")

    shap_values = explainer.shap_values(X_test)

    # ------------------------------------------------------
    # Handle binary classification output
    # ------------------------------------------------------

    if isinstance(shap_values, list):

        shap_array = shap_values[1]

    else:

        shap_array = np.array(shap_values)

        if shap_array.ndim == 3:

            shap_array = shap_array[:, :, 1]

    # ------------------------------------------------------
    # SAVE SHAP VALUES CSV
    # ------------------------------------------------------

    shap_df = pd.DataFrame(

        shap_array,
        columns=feature_names
    )

    shap_df.to_csv(

        f"{output_dir}/shap_values.csv",
        index=False
    )

    print("Saved: shap_values.csv")

    # ------------------------------------------------------
    # SAVE FEATURE IMPORTANCE CSV
    # ------------------------------------------------------

    importance = np.abs(shap_array).mean(axis=0)

    importance_df = pd.DataFrame({

        "Feature": feature_names,
        "Mean_SHAP_Importance": importance

    }).sort_values(

        by="Mean_SHAP_Importance",
        ascending=False
    )

    importance_df.to_csv(

        f"{output_dir}/feature_importance.csv",
        index=False
    )

    print("Saved: feature_importance.csv")

    # ------------------------------------------------------
    # SUMMARY PLOT
    # ------------------------------------------------------

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

    # ------------------------------------------------------
    # BAR PLOT
    # ------------------------------------------------------

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

print("\nAll tree model SHAP processing completed")