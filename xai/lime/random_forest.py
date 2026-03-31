# ==========================================================
# LIME EXPLAINABILITY — SAVE PNG ONLY
# ==========================================================

import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from lime.lime_tabular import LimeTabularExplainer

# ==========================================================
# Configuration
# ==========================================================

MODEL_NAME = "random_forest"

MODEL_PATH = f"models/{MODEL_NAME}.pkl"

DATA_PATH = "data/test_data.csv"

# Required directory structure

output_dir = f"lime_explainiblity_output/{MODEL_NAME}"

os.makedirs(output_dir, exist_ok=True)

# ==========================================================
# Load model
# ==========================================================

model = joblib.load(MODEL_PATH)

# ==========================================================
# Load data
# ==========================================================

test_df = pd.read_csv(DATA_PATH)

X_test = test_df.drop("target", axis=1)

y_test = test_df["target"]

feature_names = list(X_test.columns)

# ==========================================================
# Create LIME explainer
# ==========================================================

explainer = LimeTabularExplainer(

    training_data=X_test.values,

    feature_names=feature_names,

    class_names=["No Disease", "Disease"],

    mode="classification"
)

# ==========================================================
# Select instance to explain
# ==========================================================

index = 0

instance = X_test.iloc[index].values

# ==========================================================
# Generate explanation
# ==========================================================

explanation = explainer.explain_instance(

    data_row=instance,

    predict_fn=model.predict_proba,

    num_features=len(feature_names)
)

print("LIME explanation generated")

# ==========================================================
# SAVE PNG
# ==========================================================

fig = explanation.as_pyplot_figure()

png_path = f"{output_dir}/lime_explanation.png"

fig.savefig(

    png_path,

    bbox_inches="tight"
)

plt.close(fig)

print("Saved:", png_path)

# ==========================================================
# SAVE FEATURE IMPORTANCE CSV
# ==========================================================

lime_data = explanation.as_list()

importance_df = pd.DataFrame(

    lime_data,

    columns=["Feature", "Weight"]

)

csv_path = f"{output_dir}/lime_feature_importance.csv"

importance_df.to_csv(

    csv_path,

    index=False
)

print("Saved:", csv_path)

print("\nLIME PNG pipeline completed successfully")