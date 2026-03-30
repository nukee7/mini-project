import os
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

os.makedirs("explainability_outputs/xgboost", exist_ok=True)

model = joblib.load("models/xgboost.pkl")

test_df = pd.read_csv("data/test_data.csv")

X_test = test_df.drop("target", axis=1)

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_test)

plt.figure()

shap.summary_plot(
    shap_values,
    X_test,
    show=False
)

plt.savefig(
    "explainability_outputs/xgboost/shap_summary.png",
    bbox_inches="tight"
)