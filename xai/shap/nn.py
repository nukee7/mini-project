import os
import pandas as pd
import shap
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

os.makedirs("explainability_outputs/dnn", exist_ok=True)

model = load_model("models/dnn_model.h5")

test_df = pd.read_csv("test_data.csv")

X_test = test_df.drop("target", axis=1)

scaler = StandardScaler()

X_test_scaled = scaler.fit_transform(X_test)

explainer = shap.KernelExplainer(
    model.predict,
    X_test_scaled[:100]
)

shap_values = explainer.shap_values(
    X_test_scaled[:100]
)

plt.figure()

shap.summary_plot(
    shap_values,
    X_test.iloc[:100],
    show=False
)

plt.savefig(
    "explainability_outputs/dnn/shap_summary.png",
    bbox_inches="tight"
)