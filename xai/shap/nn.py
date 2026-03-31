# ==========================================================
# SHAP FOR PYTORCH DNN — PNG + CSV OUTPUTS
# ==========================================================

import os
import pandas as pd
import shap
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler

# ==========================================================
# Create output directory
# ==========================================================

output_dir = "explainability_outputs/dnn"

os.makedirs(output_dir, exist_ok=True)

# ==========================================================
# Define model architecture (must match training model)
# ==========================================================

class DNNModel(nn.Module):

    def __init__(self, input_size):

        super(DNNModel, self).__init__()

        self.network = nn.Sequential(

            nn.Linear(input_size, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        return self.network(x)

# ==========================================================
# Load test data
# ==========================================================

test_df = pd.read_csv("data/test_data.csv")

X_test = test_df.drop("target", axis=1)

feature_names = X_test.columns

# ==========================================================
# Scale data
# ==========================================================

scaler = StandardScaler()

X_test_scaled = scaler.fit_transform(X_test)

# Convert to tensor

X_tensor = torch.tensor(

    X_test_scaled,
    dtype=torch.float32
)

# ==========================================================
# Load trained model
# ==========================================================

input_size = X_test.shape[1]

model = DNNModel(input_size)

model.load_state_dict(

    torch.load(
        "models/dnn_model.pth",
        map_location=torch.device("cpu")
    )
)

model.eval()

# ==========================================================
# Wrapper function for SHAP
# ==========================================================

def predict_fn(data):

    tensor_data = torch.tensor(

        data,
        dtype=torch.float32
    )

    with torch.no_grad():

        outputs = model(tensor_data)

    return outputs.numpy()

# ==========================================================
# Create SHAP explainer
# ==========================================================

background = X_test_scaled[:100]

explainer = shap.KernelExplainer(

    predict_fn,
    background
)

print("Computing SHAP values...")

shap_values = explainer.shap_values(

    X_test_scaled[:100]
)

# ==========================================================
# Normalize SHAP output
# ==========================================================

if isinstance(shap_values, list):

    shap_array = shap_values[0]

else:

    shap_array = np.array(shap_values)

    if shap_array.ndim == 3:

        shap_array = shap_array[:, :, 0]

# ==========================================================
# SAVE SHAP VALUES CSV
# ==========================================================

shap_df = pd.DataFrame(

    shap_array,
    columns=feature_names
)

shap_df.to_csv(

    f"{output_dir}/shap_values.csv",

    index=False
)

print("Saved: shap_values.csv")

# ==========================================================
# SAVE FEATURE IMPORTANCE CSV
# ==========================================================

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

# ==========================================================
# SUMMARY PLOT
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
# BAR PLOT
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

print("\nPyTorch DNN SHAP pipeline completed successfully")