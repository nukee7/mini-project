# ==========================================================
# LIME EXPLAINABILITY FOR PYTORCH DNN — PNG + CSV
# ==========================================================

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from lime.lime_tabular import LimeTabularExplainer

# ==========================================================
# Configuration
# ==========================================================

MODEL_NAME = "dnn_model"

MODEL_PATH = f"models/{MODEL_NAME}.pth"

DATA_PATH = "data/test_data.csv"

output_dir = f"lime_explainiblity_output/{MODEL_NAME}"

os.makedirs(output_dir, exist_ok=True)

# ==========================================================
# Define model architecture (must match training)
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
# Load data
# ==========================================================

test_df = pd.read_csv(DATA_PATH)

X_test = test_df.drop("target", axis=1)

y_test = test_df["target"]

feature_names = list(X_test.columns)

# ==========================================================
# Scaling (same as your training)
# ==========================================================

scaler = StandardScaler()

X_test_scaled = scaler.fit_transform(X_test)

# ==========================================================
# Load PyTorch model
# ==========================================================

input_size = X_test.shape[1]

model = DNNModel(input_size)

model.load_state_dict(

    torch.load(
        MODEL_PATH,
        map_location=torch.device("cpu")
    )
)

model.eval()

print("Model loaded successfully")

# ==========================================================
# Prediction function for LIME
# ==========================================================

def predict_fn(data):

    tensor_data = torch.tensor(

        data,
        dtype=torch.float32
    )

    with torch.no_grad():

        outputs = model(tensor_data)

    probabilities = outputs.numpy()

    return np.hstack([

        1 - probabilities,
        probabilities

    ])

# ==========================================================
# Create LIME explainer
# ==========================================================

explainer = LimeTabularExplainer(

    training_data=X_test_scaled,

    feature_names=feature_names,

    class_names=["No Disease", "Disease"],

    mode="classification"
)

# ==========================================================
# Select instance
# ==========================================================

index = 0

instance = X_test_scaled[index]

# ==========================================================
# Generate explanation
# ==========================================================

explanation = explainer.explain_instance(

    data_row=instance,

    predict_fn=predict_fn,

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

print("\nLIME pipeline completed successfully")