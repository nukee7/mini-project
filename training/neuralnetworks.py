# ==========================================================
# PYTORCH DNN TRAINING — HEART DISEASE CLASSIFICATION
# ==========================================================

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# ==========================================================
# Create models directory
# ==========================================================

os.makedirs("models", exist_ok=True)

# ==========================================================
# Load data
# ==========================================================

train_df = pd.read_csv("data/train_data.csv")
test_df = pd.read_csv("data/test_data.csv")

X_train = train_df.drop("target", axis=1)
y_train = train_df["target"]

X_test = test_df.drop("target", axis=1)
y_test = test_df["target"]

# ==========================================================
# Feature scaling
# ==========================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to tensors

X_train_tensor = torch.tensor(
    X_train_scaled,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train.values,
    dtype=torch.float32
).view(-1, 1)

X_test_tensor = torch.tensor(
    X_test_scaled,
    dtype=torch.float32
)

# ==========================================================
# Define neural network
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

# Initialize model

input_size = X_train.shape[1]

model = DNNModel(input_size)

# ==========================================================
# Loss and optimizer
# ==========================================================

criterion = nn.BCELoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

# ==========================================================
# Training loop
# ==========================================================

epochs = 50

print("Training PyTorch DNN...")

for epoch in range(epochs):

    model.train()

    outputs = model(X_train_tensor)

    loss = criterion(outputs, y_train_tensor)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if (epoch + 1) % 10 == 0:

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Loss: {loss.item():.4f}"
        )

# ==========================================================
# Evaluation
# ==========================================================

model.eval()

with torch.no_grad():

    probabilities = model(X_test_tensor)

    predictions = (
        probabilities > 0.5
    ).float()

# Convert to numpy

y_prob = probabilities.numpy().flatten()

y_pred = predictions.numpy().flatten()

# Metrics

print("\nDNN (PyTorch)")

print(
    "Accuracy:",
    accuracy_score(y_test, y_pred)
)

print(
    "Precision:",
    precision_score(y_test, y_pred)
)

print(
    "Recall:",
    recall_score(y_test, y_pred)
)

print(
    "F1 Score:",
    f1_score(y_test, y_pred)
)

print(
    "ROC-AUC:",
    roc_auc_score(y_test, y_prob)
)

# ==========================================================
# Save model
# ==========================================================

torch.save(

    model.state_dict(),

    "models/dnn_model.pth"
)

print("\nModel saved: models/dnn_torch_model.pth")