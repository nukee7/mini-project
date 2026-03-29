import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Load data
train_df = pd.read_csv("train_data.csv")
test_df = pd.read_csv("test_data.csv")

X_train = train_df.drop("target", axis=1)
y_train = train_df["target"]

X_test = test_df.drop("target", axis=1)
y_test = test_df["target"]

# Scaling required
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build model
model = Sequential()

model.add(Dense(64, activation="relu", input_dim=X_train.shape[1]))
model.add(Dense(32, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(

    optimizer=Adam(),

    loss="binary_crossentropy",

    metrics=["accuracy"]
)

print("Training DNN...")

model.fit(

    X_train_scaled,
    y_train,

    epochs=50,

    batch_size=32,

    validation_split=0.2,

    verbose=1
)

# Predict
y_prob = model.predict(X_test_scaled).flatten()

y_pred = (y_prob > 0.5).astype(int)

print("DNN")

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

model.save("models/dnn_model.h5")