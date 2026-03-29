# ============================================
# HEART DISEASE DATA PIPELINE
# Load → Validate → Clean → Split → Save
# ============================================

import pandas as pd
from sklearn.model_selection import train_test_split

# ============================================
# Step 1 — Load Dataset
# ============================================

file_path = "data/processed/heart_disease_combined.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully")
print("Shape:", df.shape)


# ============================================
# Step 2 — Basic Dataset Validation
# ============================================

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nTarget distribution:")
print(df["target"].value_counts())


# ============================================
# Step 3 — Remove Duplicate Records
# ============================================

df = df.drop_duplicates()

print("\nShape after removing duplicates:")
print(df.shape)


# ============================================
# Step 4 — Handle Missing Values
# ============================================

df.fillna(
    df.median(numeric_only=True),
    inplace=True
)

print("\nMissing values after cleaning:")
print(df.isnull().sum())


# ============================================
# Step 5 — Ensure Target is Binary
# ============================================

df["target"] = df["target"].apply(
    lambda x: 1 if x > 0 else 0
)


# ============================================
# Step 6 — Separate Features and Target
# ============================================

X = df.drop("target", axis=1)

y = df["target"]

print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)


# ============================================
# Step 7 — Train/Test Split
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain/Test split completed")

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)


# ============================================
# Step 8 — Combine Features and Target Again
# (for saving as full datasets)
# ============================================

train_df = X_train.copy()
train_df["target"] = y_train

test_df = X_test.copy()
test_df["target"] = y_test


# ============================================
# Step 9 — Save Train and Test Files
# ============================================

train_file = "data/train_data.csv"
test_file = "data/test_data.csv"

train_df.to_csv(
    train_file,
    index=False
)

test_df.to_csv(
    test_file,
    index=False
)

print("\nFiles saved successfully")

print("Train file:", train_file)
print("Test file:", test_file)