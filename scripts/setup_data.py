import urllib.request
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
import os

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PARTITIONS_DIR = DATA_DIR / "partitions"

HEADERS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

def download_raw_data():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PARTITIONS_DIR.mkdir(parents=True, exist_ok=True)
    
    urls = {
        "Cleveland": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
        "Hungary": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data",
        "Switzerland": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data",
        "Long_Beach_VA": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.va.data"
    }

    for hospital, url in urls.items():
        local_file = RAW_DIR / f"{hospital}.data"
        if not local_file.exists():
            print(f"Downloading raw data for {hospital} from {url}...")
            urllib.request.urlretrieve(url, local_file)
        else:
            print(f"Verified cached raw data for geographic node: {hospital}")

def clean_and_impute():
    """
    Ethical Data Prep:
    The Swiss and Long Beach geographical nodes are notoriously sparse mathematically.
    Dropping (Standard) missing values disproportionately discriminates against these 
    target demographics, introducing severe bias against underserved regions.
    
    We employ Iterative Imputation locally per node. It recursively models each feature 
    as a function of other features, predicting and retaining the minority records 
    securely before Federated aggregation starts.
    """
    imputer = IterativeImputer(max_iter=15, random_state=42)
    scaler = StandardScaler()
    
    for file_path in RAW_DIR.glob("*.data"):
        hospital = file_path.stem
        df = pd.read_csv(file_path, names=HEADERS, na_values='?')
        
        # Binarize targets
        df['target'] = df['target'].apply(lambda x: 1 if float(x) > 0 else 0)
        
        target = df['target']
        features = df.drop('target', axis=1)
        
        # Local, ethical reconstruction of sparse minority properties
        imputed_features = pd.DataFrame(imputer.fit_transform(features), columns=features.columns)
        scaled_features = pd.DataFrame(scaler.fit_transform(imputed_features), columns=features.columns)
        
        scaled_features['target'] = target.values
        out_path = PARTITIONS_DIR / f"{hospital}.csv"
        scaled_features.to_csv(out_path, index=False)
        print(f"Imputed -> Scaled -> Saved Partition for Hospital: {hospital}")

def execute():
    download_raw_data()
    clean_and_impute()

if __name__ == "__main__":
    execute()
