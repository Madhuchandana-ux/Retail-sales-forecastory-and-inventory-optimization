from src.data_ingestion import load_data
from src.feature_engineering import create_features

import joblib
import os

os.makedirs("artifacts", exist_ok=True)

print("Loading data...")

df = load_data()

print("Creating features...")

df = create_features(df)

df = df.dropna()

joblib.dump(
    df,
    "artifacts/processed_features.pkl"
)

print("Saved processed dataset")
print(df.shape)