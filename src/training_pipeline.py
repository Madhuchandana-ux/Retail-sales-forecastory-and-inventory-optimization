import pandas as pd

from src.data_ingestion import load_data
from src.feature_engineering import create_features

from src.train_ml import (
    train_xgboost,
    train_lightgbm
)

# ------------------------------------
# Load Data
# ------------------------------------

print("Loading data...")

df = load_data()

print("Rows:", len(df))

# ------------------------------------
# Feature Engineering
# ------------------------------------

print("Creating features...")

df = create_features(df)

# ------------------------------------
# Remove rows with lag NaNs
# ------------------------------------

feature_cols_to_clean = [

    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "lag_56",

    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",

    "rolling_std_7",
    "rolling_std_14",
    "rolling_std_28"
]

df = df.dropna(
    subset=feature_cols_to_clean
)

print("Rows after cleaning:", len(df))

# ------------------------------------
# Feature Columns
# ------------------------------------

feature_cols = [

    "dayofweek",
    "is_weekend",
    
    "is_holiday",
    "month",
"quarter",
"weekofyear",

    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "lag_56",

    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",

    "rolling_std_7",
    "rolling_std_14",
    "rolling_std_28",

    "price_change",
    "discount_pct"

]

# ------------------------------------
# Target
# ------------------------------------

target_col = "sales"

X = df[feature_cols]

y = df[target_col]

# ------------------------------------
# Train Validation Split
# ------------------------------------

df = df.sort_values("date")

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_val = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_val = y.iloc[split_index:]

print("Training XGBoost...")

train_xgboost(
    X_train,
    y_train,
    X_val,
    y_val
)

print("Training LightGBM...")

train_lightgbm(
    X_train,
    y_train,
    X_val,
    y_val
)

print("Training Complete")