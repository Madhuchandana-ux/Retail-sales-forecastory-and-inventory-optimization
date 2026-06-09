import joblib
import numpy as np
import pandas as pd


def get_processed_data():

    return joblib.load(
        "artifacts/processed_features.pkl"
    )


def build_features_for_item(
    item_id,
    feature_names
):

    df = get_processed_data()

    item_df = (
        df[df["id"] == item_id]
        .copy()
    )

    if item_df.empty:
        raise ValueError(
            f"Item {item_id} not found"
        )

    latest_row = (
        item_df
        .sort_values("date")
        .iloc[-1:]
    )

    missing = [
        col
        for col in feature_names
        if col not in latest_row.columns
    ]

    if missing:
        raise ValueError(
            f"Missing features: {missing}"
        )

    return latest_row[
        feature_names
    ]


def load_model():

    model_data = joblib.load(
        "models/xgboost.pkl"
    )

    return (
        model_data["model"],
        model_data["features"]
    )


def predict_item_demand(
    item_id
):

    model, feature_names = (
        load_model()
    )

    X = build_features_for_item(
        item_id,
        feature_names
    )

    prediction = model.predict(X)

    return float(
        prediction[0]
    )


def predict_28_day_forecast(
    item_id
):

    model, feature_names = (
        load_model()
    )

    df = get_processed_data()

    item_df = (
        df[df["id"] == item_id]
        .copy()
        .sort_values("date")
    )

    if item_df.empty:

        raise ValueError(
            f"{item_id} not found"
        )

    forecasts = []

    history = (
        item_df["sales"]
        .tolist()
    )

    latest_row = (
        item_df.iloc[-1]
        .copy()
    )

    for day in range(28):

        row = latest_row.copy()

        row["lag_1"] = history[-1]

        row["lag_7"] = np.mean(
            history[-7:]
        )

        row["lag_14"] = np.mean(
            history[-14:]
        )

        row["lag_28"] = np.mean(
            history[-28:]
        )

        row["lag_56"] = np.mean(
            history[-56:]
            if len(history) >= 56
            else history
        )

        row["rolling_mean_7"] = np.mean(
            history[-7:]
        )

        row["rolling_mean_14"] = np.mean(
            history[-14:]
        )

        row["rolling_mean_28"] = np.mean(
            history[-28:]
        )

        row["rolling_std_7"] = np.std(
            history[-7:]
        )

        row["rolling_std_14"] = np.std(
            history[-14:]
        )

        row["rolling_std_28"] = np.std(
            history[-28:]
        )

        X = pd.DataFrame(
            [row]
        )[feature_names]

        pred = float(
            model.predict(X)[0]
        )

        forecasts.append(pred)

        history.append(pred)

    return np.array(
        forecasts
    )