import pandas as pd
import numpy as np


def create_features(df):

    df = df.copy()

    # -------------------------
    # Time Features
    # -------------------------

    df["dayofweek"] = df["date"].dt.dayofweek

    df["is_weekend"] = (
        df["dayofweek"]
        .isin([5, 6])
        .astype(int)
    )

    df["month"] = (
        df["date"]
        .dt.month
    )

    df["quarter"] = (
        df["date"]
        .dt.quarter
    )

    df["weekofyear"] = (
        df["date"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    # -------------------------
    # Lag Features
    # -------------------------

    for lag in [1, 7, 14, 28, 56]:

        df[f"lag_{lag}"] = (
            df.groupby("id")["sales"]
            .shift(lag)
        )

    # -------------------------
    # Rolling Features
    # -------------------------

    for window in [7, 14, 28]:

        df[f"rolling_mean_{window}"] = (
            df.groupby("id")["sales"]
            .transform(
                lambda x:
                x.shift(1)
                .rolling(window, min_periods=1)
                .mean()
            )
        )

        df[f"rolling_std_{window}"] = (
            df.groupby("id")["sales"]
            .transform(
                lambda x:
                x.shift(1)
                .rolling(window, min_periods=1)
                .std()
            )
        )

    # -------------------------
    # Price Features
    # -------------------------

    df["price_change"] = (
        df.groupby("id")["sell_price"]
        .pct_change(fill_method=None)
    )

    df["discount_pct"] = (
        1 -
        (
            df["sell_price"]
            /
            df.groupby("item_id")["sell_price"]
            .transform("max")
        )
    )

    # -------------------------
    # Holiday Feature
    # -------------------------

    df["is_holiday"] = (
        df["event_name_1"]
        .notna()
        .astype(int)
    )

    return df