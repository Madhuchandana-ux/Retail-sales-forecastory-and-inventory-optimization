from sklearn.ensemble import IsolationForest
import numpy as np


def detect_anomalies(
    residuals,
    contamination=0.05
):

    if not 0 < contamination < 0.5:
        raise ValueError(
            "contamination must be between 0 and 0.5"
        )

    model = IsolationForest(
        contamination=contamination,
        random_state=42
    )

    predictions = model.fit_predict(
        residuals.reshape(-1, 1)
    )

    scores = model.decision_function(
        residuals.reshape(-1, 1)
    )

    return predictions == -1, scores


def assess_stockout_risk(
    current_inventory,
    reorder_point
):

    if current_inventory <= 0:
        return "Critical"

    elif current_inventory < reorder_point:
        return "High"

    elif current_inventory < reorder_point * 1.2:
        return "Medium"

    else:
        return "Low"