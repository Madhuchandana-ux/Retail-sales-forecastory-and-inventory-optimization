import numpy as np

def generate_ensemble_forecast(
    pred_xgb,
    pred_lgb,
    pred_tft,
    weights=None
):

    if weights is None:
        weights = [0.4, 0.3, 0.3]

    if not np.isclose(sum(weights), 1.0):
        raise ValueError(
            "Weights must sum to 1"
        )

    forecast = (
        weights[0] * pred_xgb +
        weights[1] * pred_lgb +
        weights[2] * pred_tft
    )

    return forecast