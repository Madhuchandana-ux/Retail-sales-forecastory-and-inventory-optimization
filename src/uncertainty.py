import numpy as np


def forecast_with_confidence(
    forecast
):

    std = np.std(
        forecast
    )

    lower = (
        forecast
        - 1.96 * std
    )

    upper = (
        forecast
        + 1.96 * std
    )

    return lower, upper