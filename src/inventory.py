import numpy as np
from scipy.stats import norm


def calculate_safety_stock(
    std_dev,
    lead_time,
    service_level=0.95
):

    if lead_time <= 0:
        raise ValueError(
            "lead_time must be positive"
        )

    if not 0 < service_level < 1:
        raise ValueError(
            "service_level must be between 0 and 1"
        )

    z = norm.ppf(service_level)

    return (
        z *
        std_dev *
        np.sqrt(lead_time)
    )


def calculate_reorder_point(
    demand_lt,
    safety_stock
):
    return demand_lt + safety_stock


def calculate_eoq(
    annual_demand,
    order_cost,
    holding_cost
):

    if holding_cost <= 0:
        raise ValueError(
            "holding_cost must be > 0"
        )

    return np.sqrt(
        (2 * annual_demand * order_cost)
        / holding_cost
    )


def simulate_scenario(
    forecast,
    lead_time,
    demand_growth=1.0,
    service_level=0.95
):

    adjusted = forecast * demand_growth

    mean_demand = adjusted.mean()

    demand_std = adjusted.std()

    safety_stock = calculate_safety_stock(
        demand_std,
        lead_time,
        service_level
    )

    reorder_point = calculate_reorder_point(
        mean_demand * lead_time,
        safety_stock
    )

    return {
        "lead_time": lead_time,
        "service_level": service_level,
        "adjusted_forecast_mean": round(mean_demand, 2),
        "safety_stock": round(safety_stock, 2),
        "reorder_point": round(reorder_point, 2)
    }