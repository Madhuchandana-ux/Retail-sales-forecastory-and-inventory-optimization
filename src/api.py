from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import joblib
import logging
import time

from src.inventory import (
    simulate_scenario,
    calculate_reorder_point
)

from src.risk import (
    assess_stockout_risk
)

from src.inference import (
    predict_28_day_forecast,
    build_features_for_item
)

# ----------------------------------
# Logging
# ----------------------------------

logging.basicConfig(
    level=logging.INFO
)

logger = logging.getLogger(__name__)

# ----------------------------------
# App
# ----------------------------------

app = FastAPI(
    title="Retail Forecast & Inventory Optimization API",
    version="1.0.0",
    description="Enterprise Retail Demand Forecasting, Inventory Optimization and Explainable AI Platform"
)

# ----------------------------------
# Load Model
# ----------------------------------

MODEL = None
FEATURES = []

try:

    model_data = joblib.load(
        "models/xgboost.pkl"
    )

    MODEL = model_data["model"]

    FEATURES = model_data["features"]

    logger.info(
        f"Loaded model with {len(FEATURES)} features"
    )

except Exception as e:

    logger.exception(
        f"Model loading failed: {e}"
    )

# ----------------------------------
# Startup
# ----------------------------------

@app.on_event("startup")
async def startup_event():

    if MODEL is None:

        logger.warning(
            "Model not loaded"
        )

    else:

        logger.info(
            "Retail Forecast API Ready"
        )

# ----------------------------------
# Request Models
# ----------------------------------

class SimulateRequest(BaseModel):

    item_id: str

    lead_time: int = Field(
        gt=0,
        le=365
    )

    demand_growth: float = Field(
        default=1.0,
        gt=0
    )

    service_level: float = Field(
        default=0.95,
        gt=0,
        lt=1
    )

# ----------------------------------
# Response Models
# ----------------------------------

class ForecastResponse(BaseModel):

    item: str

    forecast_28d: list[float]

class RiskResponse(BaseModel):

    item: str

    stockout_risk: str

    reorder_point: float

# ----------------------------------
# Forecast Utility
# ----------------------------------

def get_item_forecast(
    item_id: str
):

    return predict_28_day_forecast(
        item_id
    )

# ----------------------------------
# Health
# ----------------------------------

@app.get("/health")
async def health():

    return {
        "status": "healthy",
        "model_loaded": MODEL is not None
    }

# ----------------------------------
# Version
# ----------------------------------

@app.get("/version")
async def version():

    return {
        "api_version": "2.0.0",
        "model_type": "XGBoost"
    }

# ----------------------------------
# Model Info
# ----------------------------------

@app.get("/model-info")
async def model_info():

    return {
        "model_loaded": MODEL is not None,
        "feature_count": len(FEATURES),
        "features": FEATURES
    }

# ----------------------------------
# Forecast Endpoint
# ----------------------------------

@app.get(
    "/forecast/{item_id}",
    response_model=ForecastResponse
)
async def forecast(
    item_id: str
):

    try:

        start_time = time.time()

        forecast_values = (
            get_item_forecast(
                item_id
            )
        )

        latency = (
            time.time()
            - start_time
        ) * 1000

        logger.info(
            f"{item_id} forecast generated in {latency:.2f} ms"
        )

        return {
            "item": item_id,
            "forecast_28d": forecast_values.tolist()
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# ----------------------------------
# Simulation Endpoint
# ----------------------------------

@app.post("/simulate")
async def simulate(
    request: SimulateRequest
):

    forecast = get_item_forecast(
        request.item_id
    )

    result = simulate_scenario(
        forecast=forecast,
        lead_time=request.lead_time,
        demand_growth=request.demand_growth,
        service_level=request.service_level
    )

    return result

# ----------------------------------
# Risk Endpoint
# ----------------------------------

@app.get(
    "/risk/{item_id}",
    response_model=RiskResponse
)
async def risk(
    item_id: str,
    current_inventory: int
):

    if current_inventory < 0:

        raise HTTPException(
            status_code=400,
            detail="Inventory cannot be negative"
        )

    forecast = get_item_forecast(
        item_id
    )

    avg_daily_demand = (
        forecast.mean()
    )

    safety_stock = (
        forecast.std()
        * 1.65
    )

    reorder_point = (
        calculate_reorder_point(
            demand_lt=avg_daily_demand * 7,
            safety_stock=safety_stock
        )
    )

    risk_level = (
        assess_stockout_risk(
            current_inventory,
            reorder_point
        )
    )

    return {
        "item": item_id,
        "stockout_risk": risk_level,
        "reorder_point": round(
            reorder_point,
            2
        )
    }