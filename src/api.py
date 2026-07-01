"""FastAPI Backend - Retail Sales Forecasting & Inventory Optimization API

Provides RESTful endpoints for demand forecasting, inventory optimization,
risk assessment, and scenario simulation.

Author: Madhu Chandana
Version: 2.0.0
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import logging
import time
from typing import List, Optional

from src.inventory import (
    simulate_scenario,
    calculate_reorder_point
)
from src.risk import assess_stockout_risk
from src.inference import (
    predict_28_day_forecast,
    build_features_for_item
)


# ==================================
# LOGGING CONFIGURATION
# ==================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==================================
# FASTAPI APPLICATION
# ==================================

app = FastAPI(
    title="Retail Forecasting & Inventory Optimization API",
    version="2.0.0",
    description="Enterprise-grade API for demand forecasting, inventory optimization, and supply chain planning",
    docs_url="/docs",
    redoc_url="/redoc"
)


# ==================================
# MODEL LOADING
# ==================================

MODEL = None
FEATURES = []

try:
    model_data = joblib.load("models/xgboost.pkl")
    MODEL = model_data.get("model")
    FEATURES = model_data.get("features", [])
    logger.info(f"✓ Model loaded successfully with {len(FEATURES)} features")
except Exception as e:
    logger.error(f"✗ Model loading failed: {e}")
    MODEL = None
    FEATURES = []


# ==================================
# STARTUP EVENT
# ==================================

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    if MODEL is None:
        logger.warning("⚠️ Warning: Model not loaded. API functionality limited.")
    else:
        logger.info("✓ Retail Forecasting API Ready")


# ==================================
# REQUEST MODELS
# ==================================

class SimulateRequest(BaseModel):
    """Request model for scenario simulation."""
    item_id: str = Field(..., description="Product SKU identifier")
    lead_time: int = Field(default=7, gt=0, le=365, description="Lead time in days")
    demand_growth: float = Field(default=1.0, gt=0, description="Demand growth multiplier")
    service_level: float = Field(
        default=0.95,
        gt=0,
        lt=1,
        description="Target service level (0-1)"
    )


class RiskAssessmentRequest(BaseModel):
    """Request model for risk assessment."""
    item_id: str = Field(..., description="Product SKU identifier")
    current_inventory: int = Field(..., ge=0, description="Current inventory level")


# ==================================
# RESPONSE MODELS
# ==================================

class ForecastResponse(BaseModel):
    """Response model for forecast endpoint."""
    item: str
    forecast_28d: List[float]
    mean_forecast: float
    total_forecast: float


class RiskResponse(BaseModel):
    """Response model for risk assessment endpoint."""
    item: str
    stockout_risk: str
    reorder_point: float
    days_to_stockout: float
    health_score: float


class SimulationResponse(BaseModel):
    """Response model for scenario simulation endpoint."""
    item: str
    revenue: float
    net_profit: float
    safety_stock: float
    reorder_point: float
    roi_percent: float


# ==================================
# HEALTH & METADATA ENDPOINTS
# ==================================

@app.get("/health")
async def health():
    """Health check endpoint.
    
    Returns:
        dict: System status and model availability
    """
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "timestamp": time.time()
    }


@app.get("/version")
async def version():
    """API version information.
    
    Returns:
        dict: API and model version details
    """
    return {
        "api_version": "2.0.0",
        "model_type": "XGBoost + LightGBM",
        "features_available": len(FEATURES)
    }


@app.get("/model-info")
async def model_info():
    """Get model information.
    
    Returns:
        dict: Model status and feature list
    """
    return {
        "model_loaded": MODEL is not None,
        "feature_count": len(FEATURES),
        "features": FEATURES[:10]  # Return first 10 for brevity
    }


# ==================================
# FORECAST ENDPOINT
# ==================================

@app.get("/forecast/{item_id}", response_model=ForecastResponse)
async def forecast(item_id: str):
    """Generate 28-day demand forecast for item.
    
    Args:
        item_id: Product SKU identifier
        
    Returns:
        ForecastResponse: 28-day forecast values
        
    Raises:
        HTTPException: If forecast generation fails
    """
    try:
        start_time = time.time()
        
        forecast_values = predict_28_day_forecast(item_id)
        latency = (time.time() - start_time) * 1000
        
        logger.info(f"✓ Forecast generated for {item_id} in {latency:.2f}ms")
        
        return {
            "item": item_id,
            "forecast_28d": forecast_values.tolist(),
            "mean_forecast": float(forecast_values.mean()),
            "total_forecast": float(forecast_values.sum())
        }
    
    except Exception as e:
        logger.error(f"✗ Forecast generation failed for {item_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Forecast generation failed: {str(e)}"
        )


# ==================================
# SIMULATION ENDPOINT
# ==================================

@app.post("/simulate", response_model=SimulationResponse)
async def simulate(request: SimulateRequest):
    """Run scenario simulation with demand and supply variations.
    
    Args:
        request: Simulation parameters (item_id, lead_time, demand_growth, service_level)
        
    Returns:
        SimulationResponse: Financial and inventory metrics
        
    Raises:
        HTTPException: If simulation fails
    """
    try:
        start_time = time.time()
        
        forecast = predict_28_day_forecast(request.item_id)
        result = simulate_scenario(
            forecast=forecast,
            lead_time=request.lead_time,
            demand_growth=request.demand_growth,
            service_level=request.service_level
        )
        
        latency = (time.time() - start_time) * 1000
        logger.info(f"✓ Simulation completed in {latency:.2f}ms")
        
        return {
            "item": request.item_id,
            "revenue": result.get("revenue", 0),
            "net_profit": result.get("net_profit", 0),
            "safety_stock": result.get("safety_stock", 0),
            "reorder_point": result.get("reorder_point", 0),
            "roi_percent": result.get("roi", 0)
        }
    
    except Exception as e:
        logger.error(f"✗ Simulation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Simulation failed: {str(e)}"
        )


# ==================================
# RISK ASSESSMENT ENDPOINT
# ==================================

@app.post("/risk", response_model=RiskResponse)
async def assess_risk(request: RiskAssessmentRequest):
    """Assess stockout risk for an item.
    
    Args:
        request: Item ID and current inventory level
        
    Returns:
        RiskResponse: Risk assessment with metrics
        
    Raises:
        HTTPException: If assessment fails
    """
    try:
        if request.current_inventory < 0:
            raise ValueError("Inventory cannot be negative")
        
        forecast = predict_28_day_forecast(request.item_id)
        avg_daily_demand = forecast.mean()
        safety_stock = forecast.std() * 1.65
        reorder_point = calculate_reorder_point(
            demand_lt=avg_daily_demand * 7,
            safety_stock=safety_stock
        )
        
        risk_level = assess_stockout_risk(
            request.current_inventory,
            reorder_point
        )
        
        days_to_stockout = (
            request.current_inventory / avg_daily_demand
            if avg_daily_demand > 0
            else float('inf')
        )
        
        health_score = min(
            100,
            max(0, (request.current_inventory / reorder_point) * 100)
        )
        
        logger.info(f"✓ Risk assessment completed for {request.item_id}: {risk_level}")
        
        return {
            "item": request.item_id,
            "stockout_risk": risk_level,
            "reorder_point": round(reorder_point, 2),
            "days_to_stockout": days_to_stockout,
            "health_score": health_score
        }
    
    except Exception as e:
        logger.error(f"✗ Risk assessment failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Risk assessment failed: {str(e)}"
        )


# ==================================
# ERROR HANDLERS
# ==================================

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle validation errors."""
    return {
        "error": "Validation Error",
        "message": str(exc)
    }
