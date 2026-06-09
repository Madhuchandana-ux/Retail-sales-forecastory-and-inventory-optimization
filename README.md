# Retail Sales Forecasting & Inventory Optimization

## Overview

Retail Sales Forecasting & Inventory Optimization is an end-to-end machine learning project designed to predict future product demand, optimize inventory decisions, assess stockout risks, and provide explainable AI insights for retail operations.

The project combines demand forecasting, inventory management, risk analysis, model explainability, experiment tracking, API deployment, and containerization into a single production-oriented system.

---

## Key Features

### Demand Forecasting

* 28-day demand forecasting pipeline
* Machine learning-based prediction system
* SKU-level forecasting support
* Forecast confidence interval estimation

### Explainable AI

* SHAP-based feature attribution
* Local feature importance analysis
* Model transparency and interpretability

### Inventory Optimization

* Economic Order Quantity (EOQ) calculation
* Reorder Point estimation
* Safety Stock computation
* Inventory planning recommendations

### Risk Assessment

* Stockout risk detection
* Inventory health scoring
* Emergency replenishment recommendations
* Expected stockout timeline estimation

### Scenario Simulation

* Demand growth simulations
* Lead time impact analysis
* Service level optimization
* Revenue and profit forecasting
* Supply chain sensitivity testing

### API Layer

* FastAPI-based serving infrastructure
* Forecast endpoints
* Inventory simulation endpoints
* Risk assessment endpoints
* Health monitoring endpoints

### Experiment Tracking

* MLflow experiment tracking
* Model versioning
* Artifact management

### Deployment

* Docker containerization
* Reproducible environments
* Production-ready packaging

---

## Dataset

This project uses retail demand data consisting of:

* Calendar information
* Historical sales records
* Product pricing data

Files:

* `calendar.csv`
* `sales_train_validation.csv`
* `sell_prices.csv`

---

## Project Architecture

```text
Retail Forecasting System
│
├── Data Ingestion
├── Data Validation
├── Feature Engineering
├── Model Training
├── Forecast Generation
├── Explainability (SHAP)
├── Inventory Optimization
├── Risk Assessment
├── Scenario Simulation
├── FastAPI Services
└── Streamlit Dashboard
```

---

## Project Structure

```text
.
│   app.py
│   requirements.txt
│   mlflow.db
│
├── artifacts
│   └── processed_features.pkl
│
├── data
│   ├── calendar.csv
│   ├── sales_train_validation.csv
│   └── sell_prices.csv
│
├── docker
│   ├── dockerfile
│   └── .dockerignore
│
├── mlruns
│
├── models
│   ├── lightgbm.pkl
│   └── xgboost.pkl
│
├── notebooks
│
└── src
    ├── api.py
    ├── data_ingestion.py
    ├── data_validation.py
    ├── explainability.py
    ├── feature_engineering.py
    ├── forecasting.py
    ├── inference.py
    ├── inventory.py
    ├── prepare_inference_data.py
    ├── risk.py
    ├── training_pipeline.py
    ├── train_ml.py
    ├── uncertainty.py
    └── utils.py
```

---

## Technology Stack

### Programming

* Python

### Machine Learning

* XGBoost
* LightGBM
* SHAP

### Data Processing

* Pandas
* NumPy

### Model Tracking

* MLflow

### Backend

* FastAPI

### Frontend

* Streamlit

### Deployment

* Docker

---

## API Endpoints

### Health Check

```http
GET /health
```

### Model Information

```http
GET /model-info
```

### Version

```http
GET /version
```

### Forecast

```http
GET /forecast/{item_id}
```

### Risk Assessment

```http
GET /risk/{item_id}
```

### Scenario Simulation

```http
POST /simulate
```

---

## Running Locally

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Launch Dashboard

```bash
streamlit run app.py
```

### Launch API

```bash
uvicorn src.api:app --reload
```

---

## Docker

Build image:

```bash
docker build -t retailforecast2 -f docker/dockerfile .
```

Run container:

```bash
docker run -p 8501:8501 retailforecast2
```

---

## Machine Learning Workflow

1. Data Ingestion
2. Data Validation
3. Feature Engineering
4. Model Training
5. Forecast Generation
6. Explainability Analysis
7. Inventory Optimization
8. Risk Evaluation
9. Scenario Simulation
10. Deployment

---

## Future Improvements

* Time-series deep learning models
* Real-time forecasting pipeline
* Automated retraining workflows
* CI/CD integration
* Cloud deployment
* Monitoring and observability
* Multi-store forecasting
* Advanced supply chain optimization

---

## Application Screenshots

### Executive Dashboard

![Dashboard](assets/dashboard.png)

### Forecast & SHAP Explainability

![Forecast](assets/forecast_shap.png)

### Inventory Optimizer

![Inventory](assets/inventory_optimizer.png)

### Risk Monitoring

![Risk](assets/risk_alerts.png)

### Scenario Simulator

![Scenario](assets/scenario_simulator.png)
Focused on:

* Machine Learning
* Data Science
* MLOps
* Forecasting Systems
* Explainable AI
* Production ML Engineering
