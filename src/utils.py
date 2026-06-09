# src/utils.py
import pandas as pd
import numpy as np
import streamlit as st
import mlflow
from datetime import datetime

def set_page_config():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title="RetailForecast2",
        page_icon="🛒",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def load_model(model_path='models/xgboost.pkl'):
    """Load trained model"""
    try:
        import joblib
        return joblib.load(model_path)
    except:
        st.warning("Model not found. Please train first.")
        return None

def calculate_wrmsse(actual, predicted, weights):
    """Calculate Weighted Root Mean Squared Scaled Error (M5 metric)"""
    # Simplified version - can be expanded later
    mse = np.mean((actual - predicted) ** 2)
    return np.sqrt(mse)

def log_experiment(model_name, params, metrics):
    """Log to MLflow"""
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        print(f"✅ Logged {model_name} to MLflow")

def format_number(num):
    """Format large numbers nicely"""
    return f"{num:,.0f}"

def get_hierarchical_levels():
    """Return M5 hierarchy levels"""
    return {
        'total': 'Total',
        'state': 'State',
        'store': 'Store',
        'cat': 'Category',
        'dept': 'Department',
        'item': 'Item'
    }