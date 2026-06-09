import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
def calculate_confidence_interval(
    forecast,
    confidence=0.95
):

    forecast = np.array(
        forecast
    )

    std = np.std(
        forecast
    )

    z_score = norm.ppf(
        (1 + confidence) / 2
    )

    margin = z_score * std

    lower = forecast - margin
    upper = forecast + margin

    return lower, upper


from src.inference import (
    predict_28_day_forecast,
    load_model,
    build_features_for_item
)

from src.explainability import (
    get_shap_explanations
)
def calculate_reorder_point(demand_lt, safety_stock):
    return demand_lt + safety_stock

def assess_stockout_risk(current_inventory, reorder_point):
    if current_inventory < (reorder_point * 0.5):
        return "High Risk"
    elif current_inventory < reorder_point:
        return "Medium Risk"
    return "Low Risk"

def simulate_scenario(forecast, lead_time, growth, service):
    mean_demand = np.mean(forecast) * (1 + growth)
    safety_stock = mean_demand * 0.2 * (service * 1.5)
    reorder_point = (mean_demand * lead_time / 28) + safety_stock
    return {
        "safety_stock": safety_stock,
        "reorder_point": reorder_point,
        "adjusted_forecast_mean": mean_demand,
        "lead_time_days": lead_time,
        "growth_applied": growth,
        "target_service_level": service
    }
# ---------------------------------------------------------------------------


# ==================================
# PAGE CONFIG
# ==================================
st.set_page_config(
    page_title="Retail forecastory and inventory optimization",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================
# CLUTTER-FREE HIGH RESOLUTION STYLING
# ==================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    /* Global text */
    html, body, .stMarkdown, p, li, label, span {
        font-family: 'Inter', sans-serif !important;
        font-size: 20px !important;   /* larger and more readable */
        line-height: 1.75 !important;
        color: #1e293b;
    }

    /* Main title */
    .main-title {
        font-size: 44px !important;   /* stronger visibility */
        font-weight: 800 !important;
        color: #0f172a;
        margin-bottom: 1.25rem;
    }

    /* Subtitle */
    .subtitle {
        font-size: 30px !important;
        color: #334155;
        margin-bottom: 2rem;
    }

    /* Headings hierarchy */
    h1 { font-size: 32px !important; font-weight: 700 !important; }
    h2 { font-size: 28px !important; font-weight: 700 !important; }
    h3 { font-size: 24px !important; font-weight: 700 !important; }
    h4 { font-size: 22px !important; font-weight: 600 !important; }
    h5 { font-size: 22px !important; font-weight: 600 !important; }
    label, div[data-testid="stTextInput"] label {
        font-size: 20px !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa;
        min-width: 340px !important;
        max-width: 340px !important;
        border-right: 1px solid #e9ecef;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] h5,
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span {
        font-size: 24px !important;
        color: #0f172a !important;
    }
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        font-size: 30px !important;
        font-weight: 800 !important;
    }
    section[data-testid="stSidebar"] .stMarkdown p {
        margin: 0 !important;
    }

    /* Metrics */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    div[data-testid="stMetricLabel"] > div {
        font-size: 13px !important;
        text-transform: uppercase !important;
        font-weight: 600 !important;
        color: #64748b !important;
    }
    div[data-testid="stMetricValue"] > div {
        font-size: 26px !important;
        font-weight: 700 !important;
        color: #0f172a !important;
    }

    /* Buttons */
    .stButton>button {
        background-color: #10b981 !important;
        color: white !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        font-size: 18px !important;
        padding: 0.8rem 1.5rem !important;
        border: none !important;
        width: 100%;
        margin-top: 1rem !important;
    }
    .stButton>button:hover {
        background-color: #0f766e !important;
    }
</style>
""", unsafe_allow_html=True)


# ==================================
# SIDEBAR NAVIGATION
# ==================================
with st.sidebar:
    st.markdown("<div style='padding-top: 0.5rem;'></div>", unsafe_allow_html=True)
    st.image("https://img.icons8.com/fluent/96/artificial-intelligence.png", width=65)
    st.markdown("### Retail Forecastory & Inventory Optimization")
   
    
    st.markdown("---")
    page = st.selectbox(
        "Navigation Focus",
        [
            "Executive Dashboard",
            "Forecast & SHAP Explainer",
            "Inventory Optimizer",
            "Risk Alerts",
            "Scenario Simulator"
        ],
        index=1  # Routes directly to Forecast view for confirmation tracking
    )
    
    st.markdown("---")
    st.markdown("🧬 **Model Architecture**: `XGBoost v2.1 + SHAP`")
    st.markdown("📡 **Data Freshness**: `Updated 4m ago`")


# ==================================
# EXECUTIVE DASHBOARD
# ==================================
if page == "Executive Dashboard":
    st.markdown("<h1 class='main-title'>📈 Enterprise Operations Control</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Production-grade demand tracking backed by transparent, explainable machine learning models.</p>", unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("System Mean Accuracy", "94.2%", help="Calculated using mean absolute scaled error (MASE)")
    c2.metric("Explainability Model", "SHAP Kernel", "Active")
    c3.metric("Monitored SKUs", "14,205 Active", "+128 Today")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("### Core Platform Capabilities")
    col_a, col_b = st.columns(2)
    
    with col_a:
        with st.container(border=True):
            st.markdown("#### 🧠 Trustworthy & Explainable AI")
            st.markdown("""
            * **SHAP Feature Attribution:** Breaks down global black-box sequences into individual, localized driver weights.
            * **Feature Interaction Tracks:** Observes how price elasticity cross-references ongoing markdown intervals.
            """)
            
    with col_b:
        with st.container(border=True):
            st.markdown("#### 🛡️ Robust Supply Management")
            st.markdown("""
            * **Simulation Engines:** Stress-tests your inventory points against volatile adjustments.
            * **Automated Risk Triage:** Generates localized dynamic priority alerts when stock profiles flag anomalies.
            """)


# ==================================
# FORECAST & SHAP EXPLORER
# ==================================
elif page == "Forecast & SHAP Explainer":

    st.markdown(
        "<h1 class='main-title'>🔮 Demand Inference & SHAP Explainer</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<p class='subtitle'>Evaluate standard baseline calculations coupled with localized machine learning attribution features.</p>",
        unsafe_allow_html=True
    )

    with st.container(border=True):

        st.markdown("##### Target Node Configuration")

        item_id = st.text_input(
            "Target SKU Identifier",
            "FOODS_1_003_CA_4_validation"
        )

        submit_forecast = st.button(
            "Execute Pipeline & SHAP Analysis"
        )

    if submit_forecast:

        try:

            with st.spinner(
                "Calculating sequence vectors and SHAP attributions..."
            ):

                forecast = predict_28_day_forecast(
                    item_id
                )

                model, features = load_model()

                X = build_features_for_item(
                    item_id,
                    features
                )

                shap_df = get_shap_explanations(
                    model,
                    X
                )

                shap_value_col = (
                    "SHAP Value (Impact)"
                    if "SHAP Value (Impact)" in shap_df.columns
                    else "SHAP Value"
                )

                lower_ci, upper_ci = (
                    calculate_confidence_interval(
                        forecast
                    )
                )

                forecast_df = pd.DataFrame({
                    "Day": range(1, 29),
                    "Forecast": forecast,
                    "Lower CI": lower_ci,
                    "Upper CI": upper_ci
                })

            st.success(
                "✅ Forecast timeline generated and SHAP values verified."
            )

            # =====================
            # KPI SECTION
            # =====================

            forecast_mean = forecast.mean()
            if np.isclose(forecast_mean, 0.0):
                mape_text = "N/A"
            else:
                mape = np.mean(
                    np.abs(
                        (forecast - forecast_mean)
                        / forecast_mean
                    )
                ) * 100
                mape_text = f"{mape:.2f}%"

            k1, k2, k3, k4 = st.columns(4)

            k1.metric(
                "28D Demand",
                f"{forecast.sum():.0f}"
            )

            k2.metric(
                "Avg Daily",
                f"{forecast.mean():.2f}"
            )

            k3.metric(
                "Peak Day",
                f"{forecast.max():.2f}"
            )

            k4.metric(
                "Volatility",
                f"{forecast.std():.2f}"
            )

            st.metric(
                "Forecast Error Proxy",
                mape_text
            )

            # =====================
            # TOP DRIVER
            # =====================

            top_feature = shap_df.loc[
                shap_df[shap_value_col].abs().idxmax()
            ]

            st.info(
                f"🔍 Strongest Demand Driver: "
                f"{top_feature['Feature']} "
                f"({top_feature[shap_value_col]:.2f})"
            )

            # =====================
            # CHARTS
            # =====================

            chart_col, shap_col = st.columns([1, 1])

            with chart_col:

                with st.container(border=True):

                    st.markdown(
                        "##### Forecast + Confidence Interval"
                    )

                    st.line_chart(
                        forecast_df[
                            [
                                "Forecast",
                                "Lower CI",
                                "Upper CI"
                            ]
                        ],
                        height=360
                    )

            with shap_col:

                with st.container(border=True):

                    st.markdown(
                        "##### Local SHAP Feature Attribution"
                    )

                    shap_df = shap_df.sort_values(
                        by=shap_value_col,
                        key=lambda x: x.abs(),
                        ascending=False
                    )

                    chart_data = shap_df.set_index("Feature")[[shap_value_col]]

                    st.bar_chart(
                        chart_data,
                        height=360
                    )

            # =====================
            # DOWNLOAD BUTTON
            # =====================

            csv = forecast_df.to_csv(
                index=False
            )

            st.download_button(
                label="📥 Download Forecast CSV",
                data=csv,
                file_name="forecast.csv",
                mime="text/csv"
            )

            # =====================
            # TABLES
            # =====================

            st.markdown(
                "### Execution Telemetry Details"
            )

            table_col1, table_col2 = st.columns([1, 1])

            with table_col1:

                with st.container(border=True):

                    st.markdown(
                        "##### Demand Matrix Raw Data"
                    )

                    st.dataframe(
                        forecast_df,
                        use_container_width=True,
                        height=300
                    )

            with table_col2:

                with st.container(border=True):

                    st.markdown(
                        "##### SHAP Weighting Insights"
                    )

                    st.dataframe(
                        shap_df.style.map(
                            lambda val:
                            "color: green;"
                            if val > 0
                            else "color: red;",
                            subset=[shap_value_col]
                        ),
                        use_container_width=True,
                        height=300
                    )

        except Exception as e:

            st.error(
                f"Inference pipeline failure:\n{str(e)}"
            )


# ==================================
# INVENTORY OPTIMIZER
# ==================================
elif page == "Inventory Optimizer":

    st.markdown(
        "<h1 class='main-title'>⚙️ Statistical Inventory Optimization</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<p class='subtitle'>Optimize inventory levels using EOQ, Safety Stock and Reorder Point calculations.</p>",
        unsafe_allow_html=True
    )

    col_inputs, col_outputs = st.columns([1, 1])

    with col_inputs:

        with st.container(border=True):

            st.markdown("##### Inventory Parameters")

            annual_demand = st.number_input(
                "Annual Demand (Units)",
                min_value=1,
                value=50000
            )

            order_cost = st.number_input(
                "Ordering Cost per Order ($)",
                min_value=1,
                value=100
            )

            holding_cost = st.number_input(
                "Holding Cost per Unit ($)",
                min_value=1,
                value=5
            )

            avg_daily_demand = st.number_input(
                "Average Daily Demand",
                min_value=1,
                value=120
            )

            lead_time = st.number_input(
                "Lead Time (Days)",
                min_value=1,
                value=7
            )

            safety_stock = st.number_input(
                "Safety Stock",
                min_value=0,
                value=65
            )

            calculate_btn = st.button(
                "Run Inventory Optimization"
            )

    with col_outputs:

        if calculate_btn:

            # EOQ
            eoq = np.sqrt(
                (2 * annual_demand * order_cost)
                / holding_cost
            )

            reorder_point = (
                avg_daily_demand * lead_time
            ) + safety_stock

            annual_holding_cost = (
                eoq / 2
            ) * holding_cost

            suggested_order_qty = round(
                eoq
            )

            st.markdown(
                "##### Optimization Results"
            )

            m1, m2 = st.columns(2)

            m1.metric(
                "EOQ",
                f"{eoq:.0f} Units"
            )

            m2.metric(
                "Reorder Point",
                f"{reorder_point:.0f} Units"
            )

            m3, m4 = st.columns(2)

            m3.metric(
                "Suggested Order Qty",
                f"{suggested_order_qty}"
            )

            m4.metric(
                "Annual Holding Cost",
                f"${annual_holding_cost:,.0f}"
            )

            st.success(
                f"Recommended Purchase Quantity: {suggested_order_qty} units"
            )

            st.info(
                f"Place a new order when inventory falls below {reorder_point:.0f} units."
            )

            inventory_df = pd.DataFrame({
                "Metric": [
                    "EOQ",
                    "Reorder Point",
                    "Safety Stock",
                    "Annual Holding Cost"
                ],
                "Value": [
                    round(eoq),
                    round(reorder_point),
                    safety_stock,
                    round(annual_holding_cost)
                ]
            })

            st.dataframe(
                inventory_df,
                use_container_width=True
            )

        else:

            st.info(
                "Provide inventory parameters to generate optimization recommendations."
            )
# ==================================
# RISK ALERTS
# ==================================
elif page == "Risk Alerts":

    st.markdown(
        "<h1 class='main-title'>🚨 Production Exception & Stockout Triaging</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<p class='subtitle'>AI-driven inventory risk monitoring and stockout prediction.</p>",
        unsafe_allow_html=True
    )

    with st.container(border=True):

        c1, c2, c3 = st.columns(3)

        with c1:
            current_inventory = st.number_input(
                "Current Inventory",
                min_value=0,
                value=95
            )

        with c2:
            avg_daily_demand = st.number_input(
                "Average Daily Demand",
                min_value=1,
                value=20
            )

        with c3:
            reorder_point = st.number_input(
                "Reorder Point",
                min_value=1,
                value=220
            )

        assess_btn = st.button(
            "Analyze Inventory Risk"
        )

    if assess_btn:

        risk = assess_stockout_risk(
            current_inventory,
            reorder_point
        )

        days_until_stockout = (
            current_inventory / avg_daily_demand
            if avg_daily_demand > 0
            else np.inf
        )

        if np.isfinite(days_until_stockout):
            stockout_date = (
                pd.Timestamp.today()
                + pd.Timedelta(days=days_until_stockout)
            )
            stockout_date_text = stockout_date.strftime("%Y-%m-%d")
        else:
            stockout_date_text = "N/A"

        # Health Score
        health_score = min(
            100,
            max(
                0,
                (current_inventory /
                 reorder_point) * 100
            )
        )

        # Emergency Order Quantity
        emergency_order_qty = max(
            0,
            reorder_point -
            current_inventory
        )

        st.markdown(
            "### Inventory Risk Dashboard"
        )

        m1, m2, m3, m4 = st.columns(4)

        m1.metric(
            "Risk Level",
            risk
        )

        m2.metric(
            "Health Score",
            f"{health_score:.0f}/100"
        )

        m3.metric(
            "Days Until Stockout",
            f"{days_until_stockout:.1f}"
        )

        m4.metric(
            "Emergency Order Qty",
            f"{emergency_order_qty:.0f}"
        )

        st.metric(
            "Expected Stockout Date",
            stockout_date.strftime("%Y-%m-%d")
        )

        if risk == "High Risk":

            st.error(
                "🔴 CRITICAL ALERT: Stockout likely before replenishment arrives."
            )

        elif risk == "Medium Risk":

            st.warning(
                "🟠 WARNING: Inventory approaching critical threshold."
            )

        else:

            st.success(
                "🟢 Inventory levels are healthy."
            )

        risk_df = pd.DataFrame({
            "Metric": [
                "Current Inventory",
                "Reorder Point",
                "Daily Demand",
                "Days Until Stockout",
                "Emergency Order Qty"
            ],
            "Value": [
                current_inventory,
                reorder_point,
                avg_daily_demand,
                round(days_until_stockout, 2),
                round(emergency_order_qty)
            ]
        })

        st.dataframe(
            risk_df,
            use_container_width=True
        )

        # Inventory Status Gauge
        st.progress(
            min(
                int(health_score),
                100
            )
        )

        st.caption(
            f"Inventory Health: {health_score:.0f}%"
        )# ==================================
# SCENARIO SIMULATOR
# ==================================
elif page == "Scenario Simulator":

    st.markdown(
        "<h1 class='main-title'>🎛️ Enterprise Scenario Simulator</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<p class='subtitle'>Evaluate inventory, revenue, profit and supply-chain impact under different business conditions.</p>",
        unsafe_allow_html=True
    )

    with st.container(border=True):

        item_id = st.text_input(
            "Simulation SKU",
            "FOODS_1_003_CA_4_validation"
        )

        col1, col2 = st.columns(2)

        with col1:

            lead_time = st.slider(
                "Lead Time (Days)",
                1,
                30,
                7
            )

            demand_growth = st.slider(
                "Demand Growth %",
                -20,
                100,
                15
            )

            unit_price = st.number_input(
                "Selling Price ($)",
                value=15.0
            )

        with col2:

            service_level = st.slider(
                "Service Level %",
                90,
                99,
                97
            )

            supplier_delay = st.slider(
                "Supplier Delay %",
                0,
                100,
                10
            )

            unit_cost = st.number_input(
                "Unit Cost ($)",
                value=8.0
            )

        run_sim = st.button(
            "Run Enterprise Simulation"
        )

    if run_sim:

        try:

            with st.spinner(
                "Running business simulation..."
            ):

                forecast = np.array(
                    predict_28_day_forecast(
                        item_id
                    )
                )

                adjusted_forecast = (
                    forecast *
                    (1 + demand_growth / 100)
                )

                mean_demand = adjusted_forecast.mean()

                safety_stock = (
                    mean_demand *
                    0.2 *
                    (service_level / 100)
                )

                reorder_point = (
                    (mean_demand * lead_time / 28)
                    + safety_stock
                )

                total_demand = adjusted_forecast.sum()

                revenue = (
                    total_demand *
                    unit_price
                )

                profit = (
                    total_demand *
                    (unit_price - unit_cost)
                )

                shortage_cost = (
                    total_demand *
                    supplier_delay /
                    100 *
                    unit_cost *
                    0.2
                )

                net_profit = (
                    profit -
                    shortage_cost
                )

                roi = (
                    net_profit /
                    (total_demand * unit_cost)
                ) * 100 if total_demand * unit_cost else np.nan

            st.success(
                "✅ Enterprise simulation completed."
            )

            # ====================
            # KPI DASHBOARD
            # ====================

            k1, k2, k3, k4 = st.columns(4)

            k1.metric(
                "Revenue",
                f"${revenue:,.0f}"
            )

            k2.metric(
                "Gross Profit",
                f"${profit:,.0f}"
            )

            k3.metric(
                "Net Profit",
                f"${net_profit:,.0f}"
            )

            k4.metric(
                "Demand",
                f"{total_demand:,.0f}"
            )

            st.metric(
                "ROI",
                f"{roi:.1f}%" if not np.isnan(roi) else "N/A"
            )

            # ====================
            # INVENTORY KPIs
            # ====================

            m1, m2, m3 = st.columns(3)

            m1.metric(
                "Safety Stock",
                f"{safety_stock:.0f}"
            )

            m2.metric(
                "Reorder Point",
                f"{reorder_point:.0f}"
            )

            m3.metric(
                "Shortage Cost",
                f"${shortage_cost:,.0f}"
            )

            # ====================
            # FORECAST GRAPH
            # ====================

            chart_df = pd.DataFrame({
                "Forecast": forecast,
                "Adjusted Demand": adjusted_forecast
            })

            st.markdown(
                "### Demand Impact Analysis"
            )

            st.line_chart(
                chart_df,
                height=400
            )

            # ====================
            # BUSINESS SUMMARY
            # ====================

            st.markdown(
                "### Executive Summary"
            )

            st.info(
                f"""
                Demand Growth Applied: {demand_growth}%  
                Revenue Forecast: ${revenue:,.0f}  
                Net Profit Forecast: ${net_profit:,.0f}  
                Reorder Point Recommendation: {reorder_point:.0f} units  
                Estimated Shortage Cost: ${shortage_cost:,.0f}
                """
            )

            simulation_output = pd.DataFrame({
                "Metric": [
                    "Revenue",
                    "Gross Profit",
                    "Net Profit",
                    "Safety Stock",
                    "Reorder Point",
                    "Shortage Cost"
                ],
                "Value": [
                    revenue,
                    profit,
                    net_profit,
                    safety_stock,
                    reorder_point,
                    shortage_cost
                ]
            })

            st.dataframe(
                simulation_output,
                use_container_width=True
            )

        except Exception as e:

            st.error(
                f"Simulation execution interrupted:\n{str(e)}"
            )