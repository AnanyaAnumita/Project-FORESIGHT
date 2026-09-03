import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils import (
    load_dashboard_summaries, load_inventory, load_risk, load_model_results,
    format_number, format_currency, apply_chart_theme, COLORS, RISK_COLORS
)

import os

logo_path = "src/logo2.png"
if os.path.exists(logo_path):
    st.image(logo_path, width=200)

st.title("Executive Overview")
st.markdown("High-level summary of demand and inventory health across the retail network.")

daily, store, sku, monthly, category, weekday = load_dashboard_summaries()
inv_df = load_inventory()
risk_df = load_risk()
models_df = load_model_results()

# ── Row 1: Primary KPIs ──
total_demand = daily["demand"].sum() if not daily.empty else 0
total_stock = inv_df["stock_on_hand"].sum() if not inv_df.empty else 0
critical_risk = len(risk_df[risk_df["risk_level"] == "Critical"]) if not risk_df.empty and "risk_level" in risk_df.columns else 0
active_skus = len(sku) if not sku.empty else 0
num_stores = len(store) if not store.empty else 0
avg_daily = daily["demand"].mean() if not daily.empty else 0

c1, c2, c3 = st.columns(3)
c1.metric("Total Sales Volume", format_number(total_demand))
c2.metric("Avg Daily Demand", format_number(avg_daily))
c3.metric("Total Inventory", format_number(total_stock))

st.write("") # small spacer
c4, c5, c6 = st.columns(3)
c4.metric("Critical Risk SKUs", format_number(critical_risk))
c5.metric("Active Products", format_number(active_skus))
c6.metric("Active Stores", format_number(num_stores))

# ── Row 2: Model Accuracy + Inventory Health ──
st.markdown("---")

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Forecast Model Accuracy")
    if not models_df.empty:
        best = models_df.iloc[0]
        m1, m2, m3 = st.columns(3)
        m1.metric("Best Model", best.get("model", "—"))
        m2.metric("MAPE", f"{best.get('MAPE_percent', 0):.1f}%")
        m3.metric("RMSE", format_number(best.get("RMSE", 0)))
    else:
        st.info("Model results not available.")

with col_right:
    st.subheader("Inventory Risk Breakdown")
    if not risk_df.empty and "risk_level" in risk_df.columns:
        risk_counts = risk_df["risk_level"].value_counts().reset_index()
        risk_counts.columns = ["risk_level", "count"]
        # Max 2 metrics per row inside this half-column to prevent cramping
        r_cols1 = st.columns(2)
        r_cols2 = st.columns(2)
        
        for i, row in risk_counts.iterrows():
            if i < 2:
                r_cols1[i].metric(f"{row['risk_level']} Risk", format_number(row["count"]))
            elif i < 4:
                r_cols2[i - 2].metric(f"{row['risk_level']} Risk", format_number(row["count"]))
    else:
        st.info("Risk data not available.")

# ── Row 3: Charts ──
st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Daily Demand Trend")
    if not daily.empty:
        fig = px.area(daily, x="date", y="demand", color_discrete_sequence=[COLORS["blue"]])
        st.plotly_chart(apply_chart_theme(fig, height=350), use_container_width=True)
    else:
        st.info("No daily demand data available.")

with col2:
    st.subheader("Top Categories by Volume")
    if not category.empty:
        top_cat = category.head(5)
        fig = px.bar(top_cat, x="demand", y="category", orientation="h", color_discrete_sequence=[COLORS["cyan"]])
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(apply_chart_theme(fig, height=350), use_container_width=True)
    else:
        st.info("No category data available.")

# ── Row 4: Monthly Trend + Top Stores ──
col3, col4 = st.columns(2)
with col3:
    st.subheader("Monthly Demand Evolution")
    if not monthly.empty:
        monthly["month_label"] = monthly["year"].astype(str) + "-" + monthly["month_num"].astype(str).str.zfill(2)
        fig = px.line(monthly, x="month_label", y="demand", markers=True, color_discrete_sequence=[COLORS["teal"]])
        st.plotly_chart(apply_chart_theme(fig, height=350), use_container_width=True)
    else:
        st.info("No monthly data available.")

with col4:
    st.subheader("Top 5 Stores by Demand")
    if not store.empty:
        top5 = store.head(5)
        fig = px.bar(top5, x="store_id", y="demand", color_discrete_sequence=[COLORS["purple"]])
        st.plotly_chart(apply_chart_theme(fig, height=350), use_container_width=True)
    else:
        st.info("No store data available.")
