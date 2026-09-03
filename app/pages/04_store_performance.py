import streamlit as st
import pandas as pd
import plotly.express as px
from utils import (
    load_dashboard_summaries, load_inventory,
    format_number, apply_chart_theme, COLORS
)

st.title("Store Performance")
st.markdown("Analyze demand and inventory levels across different retail locations.")

daily, store, sku, monthly, category, weekday = load_dashboard_summaries()
inv_df = load_inventory()

if not store.empty:
    # Merge with inventory if available
    if not inv_df.empty and "store_id" in inv_df.columns:
        store_inv = inv_df.groupby("store_id").agg(
            stock_on_hand=("stock_on_hand", "sum"),
            items_below_reorder=("stock_on_hand", lambda x: (x <= inv_df.loc[x.index, "reorder_point"]).sum()) if "reorder_point" in inv_df.columns else ("stock_on_hand", "count")
        ).reset_index()
        df = pd.merge(store, store_inv, on="store_id", how="left").fillna(0)
    else:
        df = store.copy()
        df["stock_on_hand"] = 0

    # ── KPIs ──
    num_stores = len(df)
    total_demand = df["demand"].sum()
    avg_demand = df["demand"].mean()
    top_store = df.iloc[0]["store_id"] if not df.empty else "—"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Stores", format_number(num_stores))
    c2.metric("Total Demand", format_number(total_demand))
    c3.metric("Avg Demand / Store", format_number(avg_demand))
    c4.metric("Top Store", str(top_store))

    st.markdown("---")

    # ── Row 1: Bar chart + Demand Distribution ──
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top Stores by Demand")
        top_stores = df.sort_values("demand", ascending=False).head(15)
        fig = px.bar(top_stores, x="store_id", y="demand", color="demand",
                     color_continuous_scale=[[0, "#1E3A5F"], [0.5, "#3B82F6"], [1, "#06B6D4"]])
        st.plotly_chart(apply_chart_theme(fig, height=400), use_container_width=True)

    with col2:
        st.subheader("Store Demand Distribution")
        fig = px.histogram(df, x="demand", nbins=20, color_discrete_sequence=[COLORS["purple"]])
        fig.update_layout(xaxis_title="Demand Volume", yaxis_title="Number of Stores")
        st.plotly_chart(apply_chart_theme(fig, height=400), use_container_width=True)

    # ── Row 2: Stock vs Demand Comparison ──
    if "stock_on_hand" in df.columns and df["stock_on_hand"].sum() > 0:
        st.subheader("Demand vs Inventory by Store")
        top_compare = df.sort_values("demand", ascending=False).head(15)
        fig = px.bar(top_compare, x="store_id", y=["demand", "stock_on_hand"],
                     barmode="group",
                     color_discrete_sequence=[COLORS["blue"], COLORS["teal"]])
        fig.update_layout(legend_title_text="Metric")
        st.plotly_chart(apply_chart_theme(fig, height=400), use_container_width=True)

    # ── Store Table ──
    st.subheader("Store Data")
    display_cols = [c for c in ["store_id", "demand", "stock_on_hand"] if c in df.columns]
    st.dataframe(df[display_cols].sort_values("demand", ascending=False).head(30), use_container_width=True)
else:
    st.info("Store data is not available.")
