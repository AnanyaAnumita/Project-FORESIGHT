import streamlit as st
import pandas as pd
import plotly.express as px
from utils import (
    load_inventory, load_sku_master, load_dashboard_summaries,
    format_number, apply_chart_theme, COLORS
)

st.title("Inventory Dashboard")
st.markdown("Current view of inventory stock levels across the supply chain.")

inv_df = load_inventory()
sku_master = load_sku_master()
daily, store, sku_demand, monthly, category, weekday = load_dashboard_summaries()

if not inv_df.empty:
    df = pd.merge(inv_df, sku_master, on="sku_id", how="left") if not sku_master.empty else inv_df

    total_stock = df["stock_on_hand"].sum()
    low_stock = len(df[df["stock_on_hand"] <= df["reorder_point"]]) if "reorder_point" in df.columns else 0
    avg_stock = df["stock_on_hand"].mean()
    total_items = len(df)

    # ── KPIs ──
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Stock Units", format_number(total_stock))
    c2.metric("SKUs Tracked", format_number(total_items))
    c3.metric("Avg Stock / SKU", format_number(avg_stock))
    c4.metric("Below Reorder Point", format_number(low_stock))

    st.markdown("---")

    # ── Row 1: Stock by Product + Stock Distribution ──
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top 15 Products by Stock")
        if "sku_name" in df.columns:
            prod_stock = df.groupby("sku_name")["stock_on_hand"].sum().reset_index().sort_values("stock_on_hand", ascending=False).head(15)
            fig = px.bar(prod_stock, x="stock_on_hand", y="sku_name", orientation="h", color_discrete_sequence=[COLORS["purple"]])
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(apply_chart_theme(fig, height=450), use_container_width=True)
        else:
            st.info("Product names not available.")

    with col2:
        st.subheader("Stock Level Distribution")
        fig = px.histogram(df, x="stock_on_hand", nbins=30, color_discrete_sequence=[COLORS["teal"]])
        fig.update_layout(xaxis_title="Stock on Hand", yaxis_title="Number of SKUs")
        st.plotly_chart(apply_chart_theme(fig, height=450), use_container_width=True)

    # ── Row 2: Stock vs Reorder + Category Pie (top 5 + Other) ──
    col3, col4 = st.columns(2)
    with col3:
        if "reorder_point" in df.columns and "sku_name" in df.columns:
            st.subheader("Lowest Stock vs Reorder Point")
            bottom = df.sort_values("stock_on_hand").head(10)
            fig = px.bar(bottom, x="sku_name", y=["stock_on_hand", "reorder_point"],
                         barmode="group",
                         color_discrete_sequence=[COLORS["blue"], COLORS["red"]])
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(apply_chart_theme(fig, height=400), use_container_width=True)

    with col4:
        if "category" in df.columns:
            st.subheader("Stock by Category")
            cat_stock = df.groupby("category")["stock_on_hand"].sum().reset_index().sort_values("stock_on_hand", ascending=False)
            top5 = cat_stock.head(5).copy()
            other_val = cat_stock.iloc[5:]["stock_on_hand"].sum() if len(cat_stock) > 5 else 0
            if other_val > 0:
                top5 = pd.concat([top5, pd.DataFrame([{"category": "Other", "stock_on_hand": other_val}])], ignore_index=True)
            fig = px.pie(top5, names="category", values="stock_on_hand", hole=0.45,
                         color_discrete_sequence=[COLORS["cyan"], COLORS["purple"], COLORS["teal"],
                                                  COLORS["blue"], COLORS["green"], COLORS["muted"]])
            fig.update_traces(textinfo="label+percent", textposition="inside")
            fig.update_layout(showlegend=False)
            st.plotly_chart(apply_chart_theme(fig, height=400), use_container_width=True)

    # ── Table ──
    st.subheader("Inventory Snapshot")
    display_cols = [c for c in ["sku_id", "sku_name", "category", "store_id", "stock_on_hand", "reorder_point", "safety_stock"] if c in df.columns]
    table_df = df[display_cols].head(50)
    
    csv = table_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name="inventory_snapshot.csv",
        mime="text/csv",
    )
    
    st.dataframe(table_df, use_container_width=True)
else:
    st.info("Inventory snapshot data is not available.")
