import streamlit as st
import pandas as pd
import plotly.express as px
from utils import (
    load_dashboard_summaries, load_sku_master,
    format_number, format_currency, apply_chart_theme, COLORS
)

st.title("Product Performance")
st.markdown("Analyze demand and revenue metrics across the product catalog.")

daily, store, sku_demand, monthly, category, weekday = load_dashboard_summaries()
sku_master = load_sku_master()

if not sku_demand.empty and not sku_master.empty:
    df = pd.merge(sku_demand, sku_master, on="sku_id", how="left")
    if "unit_price" in df.columns:
        df["revenue"] = df["demand"] * df["unit_price"]
    else:
        df["revenue"] = 0
    # ── Filters ──
    # ── Filters ──
    f_col1, f_col2 = st.columns(2)
    with f_col1:
        if "category" in df.columns:
            all_categories = sorted(df["category"].dropna().unique())
            selected_categories = st.multiselect(
                "Filter by Category (Leave empty to view all)",
                options=all_categories,
                default=[]
            )
            if selected_categories:
                df = df[df["category"].isin(selected_categories)]
    
    with f_col2:
        if "sku_name" in df.columns:
            search_query = st.text_input("Search Product Name", placeholder="e.g. Wireless Mouse")
            if search_query:
                df = df[df["sku_name"].str.contains(search_query, case=False, na=False)]

    st.markdown("---")
    # ── KPIs ──
    total_products = len(df)
    total_revenue = df["revenue"].sum()
    avg_demand = df["demand"].mean()
    top_product = df.sort_values("demand", ascending=False).iloc[0]["sku_name"] if "sku_name" in df.columns else "—"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Products", format_number(total_products))
    c2.metric("Total Revenue", format_currency(total_revenue))
    c3.metric("Avg Demand / Product", format_number(avg_demand))
    c4.metric("Top Product", str(top_product)[:20])

    st.markdown("---")

    # ── Row 1: Top by Volume + Revenue ──
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top 10 Products by Volume")
        top_vol = df.sort_values("demand", ascending=False).head(10)
        fig = px.bar(top_vol, x="demand", y="sku_name", orientation="h", color_discrete_sequence=[COLORS["cyan"]])
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(apply_chart_theme(fig, height=400), use_container_width=True)

    with col2:
        st.subheader("Top 10 Products by Revenue")
        top_rev = df.sort_values("revenue", ascending=False).head(10)
        fig = px.bar(top_rev, x="revenue", y="sku_name", orientation="h", color_discrete_sequence=[COLORS["green"]])
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(apply_chart_theme(fig, height=400), use_container_width=True)

    # ── Row 2: Category Revenue (top 5 + Other) ──
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Revenue by Category")
        if "category" in df.columns:
            cat_rev = df.groupby("category")["revenue"].sum().reset_index().sort_values("revenue", ascending=False)
            top5 = cat_rev.head(5)
            other_val = cat_rev.iloc[5:]["revenue"].sum() if len(cat_rev) > 5 else 0
            if other_val > 0:
                top5 = pd.concat([top5, pd.DataFrame([{"category": "Other", "revenue": other_val}])], ignore_index=True)
            fig = px.pie(top5, names="category", values="revenue", hole=0.45,
                         color_discrete_sequence=[COLORS["purple"], COLORS["cyan"], COLORS["teal"],
                                                  COLORS["blue"], COLORS["green"], COLORS["muted"]])
            fig.update_traces(textinfo="label+percent", textposition="inside")
            fig.update_layout(showlegend=False)
            st.plotly_chart(apply_chart_theme(fig, height=400), use_container_width=True)
        else:
            st.info("Category data not available.")

    with col4:
        st.subheader("Demand vs Unit Price")
        if "unit_price" in df.columns:
            fig = px.scatter(df, x="unit_price", y="demand",
                             hover_name="sku_name" if "sku_name" in df.columns else None,
                             color_discrete_sequence=[COLORS["cyan"]])
            fig.update_layout(xaxis_title="Unit Price", yaxis_title="Demand")
            st.plotly_chart(apply_chart_theme(fig, height=400), use_container_width=True)
        else:
            st.info("Price data not available.")

    # ── Product Table ──
    st.subheader("Product Catalog")
    display_cols = [c for c in ["sku_id", "sku_name", "category", "demand", "revenue", "unit_price"] if c in df.columns]
    table_df = df[display_cols].sort_values("demand", ascending=False).head(50)
    
    csv = table_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name="product_performance.csv",
        mime="text/csv",
    )
    
    st.dataframe(table_df, use_container_width=True)
else:
    st.info("Product data is not available.")
