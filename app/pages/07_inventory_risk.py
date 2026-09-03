import streamlit as st
import pandas as pd
import plotly.express as px
from utils import (
    load_risk, load_sku_master,
    format_number, apply_chart_theme, COLORS, RISK_COLORS
)

st.title("Inventory Risk")
st.markdown("Monitor items with high risk of stockout or overstock across the retail network.")

risk_df = load_risk()
sku_master = load_sku_master()

if not risk_df.empty:
    df = pd.merge(risk_df, sku_master, on="sku_id", how="left") if not sku_master.empty else risk_df

    # ── KPIs ──
    total_items = len(df)
    critical = len(df[df["risk_level"] == "Critical"]) if "risk_level" in df.columns else 0
    high = len(df[df["risk_level"] == "High"]) if "risk_level" in df.columns else 0
    medium = len(df[df["risk_level"] == "Medium"]) if "risk_level" in df.columns else 0
    low = len(df[df["risk_level"] == "Low"]) if "risk_level" in df.columns else 0
    avg_score = df["risk_score"].mean() if "risk_score" in df.columns else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Assessed", format_number(total_items))
    c2.metric("Avg Risk Score", f"{avg_score:.1f}")
    c3.metric("🔴 Critical Risk", format_number(critical))

    st.write("") # small spacer
    c4, c5, c6 = st.columns(3)
    c4.metric("🟠 High Risk", format_number(high))
    c5.metric("🟡 Medium Risk", format_number(medium))
    c6.metric("🟢 Low Risk", format_number(low))

    st.markdown("---")

    # ── Row 1: Risk Distribution + Top Risk Items ──
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Risk Level Distribution")
        if "risk_level" in df.columns:
            risk_dist = df["risk_level"].value_counts().reset_index()
            risk_dist.columns = ["risk_level", "count"]
            color_map = {k: v for k, v in RISK_COLORS.items() if k in risk_dist["risk_level"].values}
            fig = px.pie(risk_dist, names="risk_level", values="count", hole=0.45,
                         color="risk_level", color_discrete_map=color_map)
            fig.update_traces(textinfo="label+percent", textposition="inside")
            fig.update_layout(showlegend=False)
            st.plotly_chart(apply_chart_theme(fig, height=400), use_container_width=True)

    with col2:
        st.subheader("Top 10 Highest Risk Items")
        if "risk_score" in df.columns:
            top_risk = df.sort_values("risk_score", ascending=False).head(10)
            name_col = "sku_name" if "sku_name" in top_risk.columns else "sku_id"
            fig = px.bar(top_risk, x="risk_score", y=name_col, orientation="h",
                         color="risk_level" if "risk_level" in top_risk.columns else None,
                         color_discrete_map=RISK_COLORS)
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(apply_chart_theme(fig, height=400), use_container_width=True)
        else:
            st.info("Risk scores not available.")

    # ── Row 2: Risk by Category + Risk Score Distribution ──
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Risk Breakdown by Category")
        if "category" in df.columns and "risk_level" in df.columns:
            cat_risk = df.groupby(["category", "risk_level"]).size().reset_index(name="count")
            fig = px.bar(cat_risk, x="category", y="count", color="risk_level",
                         barmode="stack", color_discrete_map=RISK_COLORS)
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(apply_chart_theme(fig, height=400), use_container_width=True)
        else:
            st.info("Category data not available for risk breakdown.")

    with col4:
        st.subheader("Risk Score Distribution")
        if "risk_score" in df.columns:
            fig = px.histogram(df, x="risk_score", nbins=25, color_discrete_sequence=[COLORS["orange"]])
            fig.update_layout(xaxis_title="Risk Score", yaxis_title="Count")
            st.plotly_chart(apply_chart_theme(fig, height=400), use_container_width=True)
        else:
            st.info("Risk score data not available.")

    # ── Critical & High Risk Table ──
    st.subheader("Critical & High Risk Items")
    if "risk_level" in df.columns:
        high_risk_df = df[df["risk_level"].isin(["Critical", "High"])]
        display_cols = [c for c in ["sku_id", "sku_name", "category", "risk_level", "risk_score"] if c in high_risk_df.columns]
        if not high_risk_df.empty:
            table_df = high_risk_df[display_cols].sort_values("risk_score", ascending=False)
            csv = table_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="high_risk_inventory.csv",
                mime="text/csv",
            )
            st.dataframe(table_df, use_container_width=True)
        else:
            st.success("No critical or high risk items found.")
    else:
        st.dataframe(df, use_container_width=True)
else:
    st.info("Inventory risk data is not available.")
