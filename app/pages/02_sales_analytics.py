import streamlit as st
import pandas as pd
import plotly.express as px
from utils import (
    load_dashboard_summaries,
    format_number, apply_chart_theme, COLORS
)

st.title("Sales Analytics")
st.markdown("Analyze demand patterns across time periods to optimize operations and staffing.")

daily, store, sku, monthly, category, weekday = load_dashboard_summaries()

# ── KPIs ──
total_demand = daily["demand"].sum() if not daily.empty else 0
avg_monthly = monthly["demand"].mean() if not monthly.empty else 0
peak_month = ""
if not monthly.empty:
    peak_row = monthly.loc[monthly["demand"].idxmax()]
    peak_month = f"{int(peak_row.get('year', 0))}-{int(peak_row.get('month_num', 0)):02d}"
peak_weekday = weekday.loc[weekday["demand"].idxmax(), "weekday"] if not weekday.empty and "weekday" in weekday.columns else "—"

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Demand", format_number(total_demand))
c2.metric("Avg Monthly Demand", format_number(avg_monthly))
c3.metric("Peak Month", peak_month if peak_month else "—")
c4.metric("Busiest Day", str(peak_weekday))

st.markdown("---")

# ── Row 1: Monthly + Weekday ──
col1, col2 = st.columns(2)

with col1:
    st.subheader("Monthly Sales Trend")
    if not monthly.empty:
        monthly["month_label"] = monthly["year"].astype(str) + "-" + monthly["month_num"].astype(str).str.zfill(2)
        fig = px.bar(monthly, x="month_label", y="demand", color_discrete_sequence=[COLORS["teal"]])
        st.plotly_chart(apply_chart_theme(fig, height=400), use_container_width=True)
    else:
        st.info("No monthly data available.")

with col2:
    st.subheader("Demand by Day of Week")
    if not weekday.empty:
        fig = px.bar(weekday, x="weekday", y="demand", color_discrete_sequence=[COLORS["purple"]])
        st.plotly_chart(apply_chart_theme(fig, height=400), use_container_width=True)
    else:
        st.info("No weekday data available.")

# ── Row 2: Cumulative + Category (top 5 + Other) ──
col3, col4 = st.columns(2)

with col3:
    st.subheader("Cumulative Demand Over Time")
    if not daily.empty:
        cum_df = daily.copy()
        cum_df["cumulative_demand"] = cum_df["demand"].cumsum()
        fig = px.area(cum_df, x="date", y="cumulative_demand", color_discrete_sequence=[COLORS["blue"]])
        st.plotly_chart(apply_chart_theme(fig, height=400), use_container_width=True)
    else:
        st.info("No daily data available.")

with col4:
    st.subheader("Demand Share by Category")
    if not category.empty:
        top5 = category.head(5).copy()
        other_val = category.iloc[5:]["demand"].sum() if len(category) > 5 else 0
        if other_val > 0:
            top5 = pd.concat([top5, pd.DataFrame([{"category": "Other", "demand": other_val}])], ignore_index=True)
        fig = px.pie(top5, names="category", values="demand", hole=0.45,
                     color_discrete_sequence=[COLORS["cyan"], COLORS["teal"], COLORS["purple"],
                                              COLORS["blue"], COLORS["green"], COLORS["muted"]])
        fig.update_traces(textinfo="label+percent", textposition="inside")
        fig.update_layout(showlegend=False)
        st.plotly_chart(apply_chart_theme(fig, height=400), use_container_width=True)
    else:
        st.info("No category data available.")
