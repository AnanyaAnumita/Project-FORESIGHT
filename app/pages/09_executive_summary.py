import streamlit as st
import pandas as pd
from utils import (
    load_dashboard_summaries, load_inventory, load_risk, load_model_results,
    load_sku_master, load_precomputed_forecast,
    format_number, format_currency, pct
)

st.title("Executive Summary")
st.markdown("Consolidated intelligence report — auto-generated from the latest data.")

daily, store, sku_demand, monthly, category, weekday = load_dashboard_summaries()
inv_df = load_inventory()
risk_df = load_risk()
models_df = load_model_results()
sku_master = load_sku_master()
forecast_df = load_precomputed_forecast()

# ── Primary KPIs ──
st.header("Key Performance Indicators")

total_demand = daily["demand"].sum() if not daily.empty else 0
total_stock = inv_df["stock_on_hand"].sum() if not inv_df.empty else 0
critical_risk = len(risk_df[risk_df["risk_level"] == "Critical"]) if not risk_df.empty and "risk_level" in risk_df.columns else 0
high_risk = len(risk_df[risk_df["risk_level"] == "High"]) if not risk_df.empty and "risk_level" in risk_df.columns else 0
num_products = len(sku_demand) if not sku_demand.empty else 0
num_stores = len(store) if not store.empty else 0

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Total Demand", format_number(total_demand))
c2.metric("Total Inventory", format_number(total_stock))
c3.metric("Products", format_number(num_products))
c4.metric("Stores", format_number(num_stores))
c5.metric("Critical Risk", format_number(critical_risk))
c6.metric("High Risk", format_number(high_risk))

st.markdown("---")

# ── Demand Insights ──
st.header("📈 Demand Insights")
insights = []

if not daily.empty:
    avg_daily = daily["demand"].mean()
    max_day = daily.loc[daily["demand"].idxmax()]
    min_day = daily.loc[daily["demand"].idxmin()]
    insights.append(f"- **Average daily demand**: {format_number(avg_daily)} units")
    insights.append(f"- **Peak demand day**: {max_day['date'].strftime('%Y-%m-%d') if hasattr(max_day['date'], 'strftime') else max_day['date']} ({format_number(max_day['demand'])} units)")
    insights.append(f"- **Lowest demand day**: {min_day['date'].strftime('%Y-%m-%d') if hasattr(min_day['date'], 'strftime') else min_day['date']} ({format_number(min_day['demand'])} units)")

if not monthly.empty:
    peak_month = monthly.loc[monthly["demand"].idxmax()]
    insights.append(f"- **Best performing month**: {int(peak_month.get('year', 0))}-{int(peak_month.get('month_num', 0)):02d} ({format_number(peak_month['demand'])} units)")

if not weekday.empty and "weekday" in weekday.columns:
    best_day = weekday.loc[weekday["demand"].idxmax(), "weekday"]
    worst_day = weekday.loc[weekday["demand"].idxmin(), "weekday"]
    insights.append(f"- **Busiest weekday**: {best_day} | **Slowest weekday**: {worst_day}")

if not category.empty:
    top_cat = category.iloc[0]["category"]
    insights.append(f"- **Top category**: {top_cat}")

st.markdown("\n".join(insights) if insights else "No demand data available for analysis.")

st.markdown("---")

# ── Inventory & Risk Insights ──
st.header("📦 Inventory & Risk Insights")
inv_insights = []

if not inv_df.empty:
    low_stock = len(inv_df[inv_df["stock_on_hand"] <= inv_df["reorder_point"]]) if "reorder_point" in inv_df.columns else 0
    inv_insights.append(f"- **Total inventory units**: {format_number(total_stock)}")
    inv_insights.append(f"- **SKUs below reorder point**: {format_number(low_stock)}")

if not risk_df.empty and "risk_level" in risk_df.columns:
    risk_summary = risk_df["risk_level"].value_counts().to_dict()
    breakdown = " | ".join([f"**{k}**: {v}" for k, v in risk_summary.items()])
    inv_insights.append(f"- **Risk breakdown**: {breakdown}")
    if "risk_score" in risk_df.columns:
        inv_insights.append(f"- **Average risk score**: {risk_df['risk_score'].mean():.1f}")

st.markdown("\n".join(inv_insights) if inv_insights else "No inventory data available for analysis.")

st.markdown("---")

# ── Forecasting Insights ──
st.header("🔮 Forecasting Insights")
fc_insights = []

if not models_df.empty:
    best = models_df.iloc[0]
    fc_insights.append(f"- **Best model**: {best.get('Model', '—')}")
    fc_insights.append(f"- **MAPE**: {best.get('MAPE_percent', 0):.1f}% | **RMSE**: {format_number(best.get('RMSE', 0))} | **MAE**: {format_number(best.get('MAE', 0))}")

if not forecast_df.empty:
    future = forecast_df[forecast_df["forecast"].notna()]
    if not future.empty:
        avg_forecast = future["forecast"].mean()
        fc_insights.append(f"- **Average forecasted demand**: {format_number(avg_forecast)} units/day")

st.markdown("\n".join(fc_insights) if fc_insights else "No forecasting data available.")

st.markdown("---")

# ── Actionable Recommendations ──
st.header("💡 Recommendations")

recs = []
if critical_risk > 0:
    recs.append(f"🔴 **Urgent**: {critical_risk} SKU(s) are at **Critical** risk. Review reorder quantities and supplier lead times immediately.")
if high_risk > 0:
    recs.append(f"🟠 **Action needed**: {high_risk} SKU(s) are at **High** risk. Schedule restocking within the next planning cycle.")
if not inv_df.empty and "reorder_point" in inv_df.columns:
    low_stock = len(inv_df[inv_df["stock_on_hand"] <= inv_df["reorder_point"]])
    if low_stock > 0:
        recs.append(f"📦 **Replenishment**: {low_stock} items are at or below their reorder point.")
if not models_df.empty:
    mape = models_df.iloc[0].get("MAPE_percent", 999)
    if mape < 10:
        recs.append(f"✅ **Forecast quality**: MAPE of {mape:.1f}% indicates strong predictive accuracy.")
    else:
        recs.append(f"⚠️ **Forecast quality**: MAPE of {mape:.1f}% — consider refining the model or adding features.")

if recs:
    for r in recs:
        st.markdown(r)
else:
    st.success("All systems healthy. No immediate actions required.")

st.info("This summary is auto-generated from the latest available data snapshots.")
