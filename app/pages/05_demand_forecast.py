import streamlit as st
import plotly.graph_objects as go
from utils import (
    load_precomputed_forecast, load_model_results,
    format_number, pct, apply_chart_theme, COLORS
)

st.title("Demand Forecast")
st.markdown("Prophet-based machine learning forecast for future demand trends.")

forecast_df = load_precomputed_forecast()
models_df = load_model_results()

if not forecast_df.empty:
    st.subheader("Forecasted vs Actual Demand")
    fig = go.Figure()
    if "actual" in forecast_df.columns:
        fig.add_trace(go.Scatter(x=forecast_df["date"], y=forecast_df["actual"], mode="lines", name="Actual Demand", line=dict(color=COLORS["blue"])))
    fig.add_trace(go.Scatter(x=forecast_df["date"], y=forecast_df["forecast"], mode="lines", name="Forecast", line=dict(color=COLORS["teal"], dash="dot")))
    
    if "upper_bound" in forecast_df.columns and "lower_bound" in forecast_df.columns:
        fig.add_trace(go.Scatter(x=forecast_df["date"].tolist() + forecast_df["date"].tolist()[::-1],
                                 y=forecast_df["upper_bound"].tolist() + forecast_df["lower_bound"].tolist()[::-1],
                                 fill='toself', fillcolor='rgba(20, 184, 166, 0.2)', line=dict(color='rgba(255,255,255,0)'),
                                 name="Confidence Interval"))
        
    st.plotly_chart(apply_chart_theme(fig, height=450), use_container_width=True)
else:
    st.info("Forecast predictions are not available.")

st.markdown("---")
st.subheader("Forecast Model Accuracy")

if not models_df.empty:
    best_model = models_df.iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Best Model", best_model.get("Model", "Prophet"))
    c2.metric("MAPE (Mean Absolute Pct Error)", pct(best_model.get("MAPE_percent", 0)))
    c3.metric("RMSE", format_number(best_model.get("RMSE", 0)))
    
    st.dataframe(models_df, use_container_width=True)
else:
    st.info("Model accuracy results are not available.")
