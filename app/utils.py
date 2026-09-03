import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = BASE_DIR / "data raw" / "retail_clean_dataset"
OUTPUT_DIR = BASE_DIR / "outputs"
SUMMARY_DIR = DATA_DIR / "dashboard_summaries"

COLORS = {
    "navy": "#0B1220", "navy2": "#111827", "blue": "#3B82F6", "cyan": "#06B6D4",
    "teal": "#14B8A6", "green": "#22C55E", "yellow": "#EAB308", "orange": "#F97316",
    "red": "#EF4444", "purple": "#8B5CF6", "pink": "#EC4899", "text": "#E5E7EB",
    "muted": "#94A3B8", "border": "#263244", "panel": "#151F2E",
}

RISK_COLORS = {
    "Critical": COLORS["red"], "High": COLORS["orange"],
    "Medium": COLORS["yellow"], "Low": COLORS["green"],
}

def inject_custom_css():
    st.markdown(
        """
        <style>
        .stApp { background: #0B1220; color: #E5E7EB; }
        .main .block-container { padding-top: 0rem; padding-bottom: 3rem; margin-top: -1rem; }
        section[data-testid="stSidebar"] { background: #111827; border-right: 1px solid #263244; }
        section[data-testid="stSidebar"] * { color: #E5E7EB; }
        h1, h2, h3, h4 { color: #F8FAFC !important; }
        p, label, span { color: #CBD5E1; }
        hr { border-color: #263244 !important; }
        div[data-testid="stMetric"] { background: #151F2E; border: 1px solid #263244; border-radius: 14px; padding: 16px; box-shadow: 0 4px 18px rgba(0,0,0,0.18); }
        div[data-testid="stMetric"] label { color: #94A3B8 !important; white-space: normal !important; overflow: visible !important; text-overflow: clip !important; word-break: break-word !important; font-size: 0.9rem !important; }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #F8FAFC !important; white-space: normal !important; overflow: visible !important; text-overflow: clip !important; word-break: break-word !important; font-size: 1.8rem !important; }
        div[data-testid="stMetric"] [data-testid="stMetricDelta"] { color: #94A3B8 !important; }
        div[data-baseweb="select"] > div { background: #151F2E; border-color: #334155; color: #E5E7EB; }
        div[data-baseweb="input"] { background: #151F2E; }
        div[data-baseweb="input"] input { color: #F8FAFC !important; }
        div[data-testid="stDataFrame"] { border: 1px solid #263244; border-radius: 12px; overflow: hidden; }
        .stDownloadButton button { background: #2563EB; color: white; border: none; border-radius: 8px; }
        .stDownloadButton button:hover { background: #1D4ED8; color: white; }
        div[data-testid="stAlert"] { border-radius: 12px; }
        div[role="radiogroup"] label { color: #CBD5E1 !important; }
        section[data-testid="stSidebar"] h1 { color: #F8FAFC !important; }
        header[data-testid="stHeader"] { background: transparent !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

def existing_path(*paths):
    for path in paths:
        if Path(path).exists():
            return Path(path)
    return None

def safe_read_csv(path, **kwargs):
    try:
        if path is not None and Path(path).exists():
            return pd.read_csv(path, **kwargs)
    except Exception:
        pass
    return pd.DataFrame()

def format_number(value):
    if value is None or pd.isna(value): return "—"
    value = float(value)
    if abs(value) >= 1_000_000_000: return f"{value / 1_000_000_000:.2f}B"
    if abs(value) >= 1_000_000: return f"{value / 1_000_000:.2f}M"
    if abs(value) >= 1_000: return f"{value / 1_000:.1f}K"
    return f"{value:,.0f}"

def format_currency(value):
    if value is None or pd.isna(value): return "—"
    value = float(value)
    if abs(value) >= 1_000_000: return f"${value / 1_000_000:.2f}M"
    if abs(value) >= 1_000: return f"${value / 1_000:.1f}K"
    return f"${value:,.0f}"

def pct(value):
    if value is None or pd.isna(value): return "—"
    return f"{float(value):.1f}%"

def add_download_button(df, filename, label="Download data"):
    if df is None or df.empty: return
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv, file_name=filename, mime="text/csv")

def apply_chart_theme(fig, height=430, title=""):
    fig.update_layout(
        template="plotly_dark", height=height, paper_bgcolor=COLORS["panel"], plot_bgcolor=COLORS["panel"],
        font=dict(family="Arial, sans-serif", size=13, color=COLORS["text"]),
        title=dict(text=""),
        margin=dict(l=55, r=30, t=20, b=50),
        hoverlabel=dict(bgcolor="#111827", font=dict(color="#F8FAFC", size=13)),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.01,
            xanchor="right", x=1,
            font=dict(color="#CBD5E1", size=11),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
        ),
    )
    fig.update_xaxes(showgrid=False, linecolor="#334155", tickfont=dict(color="#94A3B8"), title_font=dict(color="#CBD5E1"))
    fig.update_yaxes(showgrid=True, gridcolor="#263244", zeroline=False, tickfont=dict(color="#94A3B8"), title_font=dict(color="#CBD5E1"))
    return fig

@st.cache_data(show_spinner=False)
def load_sku_master():
    path = existing_path(DATA_DIR / "sku_master.csv", RAW_DIR / "sku_master.csv")
    df = safe_read_csv(path)
    if not df.empty and "sku_id" in df.columns: df["sku_id"] = df["sku_id"].astype(str)
    return df

@st.cache_data(show_spinner=False)
def load_inventory():
    path = existing_path(DATA_DIR / "inventory_snapshot.csv", DATA_DIR / "inventory_snapshots.csv", RAW_DIR / "inventory_snapshot.csv", RAW_DIR / "inventory_snapshots.csv")
    df = safe_read_csv(path)
    if not df.empty:
        if "sku_id" in df.columns: df["sku_id"] = df["sku_id"].astype(str)
        numeric_cols = ["stock_on_hand", "reorder_point", "safety_stock"]
        for col in numeric_cols:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df

@st.cache_data(show_spinner=False)
def load_risk():
    path = OUTPUT_DIR / "inventory_risk_scoring.csv"
    df = safe_read_csv(path)
    if not df.empty:
        if "sku_id" in df.columns: df["sku_id"] = df["sku_id"].astype(str)
        if "risk_category" in df.columns and "risk_level" not in df.columns: df["risk_level"] = df["risk_category"]
        if "risk_level" in df.columns: df["risk_level"] = df["risk_level"].astype(str).str.strip().str.title()
        if "risk_score" in df.columns: df["risk_score"] = pd.to_numeric(df["risk_score"], errors="coerce").fillna(0)
    return df

@st.cache_data(show_spinner=False)
def load_model_results():
    path = OUTPUT_DIR / "final_model_comparison.csv"
    df = safe_read_csv(path)
    if not df.empty:
        for col in ["MAE", "RMSE", "MAPE_percent"]:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

@st.cache_data(show_spinner=False)
def load_dashboard_summaries():
    daily = safe_read_csv(SUMMARY_DIR / "dashboard_daily.csv")
    store = safe_read_csv(SUMMARY_DIR / "dashboard_store.csv")
    sku = safe_read_csv(SUMMARY_DIR / "dashboard_sku.csv")
    monthly = safe_read_csv(SUMMARY_DIR / "dashboard_monthly.csv")
    category = safe_read_csv(SUMMARY_DIR / "dashboard_category.csv")
    weekday = safe_read_csv(SUMMARY_DIR / "dashboard_weekday.csv")

    if not daily.empty:
        if "date" in daily.columns: daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
        if "demand" in daily.columns: daily["demand"] = pd.to_numeric(daily["demand"], errors="coerce").fillna(0)
        daily = daily.dropna(subset=["date"]).sort_values("date")

    if not store.empty:
        if "store_id" in store.columns: store["store_id"] = store["store_id"].astype(str)
        if "demand" in store.columns: store["demand"] = pd.to_numeric(store["demand"], errors="coerce").fillna(0)
        store = store.sort_values("demand", ascending=False)

    if not sku.empty:
        if "sku_id" in sku.columns: sku["sku_id"] = sku["sku_id"].astype(str)
        if "demand" in sku.columns: sku["demand"] = pd.to_numeric(sku["demand"], errors="coerce").fillna(0)
        sku = sku.sort_values("demand", ascending=False)

    if not monthly.empty:
        for col in ["year", "month_num"]:
            if col in monthly.columns: monthly[col] = pd.to_numeric(monthly[col], errors="coerce")
        if "demand" in monthly.columns: monthly["demand"] = pd.to_numeric(monthly["demand"], errors="coerce").fillna(0)
        sort_cols = [col for col in ["year", "month_num"] if col in monthly.columns]
        if sort_cols: monthly = monthly.sort_values(sort_cols)

    if not category.empty:
        if "demand" in category.columns: category["demand"] = pd.to_numeric(category["demand"], errors="coerce").fillna(0)
        category = category.sort_values("demand", ascending=False)

    if not weekday.empty:
        if "demand" in weekday.columns: weekday["demand"] = pd.to_numeric(weekday["demand"], errors="coerce").fillna(0)
        if "weekday" in weekday.columns:
            order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            weekday["weekday"] = pd.Categorical(weekday["weekday"], categories=order, ordered=True)
            weekday = weekday.sort_values("weekday")

    return daily, store, sku, monthly, category, weekday

@st.cache_data(show_spinner=False)
def load_precomputed_forecast():
    path = OUTPUT_DIR / "prophet_forecast_predictions.csv"
    df = safe_read_csv(path)
    if df.empty: return pd.DataFrame()
    if "date" in df.columns: df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["actual", "forecast", "lower_bound", "upper_bound"]:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
    required_columns = ["date", "forecast"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns: return pd.DataFrame()
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    df["forecast"] = df["forecast"].fillna(0).clip(lower=0)
    if "actual" in df.columns: df["actual"] = df["actual"].fillna(0).clip(lower=0)
    if "lower_bound" in df.columns: df["lower_bound"] = df["lower_bound"].fillna(0).clip(lower=0)
    if "upper_bound" in df.columns: df["upper_bound"] = df["upper_bound"].fillna(0).clip(lower=0)
    return df
