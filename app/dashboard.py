import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Project FORESIGHT",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DIR = BASE_DIR / "data raw" / "retail_clean_dataset"
OUTPUT_DIR = BASE_DIR / "outputs"

SUMMARY_DIR = DATA_DIR / "dashboard_summaries"


# ============================================================
# COLOR THEME
# ============================================================

COLORS = {
    "navy": "#0B1220",
    "navy2": "#111827",
    "blue": "#3B82F6",
    "cyan": "#06B6D4",
    "teal": "#14B8A6",
    "green": "#22C55E",
    "yellow": "#EAB308",
    "orange": "#F97316",
    "red": "#EF4444",
    "purple": "#8B5CF6",
    "pink": "#EC4899",
    "text": "#E5E7EB",
    "muted": "#94A3B8",
    "border": "#263244",
    "panel": "#151F2E",
}


RISK_COLORS = {
    "Critical": COLORS["red"],
    "High": COLORS["orange"],
    "Medium": COLORS["yellow"],
    "Low": COLORS["green"],
}


# ============================================================
# DASHBOARD CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main application background */
    .stApp {
        background: #0B1220;
        color: #E5E7EB;
    }

    /* Main content */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid #263244;
    }

    section[data-testid="stSidebar"] * {
        color: #E5E7EB;
    }

    /* Titles */
    h1, h2, h3, h4 {
        color: #F8FAFC !important;
    }

    /* Normal text */
    p, label, span {
        color: #CBD5E1;
    }

    /* Dividers */
    hr {
        border-color: #263244 !important;
    }

    /* Metric boxes */
    div[data-testid="stMetric"] {
        background: #151F2E;
        border: 1px solid #263244;
        border-radius: 14px;
        padding: 16px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.18);
    }

    div[data-testid="stMetric"] label {
        color: #94A3B8 !important;
    }

    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #F8FAFC !important;
    }

    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: #94A3B8 !important;
    }

    /* Select boxes */
    div[data-baseweb="select"] > div {
        background: #151F2E;
        border-color: #334155;
        color: #E5E7EB;
    }

    /* Text input */
    div[data-baseweb="input"] {
        background: #151F2E;
    }

    div[data-baseweb="input"] input {
        color: #F8FAFC !important;
    }

    /* Dataframes */
    div[data-testid="stDataFrame"] {
        border: 1px solid #263244;
        border-radius: 12px;
        overflow: hidden;
    }

    /* Buttons */
    .stDownloadButton button {
        background: #2563EB;
        color: white;
        border: none;
        border-radius: 8px;
    }

    .stDownloadButton button:hover {
        background: #1D4ED8;
        color: white;
    }

    /* Info / warning / success blocks */
    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    /* Radio buttons */
    div[role="radiogroup"] label {
        color: #CBD5E1 !important;
    }

    /* Sidebar title */
    section[data-testid="stSidebar"] h1 {
        color: #F8FAFC !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

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

    if value is None or pd.isna(value):
        return "—"

    value = float(value)

    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"

    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"

    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"

    return f"{value:,.0f}"


def format_currency(value):

    if value is None or pd.isna(value):
        return "—"

    value = float(value)

    if abs(value) >= 10_000_000:
        return f"₹{value / 10_000_000:.2f}Cr"

    if abs(value) >= 100_000:
        return f"₹{value / 100_000:.2f}L"

    if abs(value) >= 1_000:
        return f"₹{value / 1_000:.1f}K"

    return f"₹{value:,.0f}"


def pct(value):

    if value is None or pd.isna(value):
        return "—"

    return f"{float(value):.1f}%"


def add_download_button(
    df,
    filename,
    label="Download data",
):

    if df is None or df.empty:
        return

    csv = df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label=label,
        data=csv,
        file_name=filename,
        mime="text/csv",
    )


# ============================================================
# PLOTLY THEME
# ============================================================

def apply_chart_theme(
    fig,
    height=430,
):

    fig.update_layout(

        template="plotly_dark",

        height=height,

        paper_bgcolor=COLORS["panel"],
        plot_bgcolor=COLORS["panel"],

        font=dict(
            family="Arial, sans-serif",
            size=13,
            color=COLORS["text"],
        ),

        title=dict(
            font=dict(
                size=19,
                color="#F8FAFC",
            ),
            x=0.01,
            xanchor="left",
        ),

        margin=dict(
            l=55,
            r=30,
            t=70,
            b=55,
        ),

        hoverlabel=dict(
            bgcolor="#111827",
            font=dict(
                color="#F8FAFC",
                size=13,
            ),
        ),

        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(
                color="#CBD5E1"
            ),
        ),
    )

    fig.update_xaxes(
        showgrid=False,
        linecolor="#334155",
        tickfont=dict(
            color="#94A3B8"
        ),
        title_font=dict(
            color="#CBD5E1"
        ),
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="#263244",
        zeroline=False,
        tickfont=dict(
            color="#94A3B8"
        ),
        title_font=dict(
            color="#CBD5E1"
        ),
    )

    return fig


# ============================================================
# MASTER DATA
# ============================================================

@st.cache_data(show_spinner=False)
def load_sku_master():

    path = existing_path(
        DATA_DIR / "sku_master.csv",
        RAW_DIR / "sku_master.csv",
    )

    df = safe_read_csv(path)

    if not df.empty and "sku_id" in df.columns:

        df["sku_id"] = (
            df["sku_id"]
            .astype(str)
        )

    return df


@st.cache_data(show_spinner=False)
def load_inventory():

    path = existing_path(
        DATA_DIR / "inventory_snapshot.csv",
        DATA_DIR / "inventory_snapshots.csv",
        RAW_DIR / "inventory_snapshot.csv",
        RAW_DIR / "inventory_snapshots.csv",
    )

    df = safe_read_csv(path)

    if not df.empty:

        if "sku_id" in df.columns:

            df["sku_id"] = (
                df["sku_id"]
                .astype(str)
            )

        numeric_cols = [
            "stock_on_hand",
            "reorder_point",
            "safety_stock",
        ]

        for col in numeric_cols:

            if col in df.columns:

                df[col] = pd.to_numeric(
                    df[col],
                    errors="coerce",
                ).fillna(0)

    return df


@st.cache_data(show_spinner=False)
def load_risk():

    path = (
        OUTPUT_DIR
        / "inventory_risk_scoring.csv"
    )

    df = safe_read_csv(path)

    if not df.empty:

        if "sku_id" in df.columns:

            df["sku_id"] = (
                df["sku_id"]
                .astype(str)
            )

        if (
            "risk_category" in df.columns
            and "risk_level"
            not in df.columns
        ):

            df["risk_level"] = (
                df["risk_category"]
            )

        if "risk_level" in df.columns:

            df["risk_level"] = (
                df["risk_level"]
                .astype(str)
                .str.strip()
                .str.title()
            )

        if "risk_score" in df.columns:

            df["risk_score"] = pd.to_numeric(
                df["risk_score"],
                errors="coerce",
            ).fillna(0)

    return df


@st.cache_data(show_spinner=False)
def load_model_results():

    path = (
        OUTPUT_DIR
        / "final_model_comparison.csv"
    )

    df = safe_read_csv(path)

    if not df.empty:

        for col in [
            "MAE",
            "RMSE",
            "MAPE_percent",
        ]:

            if col in df.columns:

                df[col] = pd.to_numeric(
                    df[col],
                    errors="coerce",
                )

    return df


# ============================================================
# FAST DASHBOARD SUMMARIES
# ============================================================

@st.cache_data(show_spinner=False)
def load_dashboard_summaries():

    """
    Uses the files generated by:

        prepare_dashboard_data.py

    This avoids scanning the huge daily_demand.csv
    every time the dashboard starts.
    """

    daily = safe_read_csv(
        SUMMARY_DIR
        / "dashboard_daily.csv"
    )

    store = safe_read_csv(
        SUMMARY_DIR
        / "dashboard_store.csv"
    )

    sku = safe_read_csv(
        SUMMARY_DIR
        / "dashboard_sku.csv"
    )

    monthly = safe_read_csv(
        SUMMARY_DIR
        / "dashboard_monthly.csv"
    )

    category = safe_read_csv(
        SUMMARY_DIR
        / "dashboard_category.csv"
    )

    weekday = safe_read_csv(
        SUMMARY_DIR
        / "dashboard_weekday.csv"
    )

    # --------------------------------------------------------
    # DAILY
    # --------------------------------------------------------

    if not daily.empty:

        if "date" in daily.columns:

            daily["date"] = pd.to_datetime(
                daily["date"],
                errors="coerce",
            )

        if "demand" in daily.columns:

            daily["demand"] = pd.to_numeric(
                daily["demand"],
                errors="coerce",
            ).fillna(0)

        daily = (
            daily
            .dropna(subset=["date"])
            .sort_values("date")
        )

    # --------------------------------------------------------
    # STORE
    # --------------------------------------------------------

    if not store.empty:

        if "store_id" in store.columns:

            store["store_id"] = (
                store["store_id"]
                .astype(str)
            )

        if "demand" in store.columns:

            store["demand"] = pd.to_numeric(
                store["demand"],
                errors="coerce",
            ).fillna(0)

        store = store.sort_values(
            "demand",
            ascending=False,
        )

    # --------------------------------------------------------
    # SKU
    # --------------------------------------------------------

    if not sku.empty:

        if "sku_id" in sku.columns:

            sku["sku_id"] = (
                sku["sku_id"]
                .astype(str)
            )

        if "demand" in sku.columns:

            sku["demand"] = pd.to_numeric(
                sku["demand"],
                errors="coerce",
            ).fillna(0)

        sku = sku.sort_values(
            "demand",
            ascending=False,
        )

    # --------------------------------------------------------
    # MONTHLY
    # --------------------------------------------------------

    if not monthly.empty:

        for col in [
            "year",
            "month_num",
        ]:

            if col in monthly.columns:

                monthly[col] = pd.to_numeric(
                    monthly[col],
                    errors="coerce",
                )

        if "demand" in monthly.columns:

            monthly["demand"] = pd.to_numeric(
                monthly["demand"],
                errors="coerce",
            ).fillna(0)

        sort_cols = [
            col
            for col in [
                "year",
                "month_num",
            ]
            if col in monthly.columns
        ]

        if sort_cols:

            monthly = monthly.sort_values(
                sort_cols
            )

    # --------------------------------------------------------
    # CATEGORY
    # --------------------------------------------------------

    if not category.empty:

        if "demand" in category.columns:

            category["demand"] = pd.to_numeric(
                category["demand"],
                errors="coerce",
            ).fillna(0)

        category = category.sort_values(
            "demand",
            ascending=False,
        )

    # --------------------------------------------------------
    # WEEKDAY
    # --------------------------------------------------------

    if not weekday.empty:

        if "demand" in weekday.columns:

            weekday["demand"] = pd.to_numeric(
                weekday["demand"],
                errors="coerce",
            ).fillna(0)

        if "weekday" in weekday.columns:

            order = [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ]

            weekday["weekday"] = pd.Categorical(
                weekday["weekday"],
                categories=order,
                ordered=True,
            )

            weekday = weekday.sort_values(
                "weekday"
            )

    return (
        daily,
        store,
        sku,
        monthly,
        category,
        weekday,
    )


# ============================================================
# PRECOMPUTED FORECAST DATA
# ============================================================

@st.cache_data(show_spinner=False)
def load_precomputed_forecast():

    """
    Loads the small precomputed Prophet forecast output.

    This file is intentionally used instead of the large
    daily_demand.csv so that the dashboard can run on
    Streamlit Community Cloud.
    """

    path = (
        OUTPUT_DIR
        / "prophet_forecast_predictions.csv"
    )

    df = safe_read_csv(path)

    if df.empty:
        return pd.DataFrame()

    # --------------------------------------------------------
    # DATE
    # --------------------------------------------------------

    if "date" in df.columns:

        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce",
        )

    # --------------------------------------------------------
    # NUMERIC COLUMNS
    # --------------------------------------------------------

    for col in [
        "actual",
        "forecast",
        "lower_bound",
        "upper_bound",
    ]:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

    # --------------------------------------------------------
    # CLEAN
    # --------------------------------------------------------

    required_columns = [
        "date",
        "forecast",
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        return pd.DataFrame()

    df = (
        df
        .dropna(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    df["forecast"] = (
        df["forecast"]
        .fillna(0)
        .clip(lower=0)
    )

    if "actual" in df.columns:

        df["actual"] = (
            df["actual"]
            .fillna(0)
            .clip(lower=0)
        )

    if "lower_bound" in df.columns:

        df["lower_bound"] = (
            df["lower_bound"]
            .fillna(0)
            .clip(lower=0)
        )

    if "upper_bound" in df.columns:

        df["upper_bound"] = (
            df["upper_bound"]
            .fillna(0)
            .clip(lower=0)
        )

    return df


# ============================================================
# LOAD DATA
# ============================================================

sku_master = load_sku_master()

inventory = load_inventory()

risk = load_risk()

model_results = load_model_results()

(
    daily_demand,
    store_summary,
    sku_summary,
    monthly_summary,
    category_summary,
    weekday_summary,
) = load_dashboard_summaries()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "📊 Project FORESIGHT"
)

st.sidebar.caption(
    "Demand Forecasting & Inventory Intelligence"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Home Page",
        "Sales Analytics",
        "Forecast",
        "Inventory Dashboard",
        "Risk Dashboard",
        "Product Details",
        "Executive Summary",
    ],
)


# ============================================================
# HOME PAGE
# ============================================================

if page == "Home Page":

    st.title(
        "📊 Project FORESIGHT"
    )

    st.subheader(
        "Demand Forecasting & Inventory Intelligence"
    )

    st.write(
        "A retail analytics dashboard for understanding "
        "demand, forecasting future requirements, monitoring "
        "inventory, and identifying inventory risk."
    )

    st.divider()

    # --------------------------------------------------------
    # KPI
    # --------------------------------------------------------

    total_demand = (
        daily_demand["demand"].sum()
        if not daily_demand.empty
        else 0
    )

    stores = (
        store_summary["store_id"].nunique()
        if not store_summary.empty
        else 0
    )

    skus = (
        sku_master["sku_id"].nunique()
        if not sku_master.empty
        else 0
    )

    categories = (
        sku_master["category"].nunique()
        if (
            not sku_master.empty
            and "category"
            in sku_master.columns
        )
        else 0
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Demand",
        format_number(total_demand),
    )

    c2.metric(
        "Stores",
        format_number(stores),
    )

    c3.metric(
        "SKUs",
        format_number(skus),
    )

    c4.metric(
        "Categories",
        format_number(categories),
    )

    st.divider()

    # --------------------------------------------------------
    # MAIN VISUAL
    # --------------------------------------------------------

    if not daily_demand.empty:

        st.subheader(
            "📈 Overall demand trend"
        )

        fig = px.area(
            daily_demand,
            x="date",
            y="demand",
            title="Daily Retail Demand",
        )

        fig.update_traces(
            line=dict(
                color=COLORS["cyan"],
                width=3,
            ),
            fillcolor="rgba(6,182,212,0.15)",
        )

        apply_chart_theme(
            fig,
            height=480,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.divider()

    # --------------------------------------------------------
    # DASHBOARD AREAS
    # --------------------------------------------------------

    st.subheader(
        "🎯 FORESIGHT at a glance"
    )

    a, b, c = st.columns(3)

    with a:

        st.info(
            "**📊 Sales Analytics**\n\n"
            "Track demand trends, categories, stores, "
            "weekdays, and monthly performance."
        )

    with b:

        st.success(
            "**🔮 Forecasting**\n\n"
            "Review the precomputed Prophet forecast "
            "and understand expected demand."
        )

    with c:

        st.warning(
            "**⚠️ Risk Intelligence**\n\n"
            "Identify Critical and High-risk inventory "
            "positions requiring action."
        )


# ============================================================
# SALES ANALYTICS
# ============================================================

elif page == "Sales Analytics":

    st.title(
        "📊 Sales Analytics"
    )

    st.caption(
        "Explore demand patterns across time, categories, "
        "stores, and weekdays."
    )

    if daily_demand.empty:

        st.error(
            "Dashboard daily summary could not be loaded."
        )

        st.stop()

    min_date = (
        daily_demand["date"]
        .min()
        .date()
    )

    max_date = (
        daily_demand["date"]
        .max()
        .date()
    )

    selected_dates = st.date_input(
        "Select analysis period",
        value=(
            min_date,
            max_date,
        ),
        min_value=min_date,
        max_value=max_date,
    )

    if (
        isinstance(
            selected_dates,
            tuple,
        )
        and len(selected_dates) == 2
    ):

        start_date, end_date = (
            selected_dates
        )

        filtered_daily = (
            daily_demand[
                (
                    daily_demand["date"]
                    .dt.date
                    >= start_date
                )
                &
                (
                    daily_demand["date"]
                    .dt.date
                    <= end_date
                )
            ]
            .copy()
        )

    else:

        filtered_daily = (
            daily_demand.copy()
        )

    total_period_demand = (
        filtered_daily["demand"].sum()
    )

    average_daily = (
        filtered_daily["demand"].mean()
        if not filtered_daily.empty
        else 0
    )

    peak_daily = (
        filtered_daily["demand"].max()
        if not filtered_daily.empty
        else 0
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Demand",
        format_number(
            total_period_demand
        ),
    )

    c2.metric(
        "Daily Average",
        format_number(
            average_daily
        ),
    )

    c3.metric(
        "Peak Daily Demand",
        format_number(
            peak_daily
        ),
    )

    st.divider()

    # ========================================================
    # GRAPH 1
    # ========================================================

    st.subheader(
        "📈 Demand trend"
    )

    fig = px.line(
        filtered_daily,
        x="date",
        y="demand",
        title="Daily Demand Trend",
    )

    fig.update_traces(
        line=dict(
            color=COLORS["cyan"],
            width=3,
        )
    )

    apply_chart_theme(
        fig,
        height=470,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # ========================================================
    # GRAPH 2
    # ========================================================

    st.subheader(
        "🛒 Demand by category"
    )

    if not category_summary.empty:

        fig = px.bar(
            category_summary,
            x="category",
            y="demand",
            title="Total Demand by Category",
            color="demand",
            color_continuous_scale=[
                COLORS["blue"],
                COLORS["cyan"],
                COLORS["teal"],
            ],
        )

        fig.update_layout(
            coloraxis_showscale=False
        )

        apply_chart_theme(
            fig,
            height=470,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    # ========================================================
    # GRAPH 3
    # ========================================================

    st.subheader(
        "🏬 Top stores"
    )

    if not store_summary.empty:

        top_stores = (
            store_summary
            .head(10)
            .sort_values(
                "demand",
                ascending=True,
            )
        )

        fig = px.bar(
            top_stores,
            x="demand",
            y="store_id",
            orientation="h",
            title="Top 10 Stores by Demand",
        )

        fig.update_traces(
            marker_color=COLORS["purple"]
        )

        apply_chart_theme(
            fig,
            height=500,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    # ========================================================
    # GRAPH 4
    # ========================================================

    st.subheader(
        "📅 Demand by weekday"
    )

    if not weekday_summary.empty:

        fig = px.bar(
            weekday_summary,
            x="weekday",
            y="demand",
            title="Demand Distribution by Weekday",
        )

        fig.update_traces(
            marker_color=COLORS["teal"]
        )

        apply_chart_theme(
            fig,
            height=450,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    # ========================================================
    # GRAPH 5
    # ========================================================

    st.subheader(
        "📆 Monthly demand trend"
    )

    if not monthly_summary.empty:

        monthly_plot = (
            monthly_summary.copy()
        )

        if (
            "year"
            in monthly_plot.columns
            and "month"
            in monthly_plot.columns
        ):

            if "month_num" in monthly_plot.columns:

                monthly_plot["period"] = (
                    monthly_plot["year"]
                    .astype(int)
                    .astype(str)
                    + "-"
                    + monthly_plot[
                        "month"
                    ].astype(str)
                )

            else:

                monthly_plot["period"] = (
                    monthly_plot["year"]
                    .astype(int)
                    .astype(str)
                    + "-"
                    + monthly_plot[
                        "month"
                    ].astype(str)
                )

        fig = px.line(
            monthly_plot,
            x="period",
            y="demand",
            markers=True,
            title="Monthly Demand Trend",
        )

        fig.update_traces(
            line=dict(
                color=COLORS["orange"],
                width=3,
            ),
            marker=dict(
                size=8
            ),
        )

        apply_chart_theme(
            fig,
            height=460,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


# ============================================================
# FORECAST
# ============================================================

elif page == "Forecast":

    st.title(
        "🔮 Demand Forecast"
    )

    st.caption(
        "Precomputed Prophet forecast validation results."
    )

    forecast_df = load_precomputed_forecast()

    # --------------------------------------------------------
    # CHECK FORECAST FILE
    # --------------------------------------------------------

    if forecast_df.empty:

        st.error(
            "Precomputed forecast data could not be loaded."
        )

        st.info(
            "Required file: "
            "`outputs/prophet_forecast_predictions.csv`"
        )

        st.stop()

    st.info(
        "This page uses the precomputed Prophet forecast "
        "stored in `outputs/prophet_forecast_predictions.csv`. "
        "The online dashboard does not need the large "
        "`daily_demand.csv` file."
    )

    st.divider()

    # --------------------------------------------------------
    # FORECAST HORIZON
    # --------------------------------------------------------

    available_rows = len(
        forecast_df
    )

    if available_rows < 7:

        forecast_days = available_rows

    else:

        forecast_days = st.slider(
            "Forecast / validation period",
            min_value=7,
            max_value=min(
                56,
                available_rows
            ),
            value=min(
                28,
                available_rows
            ),
            step=1,
        )

    display_df = (
        forecast_df
        .tail(forecast_days)
        .copy()
    )

    # --------------------------------------------------------
    # HISTORICAL DATA
    # --------------------------------------------------------

    if "actual" in forecast_df.columns:

        historical_display = (
            forecast_df[
                forecast_df["actual"] > 0
            ]
            .tail(180)
            .copy()
        )

    else:

        historical_display = (
            pd.DataFrame()
        )

    # --------------------------------------------------------
    # CHART
    # --------------------------------------------------------

    fig = go.Figure()

    if not historical_display.empty:

        fig.add_trace(
            go.Scatter(
                x=historical_display[
                    "date"
                ],
                y=historical_display[
                    "actual"
                ],
                mode="lines",
                name="Actual Demand",
                line=dict(
                    color=COLORS["cyan"],
                    width=3,
                ),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=display_df[
                "date"
            ],
            y=display_df[
                "forecast"
            ],
            mode="lines+markers",
            name="Prophet Forecast",
            line=dict(
                color=COLORS["orange"],
                width=3,
                dash="dash",
            ),
            marker=dict(
                size=6
            ),
        )
    )

    # --------------------------------------------------------
    # PREDICTION INTERVAL
    # --------------------------------------------------------

    if (
        "lower_bound"
        in display_df.columns
        and "upper_bound"
        in display_df.columns
    ):

        fig.add_trace(
            go.Scatter(
                x=display_df[
                    "date"
                ],
                y=display_df[
                    "upper_bound"
                ],
                mode="lines",
                line=dict(
                    width=0
                ),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=display_df[
                    "date"
                ],
                y=display_df[
                    "lower_bound"
                ],
                mode="lines",
                fill="tonexty",
                fillcolor=(
                    "rgba(249,115,22,0.15)"
                ),
                line=dict(
                    width=0
                ),
                name="Prediction Interval",
            )
        )

    apply_chart_theme(
        fig,
        height=560,
    )

    fig.update_layout(
        title=(
            "Actual Demand vs Prophet Forecast"
        ),
        hovermode="x unified",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # FORECAST KPIs
    # --------------------------------------------------------

    total_forecast = (
        display_df[
            "forecast"
        ].sum()
    )

    avg_forecast = (
        display_df[
            "forecast"
        ].mean()
    )

    max_forecast = (
        display_df[
            "forecast"
        ].max()
    )

    f1, f2, f3 = st.columns(3)

    f1.metric(
        "Forecast Demand",
        format_number(
            total_forecast
        ),
    )

    f2.metric(
        "Average Daily Forecast",
        format_number(
            avg_forecast
        ),
    )

    f3.metric(
        "Peak Forecast",
        format_number(
            max_forecast
        ),
    )

    st.divider()

    # --------------------------------------------------------
    # MODEL PERFORMANCE
    # --------------------------------------------------------

    st.subheader(
        "🤖 Model Performance"
    )

    if not model_results.empty:

        performance = (
            model_results.copy()
        )

        if "MAE" in performance.columns:

            performance = (
                performance
                .sort_values(
                    "MAE",
                    ascending=True,
                )
                .reset_index(
                    drop=True
                )
            )

        st.dataframe(
            performance,
            use_container_width=True,
            hide_index=True,
        )

        if (
            "MAE"
            in performance.columns
            and not performance.empty
        ):

            best_model_row = (
                performance.iloc[0]
            )

            if "model" in performance.columns:

                best_model = str(
                    best_model_row[
                        "model"
                    ]
                )

            elif "Model" in performance.columns:

                best_model = str(
                    best_model_row[
                        "Model"
                    ]
                )

            else:

                best_model = "Best Model"

            best_mae = (
                best_model_row[
                    "MAE"
                ]
            )

            st.success(
                f"Best performing model: "
                f"**{best_model}** "
                f"with MAE of "
                f"**{best_mae:,.2f}**"
            )

    else:

        st.warning(
            "Model comparison output is not available."
        )

    st.divider()

    # --------------------------------------------------------
    # FORECAST TABLE
    # --------------------------------------------------------

    st.subheader(
        "📋 Forecast Details"
    )

    table_df = (
        display_df.copy()
    )

    rename_map = {
        "date": "Date",
        "actual": "Actual Demand",
        "forecast": "Forecast",
        "lower_bound": "Lower Bound",
        "upper_bound": "Upper Bound",
    }

    table_df = table_df.rename(
        columns=rename_map
    )

    if "Date" in table_df.columns:

        table_df["Date"] = pd.to_datetime(
            table_df["Date"],
            errors="coerce",
        ).dt.strftime(
            "%Y-%m-%d"
        )

    for col in [
        "Actual Demand",
        "Forecast",
        "Lower Bound",
        "Upper Bound",
    ]:

        if col in table_df.columns:

            table_df[col] = (
                pd.to_numeric(
                    table_df[col],
                    errors="coerce",
                )
                .round(2)
            )

    st.dataframe(
        table_df,
        use_container_width=True,
        hide_index=True,
    )

    add_download_button(
        table_df,
        "prophet_forecast_validation.csv",
        "⬇️ Download Forecast CSV",
    )


# ============================================================
# INVENTORY DASHBOARD
# ============================================================

elif page == "Inventory Dashboard":

    st.title(
        "📦 Inventory Dashboard"
    )

    st.caption(
        "Monitor stock availability and replenishment pressure."
    )

    if inventory.empty:

        st.error(
            "Inventory snapshot data could not be loaded."
        )

    else:

        inv = inventory.copy()

        # ----------------------------------------------------
        # MERGE PRODUCT DATA
        # ----------------------------------------------------

        if not sku_master.empty:

            merge_cols = [
                col
                for col in [
                    "sku_id",
                    "sku_name",
                    "category",
                    "subcategory",
                ]
                if col in sku_master.columns
            ]

            inv = inv.merge(
                sku_master[
                    merge_cols
                ],
                on="sku_id",
                how="left",
            )

        # ----------------------------------------------------
        # KPI
        # ----------------------------------------------------

        total_stock = (
            inv["stock_on_hand"]
            .sum()
        )

        avg_stock = (
            inv["stock_on_hand"]
            .mean()
        )

        below_reorder = (
            inv["stock_on_hand"]
            <
            inv["reorder_point"]
        ).sum()

        zero_stock = (
            inv["stock_on_hand"]
            <= 0
        ).sum()

        c1, c2, c3, c4 = (
            st.columns(4)
        )

        c1.metric(
            "Total Stock",
            format_number(
                total_stock
            ),
        )

        c2.metric(
            "Average Stock",
            f"{avg_stock:.1f}",
        )

        c3.metric(
            "Below Reorder",
            format_number(
                below_reorder
            ),
        )

        c4.metric(
            "Zero Stock",
            format_number(
                zero_stock
            ),
        )

        st.divider()

        # ----------------------------------------------------
        # FILTERS
        # ----------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            categories = [
                "All"
            ]

            if "category" in inv.columns:

                categories.extend(
                    sorted(
                        inv[
                            "category"
                        ]
                        .dropna()
                        .astype(str)
                        .unique()
                        .tolist()
                    )
                )

            selected_category = (
                st.selectbox(
                    "Category",
                    categories,
                )
            )

        with col2:

            stock_filter = (
                st.selectbox(
                    "Stock status",
                    [
                        "All",
                        "Zero stock",
                        "Below safety stock",
                        "Below reorder point",
                        "Healthy",
                    ],
                )
            )

        filtered = inv.copy()

        if (
            selected_category
            != "All"
        ):

            filtered = filtered[
                filtered[
                    "category"
                ]
                .astype(str)
                == selected_category
            ]

        if stock_filter == "Zero stock":

            filtered = filtered[
                filtered[
                    "stock_on_hand"
                ]
                <= 0
            ]

        elif (
            stock_filter
            == "Below safety stock"
        ):

            filtered = filtered[
                filtered[
                    "stock_on_hand"
                ]
                <
                filtered[
                    "safety_stock"
                ]
            ]

        elif (
            stock_filter
            == "Below reorder point"
        ):

            filtered = filtered[
                filtered[
                    "stock_on_hand"
                ]
                <
                filtered[
                    "reorder_point"
                ]
            ]

        elif stock_filter == "Healthy":

            filtered = filtered[
                filtered[
                    "stock_on_hand"
                ]
                >=
                filtered[
                    "reorder_point"
                ]
            ]

        st.divider()

        # ----------------------------------------------------
        # STOCK CHART
        # ----------------------------------------------------

        st.subheader(
            "📦 Inventory position"
        )

        if not filtered.empty:

            chart_data = (
                filtered.copy()
            )

            chart_data["SKU"] = (
                chart_data[
                    "sku_id"
                ].astype(str)
            )

            top_inventory = (
                chart_data
                .sort_values(
                    "stock_on_hand",
                    ascending=False,
                )
                .head(15)
                .sort_values(
                    "stock_on_hand",
                    ascending=True,
                )
            )

            fig = px.bar(
                top_inventory,
                x="stock_on_hand",
                y="SKU",
                orientation="h",
                title="Highest Stock Positions",
            )

            fig.update_traces(
                marker_color=COLORS["teal"]
            )

            apply_chart_theme(
                fig,
                height=540,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        else:

            st.info(
                "No inventory positions match "
                "the selected filters."
            )

        # ----------------------------------------------------
        # TABLE
        # ----------------------------------------------------

        st.subheader(
            "Inventory table"
        )

        table_cols = [
            col
            for col in [
                "store_id",
                "sku_id",
                "sku_name",
                "category",
                "stock_on_hand",
                "reorder_point",
                "safety_stock",
                "last_restock_date",
            ]
            if col in filtered.columns
        ]

        if (
            table_cols
            and not filtered.empty
        ):

            table_data = (
                filtered[
                    table_cols
                ]
                .sort_values(
                    "stock_on_hand",
                    ascending=True,
                )
            )

            st.dataframe(
                table_data,
                use_container_width=True,
                hide_index=True,
            )

            add_download_button(
                table_data,
                "inventory_dashboard_export.csv",
            )


# ============================================================
# RISK DASHBOARD
# ============================================================

elif page == "Risk Dashboard":

    st.title(
        "⚠️ Risk Dashboard"
    )

    st.caption(
        "Identify inventory positions requiring attention."
    )

    if risk.empty:

        st.error(
            "Risk scoring output could not be loaded."
        )

    else:

        risk_df = risk.copy()

        # ----------------------------------------------------
        # RISK FIELD FIX
        # ----------------------------------------------------

        if (
            "risk_level"
            not in risk_df.columns
            and "risk_category"
            in risk_df.columns
        ):

            risk_df[
                "risk_level"
            ] = (
                risk_df[
                    "risk_category"
                ]
            )

        if "risk_level" in risk_df.columns:

            risk_df[
                "risk_level"
            ] = (
                risk_df[
                    "risk_level"
                ]
                .astype(str)
                .str.strip()
                .str.title()
            )

        # ----------------------------------------------------
        # PRODUCT INFO
        # ----------------------------------------------------

        if not sku_master.empty:

            merge_cols = [
                col
                for col in [
                    "sku_id",
                    "sku_name",
                    "category",
                    "unit_price",
                    "cost_price",
                ]
                if col in sku_master.columns
            ]

            risk_df = risk_df.merge(
                sku_master[
                    merge_cols
                ],
                on="sku_id",
                how="left",
                suffixes=(
                    "",
                    "_master",
                ),
            )

        # ----------------------------------------------------
        # COUNTS
        # ----------------------------------------------------

        if "risk_level" in risk_df.columns:

            critical = (
                risk_df[
                    "risk_level"
                ]
                .eq("Critical")
                .sum()
            )

            high = (
                risk_df[
                    "risk_level"
                ]
                .eq("High")
                .sum()
            )

            medium = (
                risk_df[
                    "risk_level"
                ]
                .eq("Medium")
                .sum()
            )

            low = (
                risk_df[
                    "risk_level"
                ]
                .eq("Low")
                .sum()
            )

        else:

            critical = 0
            high = 0
            medium = 0
            low = 0

        c1, c2, c3, c4 = (
            st.columns(4)
        )

        c1.metric(
            "🔴 Critical",
            format_number(
                critical
            ),
        )

        c2.metric(
            "🟠 High",
            format_number(
                high
            ),
        )

        c3.metric(
            "🟡 Medium",
            format_number(
                medium
            ),
        )

        c4.metric(
            "🟢 Low",
            format_number(
                low
            ),
        )

        st.divider()

        # ----------------------------------------------------
        # RISK DISTRIBUTION
        # ----------------------------------------------------

        if "risk_level" in risk_df.columns:

            st.subheader(
                "Risk distribution"
            )

            levels = [
                "Critical",
                "High",
                "Medium",
                "Low",
            ]

            risk_counts = (
                risk_df[
                    "risk_level"
                ]
                .value_counts()
                .reindex(
                    levels,
                    fill_value=0,
                )
                .rename_axis(
                    "Risk Level"
                )
                .reset_index(
                    name="Items"
                )
            )

            fig = px.bar(
                risk_counts,
                x="Risk Level",
                y="Items",
                color="Risk Level",
                color_discrete_map=RISK_COLORS,
                category_orders={
                    "Risk Level": levels
                },
                title="Inventory Risk Levels",
            )

            apply_chart_theme(
                fig,
                height=450,
            )

            fig.update_layout(
                showlegend=False
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        # ----------------------------------------------------
        # RISK SCORE
        # ----------------------------------------------------

        if "risk_score" in risk_df.columns:

            st.subheader(
                "Risk score distribution"
            )

            fig = px.histogram(
                risk_df,
                x="risk_score",
                nbins=20,
                title="Distribution of Risk Scores",
            )

            fig.update_traces(
                marker_color=COLORS["red"]
            )

            apply_chart_theme(
                fig,
                height=450,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        st.divider()

        # ----------------------------------------------------
        # ACTIONS
        # ----------------------------------------------------

        st.subheader(
            "🎯 Recommended inventory actions"
        )

        action_filter = (
            st.selectbox(
                "Risk level",
                [
                    "All",
                    "Critical",
                    "High",
                    "Medium",
                    "Low",
                ],
            )
        )

        filtered_risk = (
            risk_df.copy()
        )

        if action_filter != "All":

            filtered_risk = (
                filtered_risk[
                    filtered_risk[
                        "risk_level"
                    ]
                    == action_filter
                ]
            )

        def action_from_risk(
            level
        ):

            if level == "Critical":
                return "Reorder now"

            if level == "High":
                return "Prioritize replenishment"

            if level == "Medium":
                return "Watch closely"

            return "Healthy"

        if "risk_level" in filtered_risk.columns:

            filtered_risk[
                "recommended_action"
            ] = (
                filtered_risk[
                    "risk_level"
                ]
                .apply(
                    action_from_risk
                )
            )

        # ----------------------------------------------------
        # VALUE
        # ----------------------------------------------------

        if (
            "stock_on_hand"
            in filtered_risk.columns
            and "unit_price"
            in filtered_risk.columns
        ):

            filtered_risk[
                "estimated_inventory_value"
            ] = (
                filtered_risk[
                    "stock_on_hand"
                ]
                *
                pd.to_numeric(
                    filtered_risk[
                        "unit_price"
                    ],
                    errors="coerce",
                ).fillna(0)
            )

        table_cols = [
            col
            for col in [
                "store_id",
                "sku_id",
                "sku_name",
                "category",
                "risk_level",
                "risk_score",
                "stock_on_hand",
                "reorder_point",
                "safety_stock",
                "reorder_gap",
                "safety_stock_gap",
                "recommended_action",
                "estimated_inventory_value",
            ]
            if col in filtered_risk.columns
        ]

        if (
            table_cols
            and not filtered_risk.empty
        ):

            sort_column = (
                "risk_score"
                if "risk_score"
                in filtered_risk.columns
                else "stock_on_hand"
            )

            table_data = (
                filtered_risk[
                    table_cols
                ]
                .sort_values(
                    sort_column,
                    ascending=False,
                )
            )

            st.dataframe(
                table_data,
                use_container_width=True,
                hide_index=True,
            )

            add_download_button(
                table_data,
                "risk_dashboard_export.csv",
            )

        elif filtered_risk.empty:

            st.info(
                "No inventory positions match "
                "the selected risk level."
            )


# ============================================================
# PRODUCT DETAILS
# ============================================================

elif page == "Product Details":

    st.title(
        "🔎 Product Details"
    )

    st.caption(
        "Explore individual SKU information, demand, "
        "and inventory status."
    )

    if sku_master.empty:

        st.error(
            "SKU master data could not be loaded."
        )

    else:

        product_search = st.text_input(
            "Search SKU or product name",
            placeholder=(
                "Example: SKU04321 or Hair Care"
            ),
        )

        product_list = (
            sku_master.copy()
        )

        if product_search.strip():

            search_text = (
                product_search
                .strip()
                .lower()
            )

            sku_match = (
                product_list[
                    "sku_id"
                ]
                .astype(str)
                .str.lower()
                .str.contains(
                    search_text,
                    na=False,
                )
            )

            name_match = (
                product_list[
                    "sku_name"
                ]
                .astype(str)
                .str.lower()
                .str.contains(
                    search_text,
                    na=False,
                )
            )

            product_list = (
                product_list[
                    sku_match
                    |
                    name_match
                ]
            )

        if product_list.empty:

            st.warning(
                "No matching products were found."
            )

        else:

            # ------------------------------------------------
            # FULL PRODUCT NAME SELECTOR
            # ------------------------------------------------

            product_options = {}

            display_labels = []

            for _, row in (
                product_list.iterrows()
            ):

                sku_id = str(
                    row.get(
                        "sku_id",
                        "",
                    )
                )

                product_name = str(
                    row.get(
                        "sku_name",
                        "",
                    )
                )

                if (
                    not product_name
                    or product_name.lower()
                    == "nan"
                ):

                    display_name = (
                        sku_id
                    )

                else:

                    display_name = (
                        f"{sku_id} — "
                        f"{product_name}"
                    )

                display_labels.append(
                    display_name
                )

                product_options[
                    display_name
                ] = sku_id

            selected_display = (
                st.selectbox(
                    "Select product",
                    display_labels,
                    help=(
                        "The complete SKU and "
                        "product name are shown."
                    ),
                )
            )

            selected_product = (
                product_options[
                    selected_display
                ]
            )

            product = sku_master[
                sku_master[
                    "sku_id"
                ]
                .astype(str)
                == str(
                    selected_product
                )
            ]

            if not product.empty:

                row = product.iloc[0]

                full_name = str(
                    row.get(
                        "sku_name",
                        selected_product,
                    )
                )

                st.subheader(
                    f"🛍️ {full_name}"
                )

                p1, p2, p3, p4 = (
                    st.columns(4)
                )

                p1.metric(
                    "SKU",
                    str(
                        row.get(
                            "sku_id",
                            "—",
                        )
                    ),
                )

                p2.metric(
                    "Category",
                    str(
                        row.get(
                            "category",
                            "—",
                        )
                    ),
                )

                p3.metric(
                    "Unit Price",
                    format_currency(
                        row.get(
                            "unit_price",
                            np.nan,
                        )
                    ),
                )

                p4.metric(
                    "Cost Price",
                    format_currency(
                        row.get(
                            "cost_price",
                            np.nan,
                        )
                    ),
                )

                st.divider()

                # ------------------------------------------------
                # DEMAND
                # ------------------------------------------------

                product_demand = (
                    pd.DataFrame()
                )

                if not sku_summary.empty:

                    product_demand = (
                        sku_summary[
                            sku_summary[
                                "sku_id"
                            ]
                            .astype(str)
                            == str(
                                selected_product
                            )
                        ]
                    )

                if not product_demand.empty:

                    st.subheader(
                        "📈 Demand contribution"
                    )

                    product_total = (
                        product_demand[
                            "demand"
                        ].iloc[0]
                    )

                    overall_total = (
                        daily_demand[
                            "demand"
                        ].sum()
                        if not daily_demand.empty
                        else 0
                    )

                    share = (
                        product_total
                        /
                        overall_total
                        *
                        100
                        if overall_total > 0
                        else 0
                    )

                    d1, d2 = (
                        st.columns(2)
                    )

                    d1.metric(
                        "Total Demand",
                        format_number(
                            product_total
                        ),
                    )

                    d2.metric(
                        "Demand Share",
                        pct(share),
                    )

                # ------------------------------------------------
                # INVENTORY
                # ------------------------------------------------

                if not inventory.empty:

                    product_inventory = (
                        inventory[
                            inventory[
                                "sku_id"
                            ]
                            .astype(str)
                            == str(
                                selected_product
                            )
                        ]
                        .copy()
                    )

                    if not product_inventory.empty:

                        st.subheader(
                            "📦 Inventory position"
                        )

                        total_stock = (
                            product_inventory[
                                "stock_on_hand"
                            ].sum()
                        )

                        average_stock = (
                            product_inventory[
                                "stock_on_hand"
                            ].mean()
                        )

                        below_reorder = (
                            product_inventory[
                                "stock_on_hand"
                            ]
                            <
                            product_inventory[
                                "reorder_point"
                            ]
                        ).sum()

                        i1, i2, i3 = (
                            st.columns(3)
                        )

                        i1.metric(
                            "Stock on Hand",
                            format_number(
                                total_stock
                            ),
                        )

                        i2.metric(
                            "Average Stock",
                            f"{average_stock:.1f}",
                        )

                        i3.metric(
                            "Stores Below Reorder",
                            format_number(
                                below_reorder
                            ),
                        )

                        # ------------------------------------------------
                        # STORE CHART
                        # ------------------------------------------------

                        chart_inventory = (
                            product_inventory.copy()
                        )

                        chart_inventory[
                            "Store"
                        ] = (
                            chart_inventory[
                                "store_id"
                            ]
                            .astype(str)
                        )

                        fig = px.bar(
                            chart_inventory,
                            x="Store",
                            y="stock_on_hand",
                            title=(
                                "Stock Position by Store"
                            ),
                        )

                        fig.update_traces(
                            marker_color=COLORS[
                                "blue"
                            ]
                        )

                        apply_chart_theme(
                            fig,
                            height=450,
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True,
                        )

                        st.subheader(
                            "Store-level inventory"
                        )

                        st.dataframe(
                            product_inventory,
                            use_container_width=True,
                            hide_index=True,
                        )


# ============================================================
# EXECUTIVE SUMMARY
# ============================================================

elif page == "Executive Summary":

    st.title(
        "📋 Executive Summary"
    )

    st.caption(
        "High-level view of demand performance, "
        "forecasting, inventory risk, and business priorities."
    )

    st.divider()

    # ========================================================
    # BUSINESS SNAPSHOT
    # ========================================================

    st.subheader(
        "Business snapshot"
    )

    total_demand = (
        daily_demand["demand"].sum()
        if not daily_demand.empty
        else 0
    )

    total_skus = (
        sku_master[
            "sku_id"
        ].nunique()
        if not sku_master.empty
        else 0
    )

    total_stores = (
        store_summary[
            "store_id"
        ].nunique()
        if not store_summary.empty
        else 0
    )

    risk_items = len(risk)

    critical_items = (
        risk[
            "risk_level"
        ]
        .eq("Critical")
        .sum()
        if (
            not risk.empty
            and "risk_level"
            in risk.columns
        )
        else 0
    )

    e1, e2, e3, e4, e5 = (
        st.columns(5)
    )

    e1.metric(
        "Total Demand",
        format_number(
            total_demand
        ),
    )

    e2.metric(
        "Stores",
        format_number(
            total_stores
        ),
    )

    e3.metric(
        "SKUs",
        format_number(
            total_skus
        ),
    )

    e4.metric(
        "Risk Items",
        format_number(
            risk_items
        ),
    )

    e5.metric(
        "Critical",
        format_number(
            critical_items
        ),
    )

    st.divider()

    # ========================================================
    # KEY INSIGHTS
    # ========================================================

    st.subheader(
        "🔍 Key analytical insights"
    )

    left, right = (
        st.columns(2)
    )

    with left:

        if not category_summary.empty:

            top_category = (
                category_summary.iloc[0]
            )

            st.info(
                f"**Leading category:** "
                f"{top_category['category']} "
                f"with approximately "
                f"{format_number(top_category['demand'])} "
                f"units of recorded demand."
            )

        if not sku_summary.empty:

            top_sku_id = (
                sku_summary.iloc[0][
                    "sku_id"
                ]
            )

            top_product = sku_master[
                sku_master[
                    "sku_id"
                ]
                .astype(str)
                == str(
                    top_sku_id
                )
            ]

            if not top_product.empty:

                product_name = str(
                    top_product.iloc[0].get(
                        "sku_name",
                        top_sku_id,
                    )
                )

                st.info(
                    f"**Highest-demand SKU:** "
                    f"{top_sku_id} — "
                    f"{product_name}."
                )

    with right:

        if not weekday_summary.empty:

            highest_day = (
                weekday_summary.loc[
                    weekday_summary[
                        "demand"
                    ].idxmax()
                ]
            )

            lowest_day = (
                weekday_summary.loc[
                    weekday_summary[
                        "demand"
                    ].idxmin()
                ]
            )

            st.info(
                f"**Weekly pattern:** "
                f"{highest_day['weekday']} has the highest "
                f"aggregate demand, while "
                f"{lowest_day['weekday']} has the lowest."
            )

        if not risk.empty:

            critical_pct = (
                critical_items
                /
                len(risk)
                *
                100
                if len(risk) > 0
                else 0
            )

            if critical_pct > 0:

                st.warning(
                    f"**Inventory attention:** "
                    f"{critical_pct:.1f}% of scored inventory "
                    f"positions are currently classified as Critical."
                )

            else:

                st.success(
                    "No inventory positions are currently "
                    "classified as Critical."
                )

    st.divider()

    # ========================================================
    # MODEL PERFORMANCE
    # ========================================================

    st.subheader(
        "🤖 Forecast model performance"
    )

    if not model_results.empty:

        performance = (
            model_results.copy()
        )

        if "MAE" in performance.columns:

            performance = (
                performance
                .sort_values(
                    "MAE",
                    ascending=True,
                )
                .reset_index(
                    drop=True
                )
            )

            if not performance.empty:

                if "model" in performance.columns:

                    best_model = (
                        performance.iloc[0].get(
                            "model",
                            "Unknown",
                        )
                    )

                elif "Model" in performance.columns:

                    best_model = (
                        performance.iloc[0].get(
                            "Model",
                            "Unknown",
                        )
                    )

                else:

                    best_model = "Unknown"

                best_mae = (
                    performance.iloc[0].get(
                        "MAE",
                        np.nan,
                    )
                )

                st.success(
                    f"**Current best model by MAE:** "
                    f"{best_model} — "
                    f"MAE {best_mae:,.2f}"
                )

        metric_cols = [
            col
            for col in [
                "Rank",
                "model",
                "Model",
                "MAE",
                "RMSE",
                "MAPE_percent",
            ]
            if col in performance.columns
        ]

        if metric_cols:

            st.dataframe(
                performance[
                    metric_cols
                ],
                use_container_width=True,
                hide_index=True,
            )

        model_column = None

        if "model" in performance.columns:
            model_column = "model"

        elif "Model" in performance.columns:
            model_column = "Model"

        if (
            model_column is not None
            and "MAE"
            in performance.columns
        ):

            fig = px.bar(
                performance,
                x=model_column,
                y="MAE",
                color="MAE",
                color_continuous_scale=[
                    COLORS["green"],
                    COLORS["yellow"],
                    COLORS["red"],
                ],
                title=(
                    "Model Performance by MAE"
                ),
            )

            fig.update_layout(
                coloraxis_showscale=False
            )

            apply_chart_theme(
                fig,
                height=470,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

    else:

        st.warning(
            "Model comparison output is not available."
        )

    st.divider()

    # ========================================================
    # RISK SUMMARY
    # ========================================================

    st.subheader(
        "⚠️ Inventory risk summary"
    )

    if (
        not risk.empty
        and "risk_level"
        in risk.columns
    ):

        levels = [
            "Critical",
            "High",
            "Medium",
            "Low",
        ]

        risk_summary = (
            risk[
                "risk_level"
            ]
            .value_counts()
            .reindex(
                levels,
                fill_value=0,
            )
            .rename_axis(
                "Risk Level"
            )
            .reset_index(
                name="Items"
            )
        )

        fig = px.bar(
            risk_summary,
            x="Risk Level",
            y="Items",
            color="Risk Level",
            color_discrete_map=RISK_COLORS,
            category_orders={
                "Risk Level": levels
            },
            title=(
                "Current Inventory Risk Profile"
            ),
        )

        fig.update_layout(
            showlegend=False
        )

        apply_chart_theme(
            fig,
            height=450,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    # ========================================================
    # RECOMMENDATIONS
    # ========================================================

    st.divider()

    st.subheader(
        "🎯 Business recommendations"
    )

    recommendations = [
        (
            "🔴 Reorder now",
            "Prioritize Critical inventory positions "
            "and items with zero or insufficient stock.",
        ),
        (
            "🟠 Replenishment priority",
            "Review High-risk SKUs and compare inventory "
            "against reorder and safety-stock thresholds.",
        ),
        (
            "🟡 Monitor",
            "Track Medium-risk products closely and "
            "review demand changes before pressure increases.",
        ),
        (
            "🟢 Maintain",
            "Continue monitoring healthy inventory "
            "using forecasting and risk indicators.",
        ),
    ]

    for title, description in recommendations:

        st.markdown(
            f"**{title}**  \n"
            f"{description}"
        )

    st.divider()

    # ========================================================
    # EXECUTIVE TAKEAWAY
    # ========================================================

    st.subheader(
        "Executive takeaway"
    )

    st.write(
        "Project FORESIGHT combines historical demand analysis, "
        "forecasting, inventory monitoring, and risk scoring into "
        "a single decision-support workflow. The recommended "
        "operating approach is to use forecasts to anticipate "
        "demand, compare expected demand with inventory availability, "
        "and prioritize replenishment for the highest-risk positions."
    )

    st.caption(
        "Model performance should be interpreted using the documented "
        "validation period and methodology. Poor performance against "
        "a baseline should be reported transparently rather than hidden."
    )