import streamlit as st

st.set_page_config(page_title="FORESIGHT Analytics", page_icon="📊", layout="wide")

from utils import inject_custom_css
inject_custom_css()



pages = {
    "OVERVIEW": [
        st.Page("pages/01_overview.py", title="Executive Overview", icon="🏠")
    ],
    "SALES INTELLIGENCE": [
        st.Page("pages/02_sales_analytics.py", title="Sales Analytics", icon="📈"),
        st.Page("pages/03_product_performance.py", title="Product Performance", icon="🛒"),
        st.Page("pages/04_store_performance.py", title="Store Performance", icon="🏬"),
    ],
    "FORECASTING": [
        st.Page("pages/05_demand_forecast.py", title="Demand Forecast", icon="🔮"),
    ],
    "INVENTORY INTELLIGENCE": [
        st.Page("pages/06_inventory_dashboard.py", title="Inventory Dashboard", icon="📦"),
        st.Page("pages/07_inventory_risk.py", title="Inventory Risk", icon="⚠️"),
    ],
    "REPORTS": [
        st.Page("pages/09_executive_summary.py", title="Executive Summary", icon="📋"),
    ]
}

pg = st.navigation(pages)
pg.run()