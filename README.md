# FORESIGHT

**Sales Forecasting & Inventory Risk Intelligence Platform**

> An end-to-end analytics system that transforms raw retail data into actionable demand forecasts, inventory health metrics, and risk intelligence — powered by Prophet ML and delivered through an interactive Streamlit dashboard.

---

## At a Glance

| | |
|---|---|
| **Category** | Sales Forecasting · Inventory Analytics · Business Intelligence |
| **Stack** | Python · Pandas · Plotly · Streamlit · Prophet · Scikit-learn |
| **Forecast Model** | Prophet (MAPE: 3.16% · RMSE: 551.58 · MAE: 394.36) |
| **Dashboard** | 8 interactive pages with KPIs, charts, and data tables |

---

## What It Does

```
Raw Data → Cleaning → Feature Engineering → EDA → ML Forecasting → Risk Scoring → Dashboard
```

- **Demand Forecasting** — Predicts future product demand using Prophet with confidence intervals
- **Sales Intelligence** — Tracks daily, monthly, and category-level demand patterns
- **Inventory Monitoring** — Surfaces stock levels, reorder alerts, and days-of-supply metrics
- **Risk Scoring** — Flags Critical/High/Medium/Low stockout and overstock risks
- **Executive Reporting** — Auto-generates data-driven insights and recommendations

---

## Dashboard Pages

| Page | What You'll See |
|---|---|
| Executive Overview | 6 KPIs · demand trends · category breakdown · top stores |
| Sales Analytics | Monthly/weekday trends · cumulative demand · category share |
| Product Performance | Top products by volume & revenue · demand vs price scatter |
| Store Performance | Store rankings · demand distribution · demand vs inventory |
| Demand Forecast | Prophet forecast with confidence bands · model accuracy |
| Inventory Dashboard | Stock levels · reorder alerts · stock vs reorder comparison |
| Inventory Risk | Risk distribution · top risk items · category risk breakdown |
| Executive Summary | Auto-generated insights · actionable recommendations |

---

## Quick Start

```bash
# Clone
git clone https://github.com/ssuptina/Foresight.git
cd Foresight

# Setup
python -m venv venv
.\venv\Scripts\activate      # Windows PowerShell

# Install & Run
pip install -r requirements.txt
streamlit run app/dashboard.py
```

---

## Project Structure

```
Foresight/
├── app/
│   ├── dashboard.py              # Main entry point
│   ├── utils.py                  # Shared utilities, theming, data loaders
│   └── pages/                    # Dashboard pages (01–09)
├── data/                         # Processed datasets & dashboard summaries
├── data raw/                     # Original source datasets
├── outputs/                      # Forecasts, model results, risk scores
├── src/                          # Processing, modeling & analysis scripts
├── requirements.txt
└── README.md
```

---

## Data Sources

| Dataset | Purpose |
|---|---|
| `sales_daily` | Historical daily sales transactions |
| `sku_master` | Product catalog (names, categories, prices) |
| `calendar` | Date features for seasonality modeling |
| `inventory_snapshots` | Point-in-time stock levels and reorder points |

---

## Notes

- Virtual environment (`venv/`) is excluded from version control
- Forecasts are analytical estimates based on historical patterns
- Raw and processed data are kept in separate directories

---

*Built by [ssuptina](https://github.com/ssuptina)*
