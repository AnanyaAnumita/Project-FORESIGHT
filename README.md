# FORESIGHT — Sales Forecasting & Inventory Risk Dashboard

## 📌 Project Overview

**FORESIGHT** is an interactive data analytics and forecasting application designed to help businesses understand sales performance, forecast future demand, monitor inventory, and identify potential inventory risks.

The project combines historical sales data, product information, calendar data, and inventory snapshots to provide actionable insights through an interactive **Streamlit dashboard**.

---

## 🎯 Objectives

The main objectives of this project are to:

* Analyze historical sales trends and patterns
* Forecast future product demand
* Evaluate forecasting model performance
* Monitor inventory levels
* Identify potential stockout and overstock risks
* Provide product-level insights
* Present business insights through an interactive dashboard
* Support data-driven inventory and sales decisions

---

## 🚀 Key Features

### 📊 Sales Analytics

* Historical sales analysis
* Daily and product-level sales trends
* Sales performance visualization
* Interactive filtering and exploration

### 🔮 Demand Forecasting

* Future sales/demand prediction
* Forecast visualization
* Historical vs predicted demand comparison
* Forecast evaluation using standard metrics

### 📦 Inventory Dashboard

* Inventory level monitoring
* Product inventory insights
* Stock availability analysis
* Identification of potential inventory issues

### ⚠️ Risk Dashboard

* Stockout risk identification
* Overstock risk identification
* Risk-level analysis
* Prioritization of products requiring attention

### 🔎 Product Details

* Product-level sales information
* Forecast information
* Inventory information
* Product-specific insights

### 📈 Executive Summary

* High-level business KPIs
* Sales performance overview
* Forecasting insights
* Inventory and risk summary

---

## 🤖 Forecasting Model

The project evaluates forecasting performance using a chronological train/test approach.

The dataset is divided into:

* **80% training data**
* **20% testing data**

The forecasting model selected for the final implementation was **Prophet**, based on its evaluation performance.

### Model Evaluation

| Metric | Result |
| ------ | -----: |
| MAE    | 394.36 |
| RMSE   | 551.58 |
| MAPE   |  3.16% |

Lower values indicate better forecasting performance.

---

## 🛠️ Technologies Used

* **Python**
* **Pandas** — Data manipulation and analysis
* **NumPy** — Numerical operations
* **Plotly** — Interactive visualizations
* **Streamlit** — Interactive web dashboard
* **Prophet** — Time-series forecasting
* **Scikit-learn** — Model evaluation and data science utilities

---

## 📁 Project Structure

```text
Zidio_Project/
│
├── app/
│   └── Streamlit dashboard application files
│
├── data/
│   └── Processed and prepared datasets
│
├── data raw/
│   └── Raw source datasets
│
├── outputs/
│   └── Model outputs, forecasts and generated results
│
├── src/
│   └── Data processing, feature engineering and forecasting code
│
├── README.md
│   └── Project documentation
│
├── requirements.txt
│   └── Python dependencies
│
└── .gitignore
    └── Files and folders excluded from Git
```

> The `venv` folder is intentionally excluded from the repository. Each user should create their own virtual environment locally.

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

### 2. Open the project directory

```bash
cd Zidio_Project
```

### 3. Create a virtual environment

Windows:

```bash
python -m venv venv
```

### 4. Activate the virtual environment

Windows PowerShell:

```powershell
.\venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit application

Depending on the dashboard entry-point file, run:

```bash
streamlit run app/<APP_FILE>.py
```

Replace `<APP_FILE>.py` with the actual Streamlit application filename in the `app` folder.

---

## 📊 Data Sources

The project uses sales, product, calendar, and inventory-related datasets for analysis and forecasting.

The main data components include:

* `sales_daily`
* `sku_master`
* `calendar`
* `inventory_snapshots`

Raw data is maintained separately from processed data to keep the data pipeline organized.

---

## 🔄 Project Workflow

```text
Raw Data
   ↓
Data Cleaning
   ↓
Data Preprocessing
   ↓
Feature Engineering
   ↓
Exploratory Data Analysis
   ↓
Forecast Model Development
   ↓
Model Evaluation
   ↓
Demand Forecasting
   ↓
Inventory Risk Analysis
   ↓
Streamlit Dashboard
```

---

## 📌 Business Value

FORESIGHT can help businesses:

* Understand sales behavior
* Anticipate future demand
* Reduce stockout risk
* Identify excess inventory
* Prioritize products requiring attention
* Improve inventory planning
* Make data-driven operational decisions

---

## 👥 Project

**Project:** Zidio Development — Employee Project
**Project Name:** FORESIGHT
**Category:** Sales Forecasting, Inventory Analytics & Business Intelligence

---

## 📄 Notes

* The `venv` directory is not included in version control.
* Dependencies are maintained in `requirements.txt`.
* Raw and processed data are organized into separate directories.
* Forecasting results are based on the available historical dataset and should be interpreted as analytical estimates rather than guaranteed future sales.

---

## ⭐ Future Enhancements

Potential future improvements include:

* Automated data ingestion
* Real-time inventory updates
* Additional forecasting models
* Automated model retraining
* Advanced anomaly detection
* Email or notification-based risk alerts
* Cloud deployment
* Role-based dashboard access
