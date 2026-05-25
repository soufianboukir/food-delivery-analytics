# Food Delivery Intelligence System with Predictive Analytics

An end-to-end data science and business intelligence system built on 15,000 food delivery transactions — combining interactive dashboards, customer analytics, delivery performance analysis, and multi-model predictive modeling for delivery time and order cancellation.

## 📄 Report
- Full analysis report: [PDF](https://github.com/soufianboukir/food-delivery-analytics/blob/main/reports/food-delivery-intelligence-system.pdf)
- Streamlit dashboard: [Live App](https://your-dashboard-url.streamlit.app/)

---

## Overview

This project develops a **5-page interactive Streamlit dashboard** that transforms raw food delivery data into actionable business intelligence. It covers the full data science pipeline:

- **Exploratory Data Analysis** — distributions, correlations, trends
- **Delivery Performance** — on-time rates, delay patterns, distance analysis
- **Customer Insights** — loyalty scores, premium segmentation, promo behavior
- **Revenue & Discount Analysis** — revenue flow, discount distribution, promo impact
- **Predictive ML System** — delivery time regression + order cancellation classification

The system is designed to answer real business questions:
- Which city tiers and customer segments drive the most revenue?
- What factors influence delivery time and order cancellations?
- Do promo codes effectively increase order completion rates?
- What is the predicted delivery time and cancellation risk for a given order?

---

## Project Structure

```
food-delivery-analytics/
│
├── .github/
│   └── workflows/
│       ├── cd.yml        # CI pipeline (code formatting, linting, run unit tests)
│       └── ci.yml        # CD pipeline (build docker image and push to AWS EC2)
├── app/
│   ├── app.py                          # Main Streamlit entry point
│   └── pages/
│       ├── 01_overview.py                 # Page 1 — KPIs, orders, revenue overview
│       ├── 02_delivery_performance.py     # Page 2 — On-time rates, delays, distance
│       ├── 03_customer_insights.py        # Page 3 — Loyalty, premium, promo funnel
│       ├── 04_revenue_discounts.py        # Page 4 — Revenue flow, discount analysis
│       └── 04_predictions.py               # Page 5 — Delivery time & cancellation prediction
│
├── api/
│   ├── app.py                          # Main flask api app
│   └── requirements.txt                # Requirements needed for flask app
├── data/
│   ├── raw/
│   │   └── food_delivery.csv           # Original dataset
│   └── processed/
|       ├── food_delivery.csv           # Cleaned dataset 
│       └── food_delivery_final.csv     # Cleaned & feature-engineered dataset
│
├── models/
│   ├── regression_model_package.pkl           # Saved delivery time model
│   └── classification_model_package.pkl         # Saved cancellation model
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modeling_regression.ipynb
│   └── 05_modeling_classification.ipynb
├── tests/
│   ├── test_classification_model.py        # unit tests for classification model
│   ├── test_data_quality.py             # test data quality (min rows, null values, etc.)
│   └── test_regression_model.py         # unit tests for classification model
│
├── requirements.txt                    # Python dependencies
├── .dockerignore
├── Dockerfile             
├── .gitignore            
├── README.md
└── reports/
    └── main.pdf                        # Full academic report
```

---

## Dashboard Pages

### Page 1 — Operations Overview
High-level business snapshot with 5 KPI cards, orders by city tier, on-time vs delayed breakdown, traffic & weather impact heatmap, customer satisfaction drivers, and revenue by city tier.

| KPI | Value |
|---|---|
| Total Orders | 15,000 |
| Avg Delivery Time | 94.1 min |
| Cancellation Rate | 13.4% |
| Total Revenue | $1.79M |
| Avg Order Value | $113.95 |

---

### Page 2 — Delivery Performance
- On-time rate, average distance, and average efficiency score KPIs
- Hourly average delivery time — Actual vs Estimated line chart
- Delivery time distribution — on-time vs delayed histogram
- Delay rate heatmap by Day × Hour
- Distance vs Delivery Time scatter with regression line

---

### Page 3 — Customer Insights
- Loyalty score distribution — Standard vs Premium customers
- Age vs Loyalty scatter plot
- Promo funnel — Total orders → Promo used → Completed → Rated ≥ 4
- Average order value by loyalty tier (Bronze, Silver, Gold, Platinum)
- Key KPIs: Avg Loyalty, Premium Share, Avg Rating, Promo Usage, Avg Tip

---

### Page 4 — Revenue & Discount Analysis
- Revenue flow waterfall chart — Gross Revenue → Discounts → Delivery Fees → Tips → Final Revenue
- Discount vs Order Value scatter plot
- Discount distribution histogram
- Key KPIs: Avg Order Value, Avg Discount, Avg Final Paid, Promo Usage %

---

### Page 5 — Delivery ML System
- **Delivery Time Prediction** — Linear Regression model using distance, preparation time, traffic, weather, efficiency score, and driver experience
- **Cancellation Prediction** — Logistic Regression model using traffic, weather, delivery time, delivery fee, discount, loyalty score, and restaurant rating
- Interactive +/− input controls for real-time predictions
- Model performance tables for both regression and classification tasks

---

## Machine Learning Models

### Regression — Delivery Time Prediction

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **Linear Regression** | **0.42** | **1.41** | **0.9983** |
| Random Forest | 4.47 | 5.63 | 0.9721 |
| XGBoost | 3.41 | 4.28 | 0.9839 |

✅ **Selected Model: Linear Regression**

### Classification — Order Cancellation Prediction

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **Logistic Regression** | 0.5603 | 0.1606 | **0.5573** | **0.2493** |
| Random Forest | 0.8693 | 0 | 0 | 0 |
| XGBoost | 0.8103 | 0.1741 | 0.1196 | 0.1418 |

✅ **Selected Model: Logistic Regression** — highest recall and F1-score for the minority class (cancelled orders)

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/soufianboukir/food-delivery-analytics.git
cd food-delivery-analytics
```

### 2. Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Usage

```bash
streamlit run app/app.py
```
```bash
python api/app.py
```

Then open your browser at `http://localhost:8501`

### Sidebar Filters
- **Date Range** — Filter all pages by order date
- **City Tier** — Filter by customer city tier (1, 2, or 3)
- **Premium / Standard** — Filter by customer membership type

---

built with ❤️ by **soufian**.
