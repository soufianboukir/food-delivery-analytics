import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import numpy as np

st.set_page_config(page_title="Revenue & Discounts", layout="wide")

st.markdown("## Revenue & Discounts Analysis")

st.markdown(
    """
<style>
[data-testid="stMetricValue"] { font-size: 1.9rem; font-weight: 700; }
[data-testid="stMetricLabel"] { font-size: 0.78rem; text-transform: uppercase; letter-spacing:.05em; }
.section-title {
    font-size: 1rem; font-weight: 600;
    border-left: 3px solid #F97316; padding-left: 10px; margin-bottom: 8px; }
</style>""",
    unsafe_allow_html=True,
)


def title(text):
    st.markdown(f'<p class="section-title">{text}</p>', unsafe_allow_html=True)


# LOAD DATA
@st.cache_data
def load_data():

    path = Path(__file__).parents[2] / "data/processed/food_delivery_final.csv"

    df = pd.read_csv(path)

    df["premium_customer_flag"] = df["premium_customer_flag"].fillna(0).astype(int)

    df["premium_label"] = np.where(
        df["premium_customer_flag"] == 1, "Premium", "Standard"
    )

    df["loyalty_tier"] = pd.qcut(
        df["customer_loyalty_score"],
        q=4,
        labels=["Bronze", "Silver", "Gold", "Platinum"],
    )

    df["age_bin"] = pd.cut(
        df["customer_age"],
        bins=[18, 25, 35, 45, 55, 100],
        labels=["18-25", "26-35", "36-45", "46-55", "55+"],
    )

    return df


df = load_data()


# KPI ROW
c1, c2, c3, c4 = st.columns(4)

c1.metric("Avg Order Value", f"{df['order_value'].mean():.2f}")
c2.metric("Avg Discount", f"{df['discount_amount'].mean():.2f}")
c3.metric("Avg Final Paid", f"{df['final_amount_paid'].mean():.2f}")
c4.metric("Promo Usage %", f"{df['promo_code_used'].mean()*100:.1f}%")

st.divider()

# ROW 1 — WATERFALL
title("Revenue Flow Breakdown")

total_revenue = df["order_value"].sum()
total_discount = -df["discount_amount"].sum()
total_fees = df["delivery_fee"].sum()
total_tips = df["tip_amount"].sum()
final_revenue = df["final_amount_paid"].sum()

fig = go.Figure(
    go.Waterfall(
        name="Revenue Flow",
        orientation="v",
        measure=["absolute", "relative", "relative", "relative", "total"],
        x=["Gross Revenue", "Discounts", "Delivery Fees", "Tips", "Final Revenue"],
        y=[total_revenue, total_discount, total_fees, total_tips, final_revenue],
        connector={"line": {"color": "gray"}},
    )
)

fig.update_layout(height=450)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ROW 2 — DISCOUNT IMPACT
c1, c2 = st.columns(2)

with c1:
    title("Discount vs Order Value")

    fig = px.scatter(
        df, x="discount_amount", y="order_value", opacity=0.4, trendline="ols"
    )

    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    title("Discount Distribution")

    fig = px.histogram(df, x="discount_amount", nbins=30)

    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ROW 3 — PROMO ROI ANALYSIS
title("Promo ROI Analysis")

promo = df[df["promo_code_used"] == 1]
no_promo = df[df["promo_code_used"] == 0]

roi_data = pd.DataFrame(
    {
        "Group": ["Promo Users", "No Promo"],
        "Avg Revenue": [
            promo["final_amount_paid"].mean(),
            no_promo["final_amount_paid"].mean(),
        ],
        "Avg Discount": [promo["discount_amount"].mean(), 0],
    }
)

fig = go.Figure()

fig.add_trace(
    go.Bar(x=roi_data["Group"], y=roi_data["Avg Revenue"], name="Avg Revenue")
)

fig.add_trace(
    go.Bar(x=roi_data["Group"], y=roi_data["Avg Discount"], name="Avg Discount")
)

fig.update_layout(barmode="group", height=400)

st.plotly_chart(fig, use_container_width=True)
