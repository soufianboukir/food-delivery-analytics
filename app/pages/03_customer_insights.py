import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import numpy as np

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="Customer Insights",
    page_icon="👥",
    layout="wide"
)

st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.9rem; font-weight: 700; }
[data-testid="stMetricLabel"] { font-size: 0.78rem; text-transform: uppercase; letter-spacing:.05em; }
.section-title {
    font-size: 1rem; font-weight: 600;
    border-left: 3px solid #F97316; padding-left: 10px; margin-bottom: 8px; }
</style>""", unsafe_allow_html=True)

# =========================================================
# THEME
# =========================================================
try:
    is_dark = st.context.theme.base == "dark"
except:
    is_dark = st.get_option("theme.base") == "dark"

PLOT_THEME = "plotly_dark" if is_dark else "plotly_white"

PAL = {
    "primary": "#F97316",
    "navy": "#0F2744",
    "success": "#16A34A",
    "purple": "#7C3AED",
    "teal": "#0D9488"
}

# =========================================================
# HELPERS
# =========================================================
def plot_layout(fig, height=320):
    fig.update_layout(
        template=PLOT_THEME,
        height=height,
        margin=dict(l=0, r=0, t=35, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)

def section(title):
    st.markdown(f"### {title}")

# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data
def load_data():

    path = Path(__file__).parents[2] / "data/processed/food_delivery_final.csv"

    df = pd.read_csv(path)

    df["premium_customer_flag"] = (
        df["premium_customer_flag"]
        .fillna(0)
        .astype(int)
    )

    df["premium_label"] = np.where(
        df["premium_customer_flag"] == 1,
        "Premium",
        "Standard"
    )

    df["loyalty_tier"] = pd.qcut(
        df["customer_loyalty_score"],
        q=4,
        labels=["Bronze", "Silver", "Gold", "Platinum"]
    )

    df["age_bin"] = pd.cut(
        df["customer_age"],
        bins=[18, 25, 35, 45, 55, 100],
        labels=["18-25", "26-35", "36-45", "46-55", "55+"]
    )

    return df

df = load_data()

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:

    st.title("Filters")

    city_sel = st.multiselect(
        "City Tier",
        [1, 2, 3],
        default=[1, 2, 3]
    )

    segment_sel = st.multiselect(
        "Customer Type",
        ["Premium", "Standard"],
        default=["Premium", "Standard"]
    )

    age_range = st.slider(
        "Customer Age",
        int(df["customer_age"].min()),
        int(df["customer_age"].max()),
        (18, int(df["customer_age"].max()))
    )

# =========================================================
# FILTER DATA
# =========================================================
dff = df[
    (df["city_tier"].isin(city_sel)) &
    (df["premium_label"].isin(segment_sel)) &
    (df["customer_age"].between(*age_range))
]

# =========================================================
# HEADER
# =========================================================
st.title("Customer Insights")

st.caption(
    f"{len(dff):,} Orders | "
    f"Age {age_range[0]}-{age_range[1]}"
)

st.divider()

# =========================================================
# KPIs
# =========================================================
c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    "Avg Loyalty",
    f"{dff['customer_loyalty_score'].mean():.1f}"
)

c2.metric(
    "Premium Share",
    f"{dff['premium_customer_flag'].mean()*100:.1f}%"
)

c3.metric(
    "Avg Rating",
    f"{dff['customer_rating'].mean():.2f}"
)

c4.metric(
    "Promo Usage",
    f"{dff['promo_code_used'].mean()*100:.1f}%"
)

c5.metric(
    "Avg Tip",
    f"${dff['tip_amount'].mean():.2f}"
)

st.divider()

def title(text):
    st.markdown(f'<p class="section-title">{text}</p>', unsafe_allow_html=True)

# =========================================================
# ROW 1
# =========================================================
col1, col2 = st.columns([2, 3])

# Loyalty Distribution
with col1:

    title("Loyalty Distribution")

    fig = px.histogram(
        dff,
        x="customer_loyalty_score",
        color="premium_label",
        nbins=30,
        barmode="overlay",
        opacity=0.7,
        color_discrete_map={
            "Premium": PAL["primary"],
            "Standard": PAL["navy"]
        }
    )

    plot_layout(fig, 340)

# Age vs Loyalty
with col2:

    title("Age vs Loyalty")

    sample = dff.sample(min(3000, len(dff)), random_state=42)

    fig = px.scatter(
        sample,
        x="customer_age",
        y="customer_loyalty_score",
        color="premium_label",
        opacity=0.5,
        hover_data=["order_value", "customer_rating"],
        color_discrete_map={
            "Premium": PAL["primary"],
            "Standard": PAL["navy"]
        }
    )

    plot_layout(fig, 340)

st.divider()

# =========================================================
# ROW 2
# =========================================================
col1, col2 = st.columns(2)

# Promo Funnel
with col1:

    title("Promo Funnel")

    stages = [
        "All Orders",
        "Promo Used",
        "Completed Orders",
        "Rating >= 4"
    ]

    values = [
        len(dff),
        dff["promo_code_used"].sum(),
        dff[dff["cancellation_flag"] == 0].shape[0],
        dff[dff["customer_rating"] >= 4].shape[0]
    ]

    fig = go.Figure(go.Funnel(
        y=stages,
        x=values
    ))

    plot_layout(fig, 320)

# Loyalty Tier Revenue
with col2:

    title("Avg Order Value by Loyalty")

    grp = (
        dff.groupby("loyalty_tier", observed=True)["order_value"]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        grp,
        x="loyalty_tier",
        y="order_value",
        color="loyalty_tier"
    )

    plot_layout(fig, 320)

st.divider()

# =========================================================
# ROW 3
# =========================================================
col1, col2 = st.columns(2)

# Orders by Age
with col1:

    title("Orders by Age Group")

    grp = (
        dff.groupby(
            ["age_bin", "premium_label"],
            observed=True
        )
        .size()
        .reset_index(name="orders")
    )

    fig = px.bar(
        grp,
        x="age_bin",
        y="orders",
        color="premium_label",
        barmode="group",
        color_discrete_map={
            "Premium": PAL["primary"],
            "Standard": PAL["navy"]
        }
    )

    plot_layout(fig, 320)

# Tip Analysis
with col2:

    title("Tip Rate by Loyalty Tier")

    tip = (
        dff.groupby("loyalty_tier", observed=True)
        .agg(
            avg_tip=("tip_amount", "mean"),
            tip_rate=("tip_amount", lambda x: (x > 0).mean() * 100)
        )
        .reset_index()
    )

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=tip["loyalty_tier"],
        y=tip["tip_rate"],
        name="Tip Rate %"
    ))

    fig.add_trace(go.Scatter(
        x=tip["loyalty_tier"],
        y=tip["avg_tip"],
        mode="lines+markers",
        name="Avg Tip"
    ))

    plot_layout(fig, 320)