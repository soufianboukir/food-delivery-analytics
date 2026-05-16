import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Overview", layout="wide", page_icon="📊")

is_dark = st.get_option("theme.base") == "dark"

plot_bg = "rgba(0,0,0,0)" if is_dark else "white"
grid_color = "#2A2A2A" if is_dark else "#F1F5F9"
text_color = "#E5E7EB" if is_dark else "#111827"

st.markdown(
    """
<style>
[data-testid="stMetricValue"] {
    font-size: 2rem;
    font-weight: 700;
}

[data-testid="stMetricLabel"] {
    font-size: 0.8rem;
    color: var(--text-color);
    text-transform: uppercase;
    letter-spacing: .05em;
}

.section-title {
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-color);
    border-left: 3px solid #F97316;
    padding-left: 10px;
    margin-bottom: 8px;
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    path = Path(__file__).parents[2] / "data/processed/food_delivery_final.csv"
    df = pd.read_csv(path)

    rng = pd.date_range("2024-01-01", periods=len(df), freq="2h")
    df["order_date"] = rng.date

    return df


df = load_data()

with st.sidebar:
    city_sel = st.multiselect("City tier", [1, 2, 3], default=[1, 2, 3])
    premium_sel = st.radio("Customer type", ["All", "Premium", "Standard"])

mask = df["city_tier"].isin(city_sel)

if premium_sel == "Premium":
    mask &= df["premium_customer_flag"] == 1
elif premium_sel == "Standard":
    mask &= df["premium_customer_flag"] == 0

dff = df[mask]


def delta(col):
    mid = len(dff) // 2
    a = dff[col].iloc[:mid].mean()
    b = dff[col].iloc[mid:].mean()
    return f"{((b - a) / a * 100):+.1f}%" if a else "0%"


st.markdown("## Operations Overview")
st.caption(f"Orders: {len(dff):,} | Cities: {city_sel} | {premium_sel}")
st.divider()

k1, k2, k3, k4, k5 = st.columns(5)

k1.metric("Total Orders", f"{len(dff):,}")
k2.metric("Avg Delivery Time", f"{dff['delivery_time_minutes'].mean():.1f} min")
k3.metric("Cancellation Rate", f"{dff['cancellation_flag'].mean()*100:.1f}%")
k4.metric("Total Revenue", f"${dff['final_amount_paid'].sum()/1e6:.2f}M")
k5.metric("Avg Order Value", f"${dff['order_value'].mean():.2f}")

st.divider()

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        '<p class="section-title">Orders by City Tier</p>', unsafe_allow_html=True
    )

    tier = dff["city_tier"].value_counts().reset_index()
    tier.columns = ["tier", "orders"]
    tier["label"] = tier["tier"].map({1: "Tier 1", 2: "Tier 2", 3: "Tier 3"})

    fig1 = px.pie(
        tier,
        names="label",
        values="orders",
        hole=0.55,
        color_discrete_sequence=["#0F2744", "#F97316", "#16A34A"],
    )

    fig1.update_layout(
        height=320,
    )

    st.plotly_chart(fig1, use_container_width=True)

# ── 2. ON-TIME VS DELAYED
with c2:
    st.markdown(
        '<p class="section-title">On-time vs Delayed</p>', unsafe_allow_html=True
    )

    delay_df = (
        dff["delayed_delivery_flag"]
        .value_counts()
        .rename({0: "On-time", 1: "Delayed"})
        .reset_index()
    )

    delay_df.columns = ["status", "count"]

    fig2 = px.bar(
        delay_df,
        x="status",
        y="count",
        color="status",
        color_discrete_map={"On-time": "#16A34A", "Delayed": "#DC2626"},
        text_auto=True,
    )

    fig2.update_layout(
        height=320,
        xaxis=dict(gridcolor=grid_color),
        yaxis=dict(gridcolor=grid_color),
        showlegend=False,
    )

    st.plotly_chart(fig2, use_container_width=True)

# ── PEAK HOURS HEATMAP 
with c3:
    st.markdown('<p class="section-title">Peak Hours</p>', unsafe_allow_html=True)
    hour_day = (
        dff.groupby(["order_day_of_week", "order_hour"])
        .size()
        .reset_index(name="orders")
    )

    pivot = hour_day.pivot(
        index="order_day_of_week", columns="order_hour", values="orders"
    ).fillna(0)

    pivot = pivot.reindex(range(7), fill_value=0)

    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    pivot.index = day_labels

    fig3 = px.imshow(
        pivot, color_continuous_scale=[[0, "#F8FAFC"], [1, "#F97316"]], aspect="auto"
    )

    fig3.update_layout(height=320, coloraxis_showscale=False)

    st.plotly_chart(fig3, use_container_width=True)


# ── SECOND ROW
c1, c2, c3 = st.columns(3)

# ── WEATHER × TRAFFIC IMPACT
with c1:
    st.markdown(
        '<p class="section-title">Traffic vs Weather Impact</p>', unsafe_allow_html=True
    )

    heat = (
        dff.groupby(["weather_severity_score", "traffic_level_score"])[
            "delivery_time_minutes"
        ]
        .mean()
        .reset_index()
    )

    fig5 = px.density_heatmap(
        heat,
        x="weather_severity_score",
        y="traffic_level_score",
        z="delivery_time_minutes",
        color_continuous_scale="Oranges",
    )

    fig5.update_layout(
        height=300,
    )

    st.plotly_chart(fig5, use_container_width=True)

# ── CUSTOMER SATISFACTION
with c2:
    st.markdown(
        '<p class="section-title">Customer Satisfaction Drivers</p>',
        unsafe_allow_html=True,
    )

    corr = dff[
        [
            "delivery_time_minutes",
            "customer_rating",
            "delivery_partner_rating",
            "delayed_delivery_flag",
        ]
    ].corr()

    fig6 = px.imshow(corr, text_auto=True, aspect="auto")

    fig6.update_layout(
        height=300,
    )

    st.plotly_chart(fig6, use_container_width=True)

# ── 4. REVENUE BY TIER
with c3:
    st.markdown(
        '<p class="section-title">Revenue by City Tier</p>', unsafe_allow_html=True
    )

    rev = dff.groupby("city_tier")["final_amount_paid"].sum().reset_index()
    rev["city_tier"] = rev["city_tier"].map({1: "Tier 1", 2: "Tier 2", 3: "Tier 3"})

    fig4 = px.bar(
        rev,
        x="city_tier",
        y="final_amount_paid",
        text_auto=".2s",
        color_discrete_sequence=["#298CE3"],
    )

    fig4.update_layout(
        height=320,
        xaxis=dict(gridcolor=grid_color),
        yaxis=dict(gridcolor=grid_color),
        showlegend=False,
    )

    st.plotly_chart(fig4, use_container_width=True)
