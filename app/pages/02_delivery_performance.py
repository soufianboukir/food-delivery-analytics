import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Delivery Performance", layout="wide", page_icon="📈")

try:
    is_dark = st.context.theme.base == "dark"
except AttributeError:
    is_dark = st.get_option("theme.base") == "dark"

PLOT_THEME = "plotly_dark" if is_dark else "plotly_white"
GRID = "#2A3A4A" if is_dark else "#F1F5F9"
FONT_COLOR = "#E5E7EB" if is_dark else "#111827"
HEATMAP_ZERO = "#1E293B" if is_dark else "#F8FAFC"
BG = "rgba(0,0,0,0)"
PALETTE = dict(
    primary="#F97316",
    navy="#0F2744",
    success="#16A34A",
    danger="#DC2626",
    purple="#7C3AED",
)

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


@st.cache_data
def load_data() -> pd.DataFrame:
    path = Path(__file__).parents[2] / "data/processed/food_delivery_final.csv"
    df = pd.read_csv(path)
    df["order_date"] = pd.date_range("2024-01-01", periods=len(df), freq="2h").date
    df["delay_gap"] = df["delivery_time_minutes"] - df["estimated_delivery_time"]
    df["is_early"] = (df["delay_gap"] < 0).astype(int)
    df["tier_label"] = df["city_tier"].map({1: "Tier 1", 2: "Tier 2", 3: "Tier 3"})
    df["traffic_bin"] = pd.cut(
        df["traffic_level_score"],
        bins=5,
        labels=["Very Low", "Low", "Med", "High", "Very High"],
    )
    df["weather_bin"] = pd.cut(
        df["weather_severity_score"],
        bins=5,
        labels=["Clear", "Mild", "Moderate", "Severe", "Extreme"],
    )
    return df


df = load_data()

with st.sidebar:
    city_sel = st.multiselect("City tier", [1, 2, 3], default=[1, 2, 3])
    delayed_only = st.toggle("Delayed orders only")
    max_dist = st.slider(
        "Max distance (km)",
        1,
        int(df["delivery_distance_km"].max()),
        int(df["delivery_distance_km"].max()),
    )

mask = df["city_tier"].isin(city_sel) & (df["delivery_distance_km"] <= max_dist)
if delayed_only:
    mask &= df["delayed_delivery_flag"] == 1
dff = df[mask]


def _theme(fig, height=320, **kw):
    fig.update_layout(
        template=PLOT_THEME,
        height=height,
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color=FONT_COLOR, size=11),
        margin=dict(l=0, r=0, t=28, b=0),
        **kw,
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


def chart(fig, height=320, **kw):
    st.plotly_chart(_theme(fig, height, **kw), use_container_width=True)


def title(text):
    st.markdown(f'<p class="section-title">{text}</p>', unsafe_allow_html=True)


def kpi(col, label, val, delta=None, help=None):
    col.metric(label, val, delta, help=help)


st.markdown("## Delivery Performance")
st.divider()

on_time_rate = (1 - dff["delayed_delivery_flag"].mean()) * 100
avg_delay_gap = dff["delay_gap"].mean()
avg_dist = dff["delivery_distance_km"].mean()
avg_prep = dff["preparation_time_minutes"].mean()
avg_eff = dff["delivery_efficiency_score"].mean()

k1, k2, k3 = st.columns(3)
k1.metric(
    "On-time Rate",
    f"{on_time_rate:.1f}%",
    help="% of orders delivered within estimated time",
)
k2.metric("Avg Distance", f"{avg_dist:.1f} km")
k3.metric("Avg Efficiency Score", f"{avg_eff:.1f} / 100")

st.divider()

# ROW 1 — scatter + distribution
r1c1, r1c2 = st.columns([3, 2])

with r1c1:
    title("Hourly Avg Delivery Time  (actual vs estimated)")
    hourly = (
        dff.groupby("order_hour")[["delivery_time_minutes", "estimated_delivery_time"]]
        .mean()
        .reset_index()
    )
    fig = go.Figure()
    for col, label, color, dash in [
        ("delivery_time_minutes", "Actual", PALETTE["danger"], "solid"),
        ("estimated_delivery_time", "Estimated", PALETTE["success"], "dot"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=hourly["order_hour"],
                y=hourly[col],
                mode="lines+markers",
                name=label,
                line=dict(color=color, width=2, dash=dash),
                marker=dict(size=5),
                hovertemplate=f"{label}: %{{y:.1f}} min<extra></extra>",
            )
        )
    fig.add_vrect(
        x0=11.5,
        x1=14.5,
        fillcolor="rgba(249,115,22,0.08)",
        line_width=0,
        annotation_text="Lunch",
        annotation_position="top left",
        annotation_font_color=FONT_COLOR,
    )
    fig.add_vrect(
        x0=18.5,
        x1=21.5,
        fillcolor="rgba(249,115,22,0.08)",
        line_width=0,
        annotation_text="Dinner",
        annotation_position="top left",
        annotation_font_color=FONT_COLOR,
    )
    chart(
        fig,
        height=300,
        xaxis=dict(tickmode="linear", tick0=0, dtick=2, title="Hour of Day"),
        yaxis_title="Avg Time (min)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
    )


with r1c2:
    title("Delivery Time Distribution")
    fig = go.Figure()
    for flag, label, color in [
        (0, "On-time", PALETTE["success"]),
        (1, "Delayed", PALETTE["danger"]),
    ]:
        sub = dff[dff["delayed_delivery_flag"] == flag]["delivery_time_minutes"]
        fig.add_trace(
            go.Histogram(
                x=sub,
                name=label,
                nbinsx=30,
                opacity=0.72,
                marker_color=color,
                hovertemplate=f"{label}<br>Minutes: %{{x}}<br>Count: %{{y}}<extra></extra>",
            )
        )
    fig.update_layout(barmode="overlay")
    chart(
        fig,
        height=360,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        xaxis_title="Delivery Time (min)",
        yaxis_title="Orders",
    )

st.divider()

# ── ROW 2 — heatmap + regression
r2c1, r2c2 = st.columns(2)

with r2c1:
    title("Delay Rate by Day × Hour")

    # Create pivot table
    pivot = (
        dff.groupby(["order_day_of_week", "order_hour"])["delayed_delivery_flag"]
        .mean()
        .mul(100)
        .unstack(fill_value=0)
    )

    # Ensure ALL days exist (0–6)
    pivot = pivot.reindex(range(7), fill_value=0)

    # Rename days
    pivot.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    # Heatmap
    fig = px.imshow(
        pivot,
        aspect="auto",
        text_auto=".0f",
        color_continuous_scale=[
            [0, HEATMAP_ZERO],
            [0.5, "#F97316"],
            [1, PALETTE["danger"]],
        ],
        labels=dict(
            x="Hour of Day",
            y="Day",
            color="Delay %"
        ),
    )

    fig.update_traces(textfont_size=8)

    chart(
        fig,
        height=320,
        coloraxis_colorbar=dict(
            title="Delay %",
            thickness=12
        )
    )

with r2c2:
    title("Distance vs Delivery Time  (with regression)")
    sample2 = dff.sample(min(2500, len(dff)), random_state=7)

    # OLS regression manually (avoids statsmodels dependency)
    x = sample2["delivery_distance_km"].values
    y = sample2["delivery_time_minutes"].values
    m, b = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 200)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            name="Orders",
            marker=dict(color=PALETTE["primary"], opacity=0.4, size=4),
            hovertemplate="Distance: %{x:.1f} km<br>Actual: %{y:.0f} min<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=m * x_line + b,
            mode="lines",
            name=f"OLS (slope={m:.1f})",
            line=dict(color=PALETTE["navy"] if not is_dark else "#93C5FD", width=2.5),
        )
    )
    chart(
        fig,
        height=320,
        xaxis_title="Distance (km)",
        yaxis_title="Delivery Time (min)",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
    )

st.divider()

# ── ROW 3 — traffic + weather bars ────────────────────────────────────────────
r3c1, r3c2 = st.columns(2)

with r3c1:
    title("Avg Delivery Time by Traffic Level")
    traffic = (
        dff.groupby("traffic_bin", observed=True)["delivery_time_minutes"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"traffic_bin": "Traffic", "mean": "Avg", "std": "Std"})
    )
    fig = go.Figure(
        go.Bar(
            x=traffic["Traffic"],
            y=traffic["Avg"],
            error_y=dict(
                type="data",
                array=traffic["Std"],
                visible=True,
                color=FONT_COLOR,
                thickness=1.2,
            ),
            marker_color=PALETTE["primary"],
            text=traffic["Avg"].round(1),
            textposition="outside",
            hovertemplate="Traffic: %{x}<br>Avg: %{y:.1f} min<extra></extra>",
        )
    )
    chart(fig, height=300, yaxis_title="Avg Delivery Time (min)", showlegend=False)

with r3c2:
    title("Avg Delivery Time by Weather Severity")
    weather = (
        dff.groupby("weather_bin", observed=True)["delivery_time_minutes"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"weather_bin": "Weather", "mean": "Avg", "std": "Std"})
    )
    colors = ["#16A34A", "#65A30D", "#F97316", "#EA580C", "#DC2626"]
    fig = go.Figure(
        go.Bar(
            x=weather["Weather"],
            y=weather["Avg"],
            error_y=dict(
                type="data",
                array=weather["Std"],
                visible=True,
                color=FONT_COLOR,
                thickness=1.2,
            ),
            marker_color=colors[: len(weather)],
            text=weather["Avg"].round(1),
            textposition="outside",
            hovertemplate="Weather: %{x}<br>Avg: %{y:.1f} min<extra></extra>",
        )
    )
    chart(fig, height=300, yaxis_title="Avg Delivery Time (min)", showlegend=False)

st.divider()

# ── ROW 4 — prep breakdown + partner efficiency ────────────────────────────────
r4c1, r4c2, r4c3 = st.columns(3)

with r4c1:
    title("Prep vs Transit Time by City Tier")
    breakdown = (
        dff.groupby("tier_label")[["preparation_time_minutes", "delivery_time_minutes"]]
        .mean()
        .reset_index()
    )
    breakdown["transit"] = (
        breakdown["delivery_time_minutes"] - breakdown["preparation_time_minutes"]
    )
    fig = go.Figure()
    for col, label, color in [
        ("preparation_time_minutes", "Preparation", PALETTE["purple"]),
        ("transit", "Transit", PALETTE["primary"]),
    ]:
        fig.add_trace(
            go.Bar(
                x=breakdown["tier_label"],
                y=breakdown[col],
                name=label,
                marker_color=color,
            )
        )
    chart(
        fig,
        height=300,
        barmode="stack",
        yaxis_title="Minutes",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
    )

with r4c2:
    title("Partner Experience vs Efficiency")
    sample3 = dff.sample(min(2000, len(dff)), random_state=3)
    fig = px.scatter(
        sample3,
        x="delivery_partner_experience_years",
        y="delivery_efficiency_score",
        color="tier_label",
        color_discrete_map={
            "Tier 1": PALETTE["navy"],
            "Tier 2": PALETTE["primary"],
            "Tier 3": PALETTE["success"],
        },
        opacity=0.5,
        size_max=6,
        labels=dict(
            delivery_partner_experience_years="Experience (yrs)",
            delivery_efficiency_score="Efficiency Score",
            tier_label="City",
        ),
    )
    chart(
        fig,
        height=300,
        legend=dict(
            title="", orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1
        ),
    )

with r4c3:
    title("Delay Gap Distribution by Tier")
    fig = go.Figure()
    for tier, color in [
        ("Tier 1", PALETTE["navy"]),
        ("Tier 2", PALETTE["primary"]),
        ("Tier 3", PALETTE["success"]),
    ]:
        sub = dff[dff["tier_label"] == tier]["delay_gap"]
        fig.add_trace(
            go.Box(
                y=sub,
                name=tier,
                marker_color=color,
                boxmean="sd",
                line_width=1.5,
                hovertemplate=f"{tier}<br>Delay gap: %{{y:.1f}} min<extra></extra>",
            )
        )
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color=FONT_COLOR,
        line_width=1,
        annotation_text="On-time line",
        annotation_font_color=FONT_COLOR,
    )
    chart(fig, height=300, yaxis_title="Delay Gap (min)", showlegend=False)
