import streamlit as st

page1 = st.Page(
    "pages/01_overview.py",
    title="Overview",
    icon="📊"
)

page2 = st.Page(
    "pages/02_delivery_performance.py",
    title="Delivery Performance",
    icon="📈"
)

page5 = st.Page(
    "pages/predictions.py",
    title="Delivery ML System",
    icon="🔎"
)

pg = st.navigation([page1, page2, page5])

pg.run()