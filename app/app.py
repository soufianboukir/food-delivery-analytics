import streamlit as st

page1 = st.Page(
    "pages/settings.py",
    title="Overview",
    icon="📊"
)
page5 = st.Page(
    "pages/predictions.py",
    title="Delivery ML System",
    icon="🔎"
)

pg = st.navigation([page1, page5])

pg.run()