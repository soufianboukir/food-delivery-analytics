import streamlit as st
import requests

API_URL = "http://localhost:5000"

st.title("Delivery ML System")

# ---------------------------
# REGRESSION UI
# ---------------------------
st.header("Delivery Time Prediction")

distance = st.number_input("Distance")
prep = st.number_input("Preparation time")
traffic = st.number_input("Traffic score")
weather = st.number_input("Weather score")
eff = st.number_input("Efficiency")
exp = st.number_input("Experience")

if st.button("Predict Delivery Time"):

    payload = {
        "delivery_distance_km": distance,
        "preparation_time_minutes": prep,
        "traffic_level_score": traffic,
        "weather_severity_score": weather,
        "delivery_efficiency_score": eff,
        "delivery_partner_experience_years": exp,
    }

    res = requests.post(API_URL + "/predict/delivery_time", json=payload)

    st.success(res.json()["delivery_time_minutes"])


# ---------------------------
# CLASSIFICATION UI
# ---------------------------
st.header("Cancellation Prediction")

traffic = st.number_input("Traffic (cancel)")
weather = st.number_input("Weather (cancel)")
est = st.number_input("Estimated delivery time")
eff = st.number_input("Efficiency (cancel)")
peak = st.number_input("Peak hour (cancel)")
fee = st.number_input("Delivery fee")
discount = st.number_input("Discount")
loyalty = st.number_input("Loyalty score")
rating = st.number_input("Restaurant rating")

if st.button("Predict Cancellation"):

    payload = {
        "traffic_level_score": traffic,
        "weather_severity_score": weather,
        "estimated_delivery_time": est,
        "delivery_efficiency_score": eff,
        "is_peak_hour": peak,
        "delivery_fee": fee,
        "discount_amount": discount,
        "customer_loyalty_score": loyalty,
        "restaurant_rating": rating
    }

    res = requests.post(API_URL + "/predict/cancellation", json=payload)

    st.json(res.json())