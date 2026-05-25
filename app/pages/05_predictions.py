# THIS CODE IS COMMENTED OUT TO AVOID UNNECESSARY API CALLS DURING PRODUCTION (THE AWS INSTANCE IS TURNED OFF).

# import streamlit as st
# import requests

# API_URL = "http://13.53.39.236:5000" # public IP

# st.set_page_config(page_title="Delivery ML System", layout="wide", page_icon="🔎")

# st.markdown("## Delivery ML System")


# col1, col2 = st.columns(2)

# # REGRESSION
# with col1:

#     st.markdown("### Delivery Time Prediction")

#     distance = st.number_input(
#         "Distance (km)", min_value=0.0, value=5.0, step=0.1, key="reg_distance"
#     )

#     prep = st.number_input("Preparation Time (minutes)", value=15, key="reg_prep")

#     traffic = st.number_input(
#         "Traffic Score", min_value=0.0, max_value=10.0, value=5.0, key="reg_traffic"
#     )

#     weather = st.number_input(
#         "Weather Severity Score",
#         min_value=0.0,
#         max_value=10.0,
#         value=5.0,
#         key="reg_weather",
#     )

#     eff = st.number_input(
#         "Delivery Efficiency Score",
#         min_value=0.0,
#         max_value=100.0,
#         value=80.0,
#         key="reg_eff",
#     )

#     exp = st.number_input(
#         "Delivery Partner Experience (years)", min_value=0.0, value=3.0, key="reg_exp"
#     )

#     if st.button("Predict Delivery Time", key="reg_button"):

#         payload = {
#             "delivery_distance_km": distance,
#             "preparation_time_minutes": prep,
#             "traffic_level_score": traffic,
#             "weather_severity_score": weather,
#             "delivery_efficiency_score": eff,
#             "delivery_partner_experience_years": exp,
#         }

#         res = requests.post(API_URL + "/predict/delivery_time", json=payload)

#         prediction = res.json()["delivery_time_minutes"]

#         st.success(f"Estimated Delivery Time: {prediction:.2f} minutes")

# # CLASSIFICATION
# with col2:

#     st.markdown("### Cancellation Prediction")

#     traffic2 = st.number_input(
#         "Traffic Score", min_value=0.0, max_value=10.0, value=5.0, key="clf_traffic"
#     )

#     weather2 = st.number_input(
#         "Weather Severity Score",
#         min_value=0.0,
#         max_value=10.0,
#         value=5.0,
#         key="clf_weather",
#     )

#     est = st.number_input(
#         "Estimated Delivery Time (minutes)", value=45.0, key="clf_est"
#     )

#     eff2 = st.number_input(
#         "Delivery Efficiency Score",
#         min_value=0.0,
#         max_value=100.0,
#         value=80.0,
#         key="clf_eff",
#     )

#     peak = st.checkbox("Peak Hour", key="clf_peak")

#     fee = st.number_input(
#         "Delivery Fee", min_value=0.0, value=10.0, step=0.5, key="clf_fee"
#     )

#     discount = st.number_input(
#         "Discount Amount", min_value=0.0, value=0.0, step=1.0, key="clf_discount"
#     )

#     loyalty = st.number_input(
#         "Customer Loyalty Score",
#         min_value=0,
#         max_value=100,
#         value=50,
#         key="clf_loyalty",
#     )

#     rating = st.number_input(
#         "Restaurant Rating",
#         min_value=0.0,
#         max_value=5.0,
#         value=4.0,
#         step=0.1,
#         key="clf_rating",
#     )

#     if st.button("Predict Cancellation", key="clf_button"):

#         payload = {
#             "traffic_level_score": traffic2,
#             "weather_severity_score": weather2,
#             "estimated_delivery_time": est,
#             "delivery_efficiency_score": eff2,
#             "is_peak_hour": int(peak),
#             "delivery_fee": fee,
#             "discount_amount": discount,
#             "customer_loyalty_score": loyalty,
#             "restaurant_rating": rating,
#         }

#         res = requests.post(API_URL + "/predict/cancellation", json=payload)

#         result = res.json()

#         prediction = result["cancellation_prediction"]
#         probability = result["probability"]

#         if prediction == 1:
#             st.error(
#                 f"Order Likely To Be Cancelled "
#                 f"({probability * 100:.2f}% probability)"
#             )
#         else:
#             st.success(
#                 f"Order Likely To Be Completed "
#                 f"({(1 - probability) * 100:.2f}% confidence)"
#             )


# NEW ONE WITH LOCAL MODEL LOADING TO AVOID API CALLS (AWS INSTANCE IS TURNED OFF TO SAVE COSTS)

import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Delivery ML System", layout="wide", page_icon="🔎")

st.markdown("## Delivery ML System")

# ============================================
# LOAD MODELS LOCALLY
# ============================================

reg_model = joblib.load("models/regression_model_package.pkl")
clf_model = joblib.load("models/classification_model_package.pkl")

# Regression
regressor = reg_model["model"]
reg_features = reg_model["features"]
reg_scaler = reg_model["scaler"]

# Classification
classifier = clf_model["model"]
clf_features = clf_model["features"]
clf_scaler = clf_model["scaler"]

# ============================================
# UI
# ============================================

col1, col2 = st.columns(2)

# ==================================================
# REGRESSION
# ==================================================
with col1:

    st.markdown("### Delivery Time Prediction")

    distance = st.number_input(
        "Distance (km)", min_value=0.0, value=5.0, step=0.1, key="reg_distance"
    )

    prep = st.number_input("Preparation Time (minutes)", value=15, key="reg_prep")

    traffic = st.number_input(
        "Traffic Score", min_value=0.0, max_value=10.0, value=5.0, key="reg_traffic"
    )

    weather = st.number_input(
        "Weather Severity Score",
        min_value=0.0,
        max_value=10.0,
        value=5.0,
        key="reg_weather",
    )

    eff = st.number_input(
        "Delivery Efficiency Score",
        min_value=0.0,
        max_value=100.0,
        value=80.0,
        key="reg_eff",
    )

    exp = st.number_input(
        "Delivery Partner Experience (years)", min_value=0.0, value=3.0, key="reg_exp"
    )

    if st.button("Predict Delivery Time", key="reg_button"):

        payload = {
            "delivery_distance_km": distance,
            "preparation_time_minutes": prep,
            "traffic_level_score": traffic,
            "weather_severity_score": weather,
            "delivery_efficiency_score": eff,
            "delivery_partner_experience_years": exp,
        }

        X = np.array([[payload[f] for f in reg_features]])

        X_scaled = reg_scaler.transform(X)

        prediction = regressor.predict(X_scaled)[0]

        st.success(f"Estimated Delivery Time: {prediction:.2f} minutes")

# ==================================================
# CLASSIFICATION
# ==================================================
with col2:

    st.markdown("### Cancellation Prediction")

    traffic2 = st.number_input(
        "Traffic Score", min_value=0.0, max_value=10.0, value=5.0, key="clf_traffic"
    )

    weather2 = st.number_input(
        "Weather Severity Score",
        min_value=0.0,
        max_value=10.0,
        value=5.0,
        key="clf_weather",
    )

    est = st.number_input(
        "Estimated Delivery Time (minutes)", value=45.0, key="clf_est"
    )

    eff2 = st.number_input(
        "Delivery Efficiency Score",
        min_value=0.0,
        max_value=100.0,
        value=80.0,
        key="clf_eff",
    )

    peak = st.checkbox("Peak Hour", key="clf_peak")

    fee = st.number_input(
        "Delivery Fee", min_value=0.0, value=10.0, step=0.5, key="clf_fee"
    )

    discount = st.number_input(
        "Discount Amount", min_value=0.0, value=0.0, step=1.0, key="clf_discount"
    )

    loyalty = st.number_input(
        "Customer Loyalty Score",
        min_value=0,
        max_value=100,
        value=50,
        key="clf_loyalty",
    )

    rating = st.number_input(
        "Restaurant Rating",
        min_value=0.0,
        max_value=5.0,
        value=4.0,
        step=0.1,
        key="clf_rating",
    )

    if st.button("Predict Cancellation", key="clf_button"):

        payload = {
            "traffic_level_score": traffic2,
            "weather_severity_score": weather2,
            "estimated_delivery_time": est,
            "delivery_efficiency_score": eff2,
            "is_peak_hour": int(peak),
            "delivery_fee": fee,
            "discount_amount": discount,
            "customer_loyalty_score": loyalty,
            "restaurant_rating": rating,
        }

        X = np.array([[payload[f] for f in clf_features]])

        X_scaled = clf_scaler.transform(X)

        prediction = classifier.predict(X_scaled)[0]

        probability = classifier.predict_proba(X_scaled)[0][1]

        if prediction == 1:

            st.error(
                f"Order Likely To Be Cancelled "
                f"({probability * 100:.2f}% probability)"
            )

        else:

            st.success(
                f"Order Likely To Be Completed "
                f"({(1 - probability) * 100:.2f}% confidence)"
            )
