import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema
import pytest

# load processed dataset
df = pd.read_csv("data/processed/delivery_data.csv")

# -----------------------------
# Schema Validation
# -----------------------------
schema = DataFrameSchema({
    "order_id": Column(
        str,
        nullable=False
    ),
    "city_tier": Column(
        int,
        checks=pa.Check.isin([1, 2, 3])
    ),
    "customer_age": Column(
        int,
        checks=pa.Check.between(18, 100)
    ),
    "customer_loyalty_score": Column(
        float,
        checks=pa.Check.between(0, 100)
    ),
    "order_hour": Column(
        int,
        checks=pa.Check.between(0, 23)
    ),
    "order_day_of_week": Column(
        int,
        checks=pa.Check.between(0, 6)
    ),
    "order_month": Column(
        int,
        checks=pa.Check.between(1, 12)
    ),
    "delivery_distance_km": Column(
        float,
        checks=pa.Check.gt(0)
    ),
    "preparation_time_minutes": Column(
        int,
        checks=pa.Check.gt(0)
    ),
    "delivery_time_minutes": Column(
        float,
        checks=pa.Check.gt(0)
    ),
    "estimated_delivery_time": Column(
        float,
        checks=pa.Check.gt(0)
    ),
    "traffic_level_score": Column(
        int,
        checks=pa.Check.between(0, 100)
    ),
    "weather_severity_score": Column(
        int,
        checks=pa.Check.between(0, 100)
    ),
    "restaurant_rating": Column(
        float,
        checks=pa.Check.between(0, 5),
        nullable=True
    ),
    "delivery_partner_rating": Column(
        float,
        checks=pa.Check.between(0, 5),
        nullable=True
    ),
    "customer_rating": Column(
        float,
        checks=pa.Check.between(0, 5),
        nullable=True
    ),
    "order_value": Column(
        float,
        checks=pa.Check.ge(0)
    ),
    "delivery_fee": Column(
        float,
        checks=pa.Check.ge(0)
    ),
    "discount_amount": Column(
        float,
        checks=pa.Check.ge(0)
    ),
    "tip_amount": Column(
        float,
        checks=pa.Check.ge(0)
    ),
    "final_amount_paid": Column(
        float,
        checks=pa.Check.ge(0)
    ),
    "number_of_items": Column(
        int,
        checks=pa.Check.gt(0)
    ),
    "cancellation_flag": Column(
        int,
        checks=pa.Check.isin([0, 1])
    ),
    "delayed_delivery_flag": Column(
        int,
        checks=pa.Check.isin([0, 1])
    ),
    "refund_flag": Column(
        int,
        checks=pa.Check.isin([0, 1])
    ),
    "promo_code_used": Column(
        int,
        checks=pa.Check.isin([0, 1])
    ),
    "premium_customer_flag": Column(
        int,
        checks=pa.Check.isin([0, 1])
    ),
    "festival_or_weekend_flag": Column(
        int,
        checks=pa.Check.isin([0, 1])
    ),
    "delivery_partner_experience_years": Column(
        int,
        checks=pa.Check.ge(0)
    ),
    "delivery_efficiency_score": Column(
        float,
        nullable=False
    ),
    "customer_rating_missing": Column(
        int,
        checks=pa.Check.isin([0, 1])
    ),
    "delivery_partner_rating_missing": Column(
        int,
        checks=pa.Check.isin([0, 1])
    ),
    "delivery_delay_gap": Column(
        float
    ),
    "is_peak_hour": Column(
        int,
        checks=pa.Check.isin([0, 1])
    ),
    "cost_per_km": Column(
        float,
        checks=pa.Check.ge(0)
    ),
    "revenue_per_item": Column(
        float,
        checks=pa.Check.ge(0)
    ),
    "is_night_order": Column(
        int,
        checks=pa.Check.isin([0, 1])
    )
})


# TEST 1 — Schema Validation
def test_schema_validation():
    schema.validate(df)


# TEST 2 — Row Count Threshold
def test_row_count():
    assert len(df) >= 10000, \
        "Dataset row count unexpectedly low"


# TEST 3 — Null Percentage
def test_null_threshold():
    null_percentages = df.isnull().mean()
    assert (null_percentages < 0.05).all(), \
        f"Columns exceeding null threshold:\n{null_percentages[null_percentages >= 0.05]}"


# TEST 4 — No Duplicate Orders
def test_no_duplicate_order_ids():
    duplicates = df['order_id'].duplicated().sum()
    assert duplicates == 0, \
        f"Found {duplicates} duplicated order IDs"


# TEST 5 — No Negative Monetary Values
def test_positive_amounts():

    monetary_cols = [
        'order_value',
        'delivery_fee',
        'final_amount_paid'
    ]

    for col in monetary_cols:

        assert (df[col] >= 0).all(), \
            f"Negative values found in {col}"


# TEST 6 — Delivery Time Logical
def test_delivery_time_reasonable():

    assert (df['delivery_time_minutes'] <= 300).all(), \
        "Unrealistic delivery times detected"