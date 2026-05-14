import numpy as np
import joblib
import os
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
package = joblib.load(os.path.join(BASE_DIR, "..", "model", "regression_model_package.pkl"))

model = package["model"]
scaler = package["scaler"]
features = package["features"]
X_test = package["X_test"]
y_test = package["y_test"]



# test that the model predicts a value of the correct shape
def test_prediction_shape():
    X_dummy = np.random.rand(1, 6)
    X_scaled = scaler.transform(X_dummy)
    pred = model.predict(X_scaled)
    assert pred.shape == (1,)



# test that the model predicts a non-negative value
def test_prediction_positive():
    X_dummy = np.random.rand(1, 6)
    X_scaled = scaler.transform(X_dummy)
    pred = model.predict(X_scaled)
    assert pred[0] >= 0



# test that the model does not use features that would cause data leakage
def test_no_data_leakage():
    assert "delivery_time_minutes" not in features
    assert "delayed_delivery_flag" not in features
    assert "delivery_delay_gap" not in features



# model metrics from previous runs (IMPORTANT: BEST ones, not just the last run)
BASELINE_R2 = 0.9983
BASELINE_MAE = 0.42
BASELINE_RMSE = 1.42
TOLERANCE = 0.002

def test_model_does_not_degrade():

    preds = model.predict(X_test)

    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    assert r2 >= BASELINE_R2 - TOLERANCE, \
        f"Model degraded! R2={r2:.4f}, baseline={BASELINE_R2}"

    assert mae <= BASELINE_MAE + TOLERANCE, \
        f"Model degraded! MAE={mae:.4f}, baseline={BASELINE_MAE}"

    assert rmse <= BASELINE_RMSE + TOLERANCE, \
        f"Model degraded! RMSE={rmse:.4f}, baseline={BASELINE_RMSE}"
    