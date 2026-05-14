import numpy as np
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
package = joblib.load(os.path.join(BASE_DIR, "..", "model", "lr_model_package.pkl"))

lr = package["model"]
scaler = package["scaler"]
features = package["features"]



# test that the model predicts a value of the correct shape
def test_prediction_shape():
    X_dummy = np.random.rand(1, 6)
    X_scaled = scaler.transform(X_dummy)
    pred = lr.predict(X_scaled)
    assert pred.shape == (1,)



# test that the model predicts a non-negative value
def test_prediction_positive():
    X_dummy = np.random.rand(1, 6)
    X_scaled = scaler.transform(X_dummy)
    pred = lr.predict(X_scaled)
    assert pred[0] >= 0
