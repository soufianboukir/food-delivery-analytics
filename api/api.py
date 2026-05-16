from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)


# load models
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(BASE_DIR, "../models"))

reg_model = joblib.load(os.path.join(MODEL_DIR, "regression_model_package.pkl"))
clf_model = joblib.load(os.path.join(MODEL_DIR, "classification_model_package.pkl"))

regressor = reg_model["model"]
reg_features = reg_model["features"]
reg_scaler = reg_model["scaler"]

classifier = clf_model["model"]
clf_features = clf_model["features"]
clf_scaler = clf_model["scaler"]


# -----------------------------
# REGRESSION ENDPOINT
# -----------------------------
@app.route("/predict/delivery_time", methods=["POST"])
def predict_delivery():

    data = request.json

    X = np.array([[data[f] for f in reg_features]])
    x_scaled = reg_scaler.transform(X)

    pred = regressor.predict(x_scaled)[0]

    return jsonify({"delivery_time_minutes": float(pred)})


# -----------------------------
# CLASSIFICATION ENDPOINT
# -----------------------------
@app.route("/predict/cancellation", methods=["POST"])
def predict_cancellation():

    data = request.json

    X = np.array([[data[f] for f in clf_features]])

    X_scaled = clf_scaler.transform(X)

    pred = classifier.predict(X_scaled)[0]
    prob = classifier.predict_proba(X_scaled)[0][1]

    return jsonify({"cancellation_prediction": int(pred), "probability": float(prob)})


# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
