import numpy as np
import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# load saved classification model package
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

package = joblib.load(
    os.path.join(
        BASE_DIR,
        "..",
        "models",
        "classification_model_package.pkl"
    )
)

model = package["model"]
scaler = package["scaler"]
features = package["features"]
X_test = package["X_test"]
y_test = package["y_test"]

# BASELINE METRICS

BASELINE_ACCURACY = 0.5603
BASELINE_F1 = 0.2493
BASELINE_RECALL = 0.5573
BASELINE_PRECISION = 0.1606

TOLERANCE = 0.02

# test prediction shape
def test_prediction_shape():
    X_dummy = np.random.rand(1, len(features))
    X_scaled = scaler.transform(X_dummy)
    pred = model.predict(X_scaled)
    assert pred.shape == (1,)


# test predictions are only 0 or 1
def test_prediction_classes():
    X_dummy = np.random.rand(10, len(features))
    X_scaled = scaler.transform(X_dummy)
    preds = model.predict(X_scaled)
    assert set(np.unique(preds)).issubset({0, 1})


# test probability outputs are valid
def test_prediction_probabilities():
    X_dummy = np.random.rand(10, len(features))
    X_scaled = scaler.transform(X_dummy)
    probs = model.predict_proba(X_scaled)[:, 1]
    assert probs.min() >= 0
    assert probs.max() <= 1


# test model predicts at least some cancellations
def test_predicts_positive_class():
    X_test_scaled = scaler.transform(X_test)
    preds = model.predict(X_test_scaled)
    assert preds.sum() > 0, (
        "Model predicts no cancellations "
        "(all predictions are class 0)"
    )


# test no data leakage
def test_no_data_leakage():
    forbidden_features = [
        "delivery_time_minutes",
        "delivery_delay_gap",
        "delayed_delivery_flag",
        "refund_flag",
        "cancellation_flag"
    ]
    for col in forbidden_features:
        assert col not in features


# test model performance did not degrade
def test_model_does_not_degrade():
    X_test_scaled = scaler.transform(X_test)
    preds = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, preds)

    precision = precision_score(
        y_test,
        preds,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        preds,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        preds,
        zero_division=0
    )

    assert (
            accuracy >= BASELINE_ACCURACY - TOLERANCE
        ), (
            f"Model degraded! "
            f"Accuracy={accuracy:.4f}, "
            f"baseline={BASELINE_ACCURACY}"
        )

    assert (
        f1 >= BASELINE_F1 - TOLERANCE
    ), (
        f"Model degraded! "
        f"F1={f1:.4f}, "
        f"baseline={BASELINE_F1}"
    )

    assert (
        recall >= BASELINE_RECALL - TOLERANCE
    ), (
        f"Model degraded! "
        f"Recall={recall:.4f}, "
        f"baseline={BASELINE_RECALL}"
    )

    assert (
        precision >= BASELINE_PRECISION - TOLERANCE
    ), (
        f"Model degraded! "
        f"Precision={precision:.4f}, "
        f"baseline={BASELINE_PRECISION}"
    )


# confusion matrix sanity check
def test_confusion_matrix_has_true_positives():

    X_test_scaled = scaler.transform(X_test)

    preds = model.predict(X_test_scaled)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        preds
    ).ravel()

    assert tp > 0, (
        "Model detected zero true positives"
    )