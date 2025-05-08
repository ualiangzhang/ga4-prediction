"""
Integration tests for FastAPI endpoints `/classify` and `/forecast`.
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_classify_endpoint(tmp_classification_model: Path) -> None:
    """
    POST /classify returns JSON with 'predicted_label' and 'predicted_proba'.
    """
    features = [
        {
            "num_pageviews": 1,
            "num_addtocart": 0,
            "num_sessions": 1,
            "device_type": "mobile",
            "traffic_source": "google",
            "country": "US"
        }
    ]
    payload = {
        "model_type": "lr",
        "features": features,
        "model_path": str(tmp_classification_model)
    }
    resp = client.post("/classify", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data["predicted_label"], list)
    assert isinstance(data["predicted_proba"], list)


def test_forecast_endpoint(tmp_timeseries_models: dict[str, Any]) -> None:
    """
    POST /forecast returns JSON with 'ds' (ISO strings) and 'yhat' lists.
    """
    history = tmp_timeseries_models["history"]
    subset = history.tail(30).copy()
    # convert Timestamps to ISO strings
    subset["ds"] = subset["ds"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    records = subset.to_dict(orient="records")

    payload = {
        "model_type": "arima",
        "history": records,
        "horizon": 5,
        "model_path": str(tmp_timeseries_models["arima"])
    }
    resp = client.post("/forecast", json=payload)
    assert resp.status_code == 200

    data = resp.json()
    assert isinstance(data["ds"], list) and len(data["ds"]) == 5
    assert isinstance(data["yhat"], list) and len(data["yhat"]) == 5
