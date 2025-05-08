#!/usr/bin/env python3
# File: api/main.py

"""
GA4 Prediction Service

This FastAPI application exposes two endpoints:
  1. POST /classify  — run inference with a classification model (XGB, RF, LR, or Keras DNN)
  2. POST /forecast  — run hourly forecasting with a timeseries model (ARIMA, Prophet, or Keras LSTM)
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import List, Literal, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Import your existing prediction functions
from src.classification.predict import (
    preprocess_features,
    predict_sklearn,
    predict_keras,
)
from src.timeseries.predict import (
    predict_arima,
    predict_prophet,
    predict_lstm,
)

# ─── Logging Setup ────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="GA4 Prediction API",
    version="1.0",
    description="Serve GA4 classification and hourly forecasting models via REST"
)


# ─── Classification ───────────────────────────────────────────────────────────────
class ClassificationRequest(BaseModel):
    """
    Request model for classification inference.
    """
    model_type: Literal["xgb", "rf", "lr", "dnn"] = Field(
        "xgb", description="Which model to use for classification"
    )
    features: List[dict] = Field(
        ..., description="List of feature records (each a dict of column→value)"
    )
    model_path: str = Field(
        ..., description="Filesystem path to the trained model pickle or .h5 (inside container)"
    )
    params_path: Optional[str] = Field(
        None, description="Path to DNN params JSON (required if model_type=='dnn')"
    )


class ClassificationResponse(BaseModel):
    """
    Response model for classification inference.
    """
    predicted_label: List[int] = Field(
        ..., description="Predicted class labels (0 or 1) for each record"
    )
    predicted_proba: List[float] = Field(
        ..., description="Predicted probability of the positive class"
    )


@app.post(
    "/classify",
    response_model=ClassificationResponse,
    summary="Run GA4 conversion classification"
)
async def classify(req: ClassificationRequest) -> ClassificationResponse:
    """
    Run inference on a batch of feature records using the specified model.
    """
    # Load features into DataFrame
    df = pd.DataFrame(req.features)
    if df.empty:
        raise HTTPException(400, "Feature list is empty")

    # Preprocess into numpy array
    X = preprocess_features(df)

    # Resolve model file
    model_file = Path(req.model_path)
    if not model_file.is_file():
        raise HTTPException(404, f"Model file not found at {model_file}")

    # Dispatch to the correct predict function
    if req.model_type == "dnn":
        # Keras DNN needs params_path for architecture (if any)
        labels, proba = predict_keras(model_file, X)
    else:
        labels, proba = predict_sklearn(model_file, X)

    return ClassificationResponse(
        predicted_label=labels.tolist(),
        predicted_proba=proba.tolist()
    )


# ─── Time-Series Forecasting ──────────────────────────────────────────────────────
class TimeSeriesRequest(BaseModel):
    """
    Request model for hourly time-series forecasting.
    """
    model_type: Literal["arima", "prophet", "lstm"] = Field(
        "arima", description="Which forecasting model to use"
    )
    history: List[dict] = Field(
        ..., description="Historical series, each dict with 'ds' (ISO datetime) and 'y' (float)"
    )
    horizon: int = Field(
        24, description="Number of future hours to forecast"
    )
    model_path: str = Field(
        ..., description="Filesystem path to the trained TS model (.pkl or .h5)"
    )
    params_path: Optional[str] = Field(
        None, description="Path to LSTM params JSON (required if model_type=='lstm')"
    )


class TimeSeriesResponse(BaseModel):
    """
    Response model for hourly forecasting.
    """
    ds: List[str] = Field(
        ..., description="List of forecast timestamps (ISO format)"
    )
    yhat: List[float] = Field(
        ..., description="List of forecast values"
    )


@app.post(
    "/forecast",
    response_model=TimeSeriesResponse,
    summary="Run hourly time-series forecast"
)
async def forecast(req: TimeSeriesRequest) -> TimeSeriesResponse:
    """
    Forecast the next `horizon` hours using the specified time-series model.
    """
    # Load history into DataFrame
    hist_df = pd.DataFrame(req.history)
    if "ds" not in hist_df.columns or "y" not in hist_df.columns:
        raise HTTPException(400, "History must include 'ds' and 'y' fields")
    hist_df["ds"] = pd.to_datetime(hist_df["ds"])

    # Resolve model file
    model_file = Path(req.model_path)
    if not model_file.is_file():
        raise HTTPException(404, f"Model file not found at {model_file}")

    # Dispatch to correct TS predict function
    if req.model_type == "arima":
        result_df = predict_arima(model_file, hist_df, req.horizon)
    elif req.model_type == "prophet":
        result_df = predict_prophet(model_file, hist_df, req.horizon)
    else:
        params_file = Path(req.params_path) if req.params_path else model_file.parent / "lstm_params.json"
        if not params_file.is_file():
            raise HTTPException(400, f"LSTM params JSON not found at {params_file}")
        result_df = predict_lstm(model_file, params_file, hist_df, req.horizon)

    # Format output
    ds_list = result_df["ds"].dt.strftime("%Y-%m-%dT%H:%M:%SZ").tolist()
    yhat_list = result_df["yhat"].tolist()

    return TimeSeriesResponse(ds=ds_list, yhat=yhat_list)
