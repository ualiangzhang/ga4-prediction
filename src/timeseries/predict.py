#!/usr/bin/env python3
# File: src/timeseries/predict.py

"""
Run hourly forecasts using a trained time-series model.

Supported models and their default locations under models/timeseries/:
  - ARIMA:   arima_model.pkl
  - Prophet: prophet_model.pkl
  - LSTM:    lstm_model.h5  (Keras)

All arguments have sensible defaults; none are required.
"""

from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

# ─── Logging Setup ────────────────────────────────────────────────────────────────
logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def load_series(csv_path: Path) -> pd.DataFrame:
    """
    Load the historical hourly series from CSV.

    Args:
        csv_path: Path to a CSV with columns ['ds', 'y'].

    Returns:
        DataFrame sorted by 'ds'.
    """
    df = pd.read_csv(csv_path, parse_dates=["ds"])
    df.sort_values("ds", inplace=True)
    df.reset_index(drop=True, inplace=True)
    logger.info(f"Loaded {len(df)} rows from {csv_path}")
    return df


def predict_arima(
        model_path: Path,
        history: pd.DataFrame,
        horizon: int
) -> pd.DataFrame:
    """
    Load an ARIMA model and forecast the next `horizon` hours.

    Args:
        model_path: Path to a joblib-dumped ARIMA model (.pkl).
        history: DataFrame with historical ['ds','y'].
        horizon: Number of hours to forecast.

    Returns:
        DataFrame with columns ['ds','yhat'].
    """
    model = joblib.load(model_path)
    start = len(history)
    end = start + horizon - 1
    forecast = model.predict(start=start, end=end).values

    last_ts = history["ds"].iloc[-1]
    future = pd.date_range(
        start=last_ts + pd.Timedelta(hours=1),
        periods=horizon,
        freq="H"
    )
    return pd.DataFrame({"ds": future, "yhat": forecast})


def predict_prophet(
        model_path: Path,
        history: pd.DataFrame,
        horizon: int
) -> pd.DataFrame:
    """
    Load a Prophet model and forecast the next `horizon` hours.

    Args:
        model_path: Path to a joblib-dumped Prophet model (.pkl).
        history: DataFrame with historical ['ds','y'].
        horizon: Number of hours to forecast.

    Returns:
        DataFrame with columns ['ds','yhat'].
    """
    m = joblib.load(model_path)  # type: ignore
    future = m.make_future_dataframe(periods=horizon, freq="H")
    fc = m.predict(future)
    tail = fc[["ds", "yhat"]].tail(horizon).reset_index(drop=True)
    return tail


def predict_lstm(
        model_path: Path,
        params_path: Path,
        history: pd.DataFrame,
        horizon: int
) -> pd.DataFrame:
    """
    Load a Keras LSTM and forecast the next `horizon` hours using the last
    `units` timestamps as input.

    Args:
        model_path: Path to a .h5 Keras model file.
        params_path: Path to the JSON params file containing 'units'.
        history: DataFrame with historical ['ds','y'].
        horizon: Number of hours to forecast.

    Returns:
        DataFrame with columns ['ds','yhat'].
    """
    try:
        from tensorflow.keras.models import load_model
    except ImportError as e:
        raise RuntimeError("TensorFlow/Keras is required for LSTM inference") from e

    # Load params to get window size
    if not params_path.is_file():
        raise FileNotFoundError(f"LSTM params file not found: {params_path}")
    params = json.loads(params_path.read_text(encoding="utf-8"))
    window = int(params.get("units", 168))

    # Prepare the last `window` observations
    series = history["y"].values
    if len(series) < window:
        raise ValueError(f"Not enough history ({len(series)}) for window={window}")
    window_vals = series[-window:].copy().reshape(1, window, 1)

    # Load the model without compiling (avoids legacy loss errors)
    model = load_model(model_path, compile=False)
    logger.info(f"Loaded Keras LSTM from {model_path}")

    # Iteratively forecast
    preds: list[float] = []
    for _ in range(horizon):
        yhat = float(model.predict(window_vals, verbose=0)[0, 0])
        preds.append(yhat)
        window_vals = np.append(window_vals[:, 1:, :], [[[yhat]]], axis=1)

    last_ts = history["ds"].iloc[-1]
    future = pd.date_range(
        start=last_ts + pd.Timedelta(hours=1),
        periods=horizon,
        freq="H"
    )
    return pd.DataFrame({"ds": future, "yhat": preds})


def main() -> None:
    """
    CLI entry-point: parse args, run forecast, and save results.
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    default_csv = project_root / "etl" / "data" / "processed" / "hourly_pageviews.csv"
    ts_model_dir = project_root / "models" / "timeseries"
    default_output = project_root / "models" / "timeseries_forecast.csv"

    parser = argparse.ArgumentParser(description="Forecast next N hours with a TS model")
    parser.add_argument("--data", "-d", type=Path, default=default_csv, help="Input CSV")
    parser.add_argument("--model", "-m", choices=["arima", "prophet", "lstm"], default="arima")
    parser.add_argument("--model-file", type=Path, help="Override model file path")
    parser.add_argument("--params-file", type=Path, help="LSTM params JSON path")
    parser.add_argument("--horizon", "-n", type=int, default=24, help="Hours to forecast")
    parser.add_argument("--output", "-o", type=Path, default=default_output, help="Output CSV")
    args = parser.parse_args()

    # Resolve paths
    model_file = args.model_file or (ts_model_dir / f"{args.model}_model{'.h5' if args.model=='lstm' else '.pkl'}")
    if not model_file.is_file():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    params_file = args.params_file or (model_file.parent / "lstm_params.json")
    history = load_series(args.data)

    # Dispatch
    if args.model == "arima":
        forecast = predict_arima(model_file, history, args.horizon)
    elif args.model == "prophet":
        forecast = predict_prophet(model_file, history, args.horizon)
    else:
        forecast = predict_lstm(model_file, params_file, history, args.horizon)

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(args.output, index=False)
    logger.info(f"Saved forecast ({len(forecast)} rows) to {args.output}")
