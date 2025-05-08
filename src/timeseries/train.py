#!/usr/bin/env python3
# File: src/timeseries/train.py

"""
Train an hourly forecasting model (ARIMA, Prophet, or Keras LSTM) on GA4 data.

All models forecast the next 24 hours (ds, yhat).
The LSTM uses the past 7 days (168 hours) of observations to predict those 24 hours.

Usage:
  # ARIMA (default)
  python train.py --model arima

  # Prophet with custom params
  python train.py --model prophet --param-file configs/prophet_params.json

  # Keras LSTM (uses past 168h window by default)
  python train.py --model lstm --param-file configs/lstm_params.json
"""

from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

# ———————————————————————————————————————————————————————————————————————————————
# Logging
# ———————————————————————————————————————————————————————————————————————————————
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# ———————————————————————————————————————————————————————————————————————————————
# Default hyperparameters
# ———————————————————————————————————————————————————————————————————————————————
DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {
    "arima": {
        "order": [5, 1, 3]
    },
    "prophet": {
        "changepoint_prior_scale": 0.0011101021208737736,
        "seasonality_mode": "additive"
    },
    "lstm": {
        # Use past week (168 hours) as look-back window
        "units": 168,
        "dropout": 0.2,
        "epochs": 20,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "random_state": 42
    }
}


def load_params(model_name: str, param_file: Path | None) -> Dict[str, Any]:
    """
    Load hyperparameters from JSON or fallback to DEFAULT_PARAMS[model_name].
    """
    if param_file and param_file.is_file():
        params = json.loads(param_file.read_text(encoding="utf-8"))
        logger.info(f"Loaded params from {param_file}")
    else:
        params = DEFAULT_PARAMS[model_name].copy()
        logger.info(f"Using default parameters for {model_name}")
    return params


def load_series(csv_path: Path) -> pd.DataFrame:
    """
    Load hourly time series from CSV with columns:
      - ds: datetime (hourly)
      - y: numeric target (e.g., pageviews)
    """
    df = pd.read_csv(csv_path, parse_dates=["ds"])
    df.sort_values("ds", inplace=True)
    df.reset_index(drop=True, inplace=True)
    logger.info(f"Loaded {len(df)} rows from {csv_path}")
    return df


def train_arima(
        df: pd.DataFrame, params: Dict[str, Any]
) -> Tuple[ARIMA, pd.DataFrame]:
    """
    Train ARIMA on all but last 24h, forecast those final 24h.
    Returns the fitted ARIMA model and a DataFrame with columns [ds,y,yhat].
    """
    train, test = df.iloc[:-24], df.iloc[-24:]
    model = ARIMA(train["y"], order=tuple(params["order"])).fit()
    forecast = model.predict(start=len(train), end=len(train) + 23)
    result = test.copy()
    result["yhat"] = forecast.values
    return model, result


def train_prophet(
        df: pd.DataFrame, params: Dict[str, Any]
) -> Tuple[Prophet, pd.DataFrame]:
    """
    Train Prophet on all but last 24h, forecast next 24h.
    Returns the fitted Prophet model and a DataFrame with [ds,y,yhat].
    """
    train, test = df.iloc[:-24], df.iloc[-24:]
    m = Prophet(**params)
    m.fit(train.rename(columns={"y": "y"}))
    future = m.make_future_dataframe(periods=24, freq="H")
    forecast = m.predict(future)
    preds = (
        forecast.set_index("ds")["yhat"]
        .reindex(test["ds"])
        .reset_index()
        .rename(columns={"yhat": "yhat"})
    )
    result = test.copy()
    result["yhat"] = preds["yhat"].values
    return m, result


def train_lstm(
        df: pd.DataFrame, params: Dict[str, Any]
) -> Tuple[Any, pd.DataFrame]:
    """
    Train a Keras LSTM using the past 168h (params['units']) to forecast the next 24h.

    Imports TensorFlow/Keras on demand to avoid top-level dependency.
    Returns the trained model and a DataFrame with [ds,y,yhat].
    """
    try:
        import tensorflow as tf
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
    except ImportError as e:
        raise RuntimeError("TensorFlow/Keras required for LSTM") from e

    series = df["y"].values
    window = params["units"]  # should be 168 for one week
    train_vals = series[:-24]

    # Build sliding-window dataset of shape (n_samples, window, 1)
    X, y = [], []
    for i in range(len(train_vals) - window):
        X.append(train_vals[i: i + window])
        y.append(train_vals[i + window])
    X_arr = np.array(X).reshape(-1, window, 1)
    y_arr = np.array(y)

    # Build model
    tf.random.set_seed(params["random_state"])
    model = Sequential([
        LSTM(params["units"], input_shape=(window, 1)),
        Dropout(params["dropout"]),
        Dense(1)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(params["learning_rate"]),
        loss="mse"
    )

    # Train with early stopping
    es = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
    model.fit(
        X_arr,
        y_arr,
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        callbacks=[es],
        verbose=1
    )

    # Forecast next 24 hours
    preds: list[float] = []
    window_vals = train_vals[-window:].copy()
    for _ in range(24):
        x_input = window_vals.reshape(1, window, 1)
        yhat = model.predict(x_input, verbose=0)[0, 0]
        preds.append(float(yhat))
        window_vals = np.append(window_vals[1:], yhat)

    result = df.iloc[-24:].copy()
    result["yhat"] = preds
    return model, result


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Root Mean Squared Error.
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def evaluate(test_df: pd.DataFrame) -> None:
    """
    Log RMSE and MAPE on the test DataFrame (must contain 'y' and 'yhat').
    """
    y_true = test_df["y"].values
    yhat = test_df["yhat"].values
    error = rmse(y_true, yhat)
    mape = np.mean(np.abs((y_true - yhat) / np.clip(y_true, 1e-8, None))) * 100
    logger.info(f"Test   RMSE = {error:.4f}, MAPE = {mape:.2f}%")


def main() -> None:
    """
    Parse CLI args, train the selected model, evaluate, and save both model and params.
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    default_csv = project_root / "etl" / "data" / "processed" / "hourly_pageviews.csv"
    output_dir = project_root / "models" / "timeseries"
    output_dir.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(description="Train hourly forecast model")
    parser.add_argument(
        "--data", type=Path, default=default_csv,
        help="CSV with columns [ds,y]"
    )
    parser.add_argument(
        "--model",
        choices=["arima", "prophet", "lstm"],
        default="prophet",
        help="Which model to train"
    )
    parser.add_argument(
        "--param-file",
        type=Path,
        help="Optional JSON file with hyperparameters"
    )
    args = parser.parse_args()

    params = load_params(args.model, args.param_file)
    df = load_series(args.data)

    if args.model == "arima":
        model, test_df = train_arima(df, params)
        joblib.dump(model, output_dir / "arima_model.pkl")

    elif args.model == "prophet":
        model, test_df = train_prophet(df, params)
        joblib.dump(model, output_dir / "prophet_model.pkl")

    else:  # lstm
        model, test_df = train_lstm(df, params)
        # Save complete Keras model (architecture + weights)
        model.save(output_dir / "lstm_model.h5")

    # Save params for reproducibility
    (output_dir / f"{args.model}_params.json").write_text(
        json.dumps(params, indent=2), encoding="utf-8"
    )

    evaluate(test_df)


if __name__ == "__main__":
    main()
