#!/usr/bin/env python3
# File: src/timeseries/tune.py

"""
Hyperparameter tuning for hourly forecasting models (ARIMA, Prophet, or LSTM) using Optuna.

Minimizes RMSE on the final 24-hour holdout.

Supported models:
  - arima: ARIMA(p, d, q)
  - prophet: Prophet hyperparameters
  - lstm: Keras LSTM sliding-window model

All arguments have sensible defaults; no required flags.

Usage examples:
  # Tune ARIMA (default model) for 30 trials
  python tune.py --trials 30

  # Tune LSTM
  python tune.py --model lstm --trials 50
"""

from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import optuna
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

# ─── Logging Setup ────────────────────────────────────────────────────────────────
logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def load_series(csv_path: Path) -> pd.DataFrame:
    """
    Load hourly time series CSV with columns:
      - ds: datetime (hourly)
      - y: numeric target
    """
    df = pd.read_csv(csv_path, parse_dates=["ds"])
    df.sort_values("ds", inplace=True)
    df.reset_index(drop=True, inplace=True)
    logger.info(f"Loaded {len(df)} rows from {csv_path}")
    return df


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Root Mean Squared Error between true and predicted arrays.
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def objective(
        trial: optuna.trial.Trial,
        df: pd.DataFrame,
        model_name: str
) -> float:
    """
    Optuna objective: sample hyperparameters, train on all but last 24h,
    forecast next 24h, and return RMSE.
    """
    train, test = df.iloc[:-24], df.iloc[-24:]

    if model_name == "arima":
        p = trial.suggest_int("p", 0, 5)
        d = trial.suggest_int("d", 0, 2)
        q = trial.suggest_int("q", 0, 5)
        model = ARIMA(train["y"], order=(p, d, q)).fit()
        pred = model.predict(start=len(train), end=len(train) + 23).values

    elif model_name == "prophet":
        cps = trial.suggest_loguniform("changepoint_prior_scale", 1e-3, 0.5)
        sm = trial.suggest_categorical("seasonality_mode", ["additive", "multiplicative"])
        m = Prophet(changepoint_prior_scale=cps, seasonality_mode=sm)
        m.fit(train.rename(columns={"y": "y"}))
        future = m.make_future_dataframe(periods=24, freq="H")
        fc = m.predict(future)
        pred = (
            fc.set_index("ds")["yhat"]
            .reindex(test["ds"])
            .values
        )

    else:  # lstm
        # defer TensorFlow import
        try:
            import tensorflow as tf  # noqa: F401
            from tensorflow.keras import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
        except ImportError as e:
            raise RuntimeError("TensorFlow required for LSTM tuning") from e

        units = trial.suggest_int("units", 24, 168, step=24)
        dropout = trial.suggest_uniform("dropout", 0.0, 0.5)
        lr = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
        epochs = trial.suggest_int("epochs", 5, 30)
        batch = trial.suggest_int("batch_size", 16, 128, step=16)

        series = df["y"].values
        window = units
        train_vals = series[:-24]

        # build sliding-window dataset
        X_train, y_train = [], []
        for i in range(len(train_vals) - window):
            X_train.append(train_vals[i: i + window])
            y_train.append(train_vals[i + window])
        X_arr = np.array(X_train).reshape(-1, window, 1)
        y_arr = np.array(y_train)

        tf.random.set_seed(42)
        model = Sequential([
            LSTM(units, input_shape=(window, 1)),
            Dropout(dropout),
            Dense(1)
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr),
            loss="mse"
        )
        model.fit(X_arr, y_arr, epochs=epochs, batch_size=batch, verbose=0)

        # forecast next 24h
        preds = []
        window_vals = train_vals[-window:].copy()
        for _ in range(24):
            x_in = window_vals.reshape(1, window, 1)
            yhat = model.predict(x_in, verbose=0)[0, 0]
            preds.append(float(yhat))
            window_vals = np.append(window_vals[1:], yhat)
        pred = np.array(preds)

    return rmse(test["y"].values, pred)


def main() -> None:
    """
    Parse CLI args (all optional), run Optuna study, and save best params to JSON.
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    default_csv = project_root / "etl" / "data" / "processed" / "hourly_pageviews.csv"
    default_out = project_root / "models" / "timeseries" / "best_params.json"

    parser = argparse.ArgumentParser(description="Optuna tuning for TS models")
    parser.add_argument(
        "--model", "-m",
        choices=["arima", "prophet", "lstm"],
        default="lstm",
        help="Model to tune (default: arima)"
    )
    parser.add_argument(
        "--trials", "-t",
        type=int,
        default=50,
        help="Number of Optuna trials (default: 50)"
    )
    parser.add_argument(
        "--data", "-d",
        type=Path,
        default=default_csv,
        help=f"Input CSV (default: {default_csv})"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Path to save best params JSON (default: models/timeseries/<model>_best_params.json)"
    )
    args = parser.parse_args()

    # Determine output file if not provided
    if args.output is None:
        out_dir = project_root / "models" / "timeseries"
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output = out_dir / f"{args.model}_best_params.json"

    # Load data
    df = load_series(args.data)

    # Run Optuna study
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, df, args.model), n_trials=args.trials)

    logger.info(f"Best RMSE = {study.best_value:.4f}")
    logger.info(f"Best params = {study.best_params}")

    # Save best parameters
    args.output.write_text(json.dumps(study.best_params, indent=2), encoding="utf-8")
    logger.info(f"Saved best params to {args.output}")


if __name__ == "__main__":
    main()
