#!/usr/bin/env python3
# File: src/timeseries/forecast.py

"""
Compare time series forecasting models (Prophet, ARIMA, LSTM) on daily GA4 page views,
with optional LSTM implementation in Keras or PyTorch.

Assumes a CSV at etl/data/processed/daily_pageviews.csv with columns:
  - ds: date string in 'YYYY-MM-DD' format
  - y : float count of page views

Dependencies:
  pip install pandas numpy prophet statsmodels scikit-learn tensorflow torch

Usage:
  # Use defaults (LSTM via Keras)
  python forecast.py

  # Use PyTorch for LSTM
  python forecast.py --lstm-framework pytorch

  # Override data and output
  python forecast.py \
    --data /path/to/daily_pageviews.csv \
    --output-dir /path/to/output \
    --lstm-framework pytorch
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error



# PyTorch imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ─── Logging Setup ───────────────────────────────────────────────────────────────
logging.basicConfig(fmt="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


# ─── Data Loader ─────────────────────────────────────────────────────────────────
def load_series(csv_path: Path) -> pd.DataFrame:
    """Load daily time series data from CSV."""
    df = pd.read_csv(csv_path, parse_dates=['ds'])[['ds', 'y']].sort_values('ds')
    df['y'] = df['y'].astype(float)
    logger.info(f"Loaded {len(df)} rows from {csv_path}")
    return df


# ─── Prophet Model ───────────────────────────────────────────────────────────────
def train_prophet(train_df: pd.DataFrame) -> pd.Series:
    """Train Prophet and forecast next 14 days."""
    model = Prophet()
    model.fit(train_df)
    future = model.make_future_dataframe(periods=14, freq='D')
    forecast = model.predict(future)
    return forecast['yhat'].iloc[-14:].reset_index(drop=True)


# ─── ARIMA Model ─────────────────────────────────────────────────────────────────
def train_arima(train_df: pd.DataFrame, order: Tuple[int, int, int]=(1,1,1)) -> pd.Series:
    """Train ARIMA and forecast next 14 days."""
    series = train_df['y']
    model = ARIMA(series, order=order).fit()
    forecast = model.forecast(steps=14)
    return pd.Series(forecast).reset_index(drop=True)


# ─── Keras LSTM ───────────────────────────────────────────────────────────────
# def train_lstm_keras(train_df: pd.DataFrame, lookback: int=7) -> pd.Series:
#     """Train LSTM in Keras and forecast next 14 days."""
#     series = train_df['y'].values
#     X, y = [], []
#     for i in range(len(series) - lookback):
#         X.append(series[i:i+lookback])
#         y.append(series[i+lookback])
#     X = np.array(X)[:, :, None]
#     y = np.array(y)
#
#     train_size = len(y) - 14
#     X_train, X_test = X[:train_size], X[train_size:]
#     y_train = y[:train_size]
#
#     model = KSequential([KLSTM(32, input_shape=(lookback,1)), KDense(1)])
#     model.compile(optimizer='adam', loss='mse')
#     es = EarlyStopping(patience=5, restore_best_weights=True)
#     model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.1,
#               callbacks=[es], verbose=0)
#
#     preds = model.predict(X_test).flatten()
#     return pd.Series(preds)


# ─── PyTorch LSTM ───────────────────────────────────────────────────────────────
class TimeSeriesDataset(Dataset):
    def __init__(self, series: np.ndarray, lookback: int):
        X, y = [], []
        for i in range(len(series) - lookback):
            X.append(series[i:i+lookback])
            y.append(series[i+lookback])
        self.X = np.array(X)
        self.y = np.array(y)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.X[idx], dtype=torch.float32).unsqueeze(-1),
            torch.tensor(self.y[idx], dtype=torch.float32)
        )


class LSTMModel(nn.Module):
    def __init__(self, input_size: int=1, hidden_size: int=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out).squeeze()


def train_lstm_pytorch(train_df: pd.DataFrame, lookback: int=7) -> pd.Series:
    """Train LSTM in PyTorch and forecast next 14 days."""
    series = train_df['y'].values
    dataset = TimeSeriesDataset(series, lookback)
    train_size = len(dataset) - 14
    ds_train, ds_test = torch.utils.data.random_split(dataset, [train_size, 14])
    loader = DataLoader(ds_train, batch_size=16, shuffle=True)
    test_loader = DataLoader(ds_test, batch_size=14)

    model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(50):
        for xb, yb in loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    preds_list = []
    with torch.no_grad():
        for xb, _ in test_loader:
            preds_list.append(model(xb).cpu().numpy())
    return pd.Series(np.concatenate(preds_list))


# ─── Evaluation ─────────────────────────────────────────────────────────────────
def evaluate(true: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """Compute MAE and RMSE."""
    mae = mean_absolute_error(true, pred)
    rmse = mean_squared_error(true, pred, squared=False)
    return {'mae': mae, 'rmse': rmse}


# ─── Main Routine ────────────────────────────────────────────────────────────────
def main() -> None:
    project_root = Path(__file__).resolve().parent.parent.parent
    default_data = project_root / "etl" / "data" / "processed" / "daily_pageviews.csv"
    default_out = project_root / "models" / "ts"

    parser = argparse.ArgumentParser(description="TS Forecast Comparison")
    parser.add_argument("--data", type=Path, default=default_data, help="Daily series CSV")
    parser.add_argument("--output-dir", type=Path, default=default_out, help="Output dir")
    parser.add_argument(
        "--lstm-framework",
        type=str,
        choices=["keras", "pytorch"],
        default="keras",
        help="Which framework to use for LSTM"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_series(args.data)
    train_df = df[:-14]
    true_vals = df['y'].iloc[-14:].values

    # Prophet
    logger.info("Training Prophet...")
    prop = train_prophet(train_df)
    prop_eval = evaluate(true_vals, prop.values)

    # ARIMA
    logger.info("Training ARIMA...")
    arima = train_arima(train_df)
    arima_eval = evaluate(true_vals, arima.values)

    # LSTM
    if args.lstm_framework == "keras":
        logger.info("Training LSTM with Keras...")
        lstm = train_lstm_keras(train_df)
    else:
        logger.info("Training LSTM with PyTorch...")
        lstm = train_lstm_pytorch(train_df)
    lstm_eval = evaluate(true_vals, lstm.values)

    # Save forecasts
    out_df = pd.DataFrame({
        'ds': df['ds'].iloc[-14:].values,
        'true': true_vals,
        'prophet': prop.values,
        'arima': arima.values,
        'lstm': lstm.values
    })
    out_df.to_csv(args.output_dir / "ts_forecasts.csv", index=False)
    logger.info(f"Saved forecasts to {args.output_dir/'ts_forecasts.csv'}")

    # Save metrics
    metrics = {'prophet': prop_eval, 'arima': arima_eval, 'lstm': lstm_eval}
    with open(args.output_dir / "ts_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {args.output_dir/'ts_metrics.json'}")


if __name__ == "__main__":
    main()
