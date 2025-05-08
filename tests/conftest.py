"""
Pytest configuration and common fixtures.

- Inserts project root into sys.path so that `src` and `api` are importable.
- Provides fixtures for temporary classification and time-series models.
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import joblib

# 1) Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def tmp_classification_model(tmp_path: Path) -> Path:
    """
    Train a tiny LogisticRegression on random 6-feature data,
    save as .pkl, and return the path.
    """
    from sklearn.linear_model import LogisticRegression

    # 20 samples, 6 features
    X = np.random.rand(20, 6)
    y = np.random.randint(0, 2, size=20)
    model = LogisticRegression().fit(X, y)

    path = tmp_path / "lr_model.pkl"
    joblib.dump(model, path)
    return path


@pytest.fixture
def tmp_timeseries_models(tmp_path: Path) -> dict[str, any]:
    """
    Create and save ARIMA, Prophet, and LSTM models on a constant series.
    Returns dict with:
      - history: pd.DataFrame
      - arima: Path to .pkl
      - prophet: Path to .pkl
      - lstm: (Path to .h5, Path to params.json)
    """
    from statsmodels.tsa.arima.model import ARIMA
    from prophet import Prophet
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import LSTM, Dense

    # constant series of 50 hourly points
    dates = pd.date_range("2021-01-01", periods=50, freq="H")
    df = pd.DataFrame({"ds": dates, "y": np.ones(len(dates))})

    # ARIMA(0,0,0)
    arima_m = ARIMA(df["y"][:-24], order=(0, 0, 0)).fit()
    arima_path = tmp_path / "arima_model.pkl"
    joblib.dump(arima_m, arima_path)

    # Prophet
    prophet_m = Prophet().fit(df.rename(columns={"y": "y"})[:-24])
    prophet_path = tmp_path / "prophet_model.pkl"
    joblib.dump(prophet_m, prophet_path)

    # LSTM with window=24
    window = 24
    series = df["y"].values
    X_list, y_list = [], []
    for i in range(len(series) - window - 24):
        X_list.append(series[i : i + window])
        y_list.append(series[i + window])
    X_arr = np.array(X_list).reshape(-1, window, 1)
    y_arr = np.array(y_list)

    tf.random.set_seed(0)
    lstm = Sequential([LSTM(8, input_shape=(window, 1)), Dense(1)])
    lstm.compile(optimizer="adam", loss="mse")
    lstm.fit(X_arr, y_arr, epochs=1, verbose=0)

    lstm_model_path = tmp_path / "lstm_model.h5"
    lstm.save(lstm_model_path)

    params = {"units": window}
    params_path = tmp_path / "lstm_params.json"
    params_path.write_text(json.dumps(params), encoding="utf-8")

    return {
        "history": df,
        "arima": arima_path,
        "prophet": prophet_path,
        "lstm": (lstm_model_path, params_path)
    }
