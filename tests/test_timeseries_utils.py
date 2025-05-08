"""
Unit tests for time-series forecasting utilities.
"""

import pandas as pd
import pytest
from typing import Any

from src.timeseries.predict import (
    predict_arima,
    predict_prophet,
    predict_lstm,
)


def test_predict_arima(tmp_timeseries_models: dict[str, Any]) -> None:
    """
    ARIMA forecast must return a DataFrame with columns ['ds','yhat']
    and length equal to the horizon.
    """
    history = tmp_timeseries_models["history"]
    arima_path = tmp_timeseries_models["arima"]

    out = predict_arima(arima_path, history, horizon=24)
    assert list(out.columns) == ["ds", "yhat"]
    assert len(out) == 24


def test_predict_prophet(tmp_timeseries_models: dict[str, Any]) -> None:
    """
    Prophet forecast must return a DataFrame with columns ['ds','yhat']
    and length equal to the horizon.
    """
    history = tmp_timeseries_models["history"]
    prophet_path = tmp_timeseries_models["prophet"]

    out = predict_prophet(prophet_path, history, horizon=24)
    assert list(out.columns) == ["ds", "yhat"]
    assert len(out) == 24


def test_predict_lstm(tmp_timeseries_models: dict[str, Any]) -> None:
    """
    LSTM forecast must return a DataFrame with columns ['ds','yhat']
    and length equal to the horizon.
    """
    history = tmp_timeseries_models["history"]
    lstm_path, params_path = tmp_timeseries_models["lstm"]

    out = predict_lstm(lstm_path, params_path, history, horizon=24)
    assert list(out.columns) == ["ds", "yhat"]
    assert len(out) == 24
