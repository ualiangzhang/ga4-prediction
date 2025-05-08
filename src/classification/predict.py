#!/usr/bin/env python3
# File: src/classification/predict.py

"""
Run inference with a trained GA4 conversion model.

Models and their params live under:
    models/classification/

Defaults:
  - XGBoost:        models/classification/xgb_model.pkl
  - RandomForest:   models/classification/rf_model.pkl
  - LogisticRegress:models/classification/lr_model.pkl
  - Keras DNN:      models/classification/dnn_model.h5

Usage:
  # Default (XGBoost):
  python predict.py

  # RandomForest:
  python predict.py --model-type rf

  # Keras DNN:
  python predict.py --model-type dnn
"""

from __future__ import annotations
import argparse
import logging
from pathlib import Path
from typing import Union, Tuple

import joblib
import numpy as np
import pandas as pd

# ─── Logging Setup ────────────────────────────────────────────────────────────────
logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(csv_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load processed feature CSV for inference.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        DataFrame of features.
    """
    p = Path(csv_path)
    if not p.is_file():
        raise FileNotFoundError(f"Input file not found: {p}")
    df = pd.read_csv(p)
    logger.info(f"Loaded {len(df)} rows from {p}")
    return df


def preprocess_features(df: pd.DataFrame) -> np.ndarray:
    """
    Drop identifiers and labels, fill missing, and factorize categoricals.

    Args:
        df: Raw feature DataFrame.

    Returns:
        Feature matrix of shape (n_samples, n_features).
    """
    df_clean = df.copy()
    for col in ("user_pseudo_id", "converted"):
        if col in df_clean:
            df_clean.drop(columns=[col], inplace=True)
            logger.debug(f"Dropped column '{col}'")
    df_clean.fillna("unknown", inplace=True)
    for col in ("device_type", "traffic_source", "country"):
        if col in df_clean:
            df_clean[col], _ = pd.factorize(df_clean[col])
            logger.debug(f"Encoded '{col}'")
    X = df_clean.values
    logger.info(f"Feature matrix shape: {X.shape}")
    return X


def predict_sklearn(model_file: Path, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a sklearn .pkl model and predict probabilities & labels.

    Args:
        model_file: Path to a .pkl file.
        X: Feature matrix.

    Returns:
        labels (0/1) and probability of class 1.
    """
    model = joblib.load(model_file)
    logger.info(f"Loaded sklearn model from {model_file}")
    proba = model.predict_proba(X)[:, 1]
    labels = (proba >= 0.5).astype(int)
    return labels, proba


def predict_keras(model_file: Path, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a Keras .h5 model and predict probabilities & labels.

    Args:
        model_file: Path to a .h5 Keras model.
        X: Feature matrix.

    Returns:
        labels (0/1) and probability of class 1.
    """
    try:
        from tensorflow.keras.models import load_model
    except ImportError as e:
        raise RuntimeError("TensorFlow/Keras is required for DNN inference") from e

    model = load_model(model_file)
    logger.info(f"Loaded Keras DNN from {model_file}")
    proba = model.predict(X).ravel()
    labels = (proba >= 0.5).astype(int)
    return labels, proba


def main() -> None:
    """
    Parse arguments, run inference, and save predictions.
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    default_data = project_root / "etl" / "data" / "processed" / "ga4_training_data.csv"
    classification_dir = project_root / "models" / "classification"
    default_output = project_root / "models" / "predictions.csv"

    parser = argparse.ArgumentParser(description="GA4 conversion model inference")
    parser.add_argument(
        "--data", type=Path, default=default_data,
        help="Input features CSV"
    )
    parser.add_argument(
        "--model-type", type=str, default="dnn",
        choices=["xgb", "rf", "lr", "dnn"],
        help="Which model to use"
    )
    parser.add_argument(
        "--model-file", type=Path,
        help="Override model file path"
    )
    parser.add_argument(
        "--output", type=Path, default=default_output,
        help="Path to save predictions CSV"
    )
    args = parser.parse_args()

    # Determine model file
    if args.model_file:
        model_file = args.model_file
    else:
        ext = ".h5" if args.model_type == "dnn" else ".pkl"
        model_file = classification_dir / f"{args.model_type}_model{ext}"
    if not model_file.is_file():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    # Load and preprocess features
    df = load_data(args.data)
    X = preprocess_features(df)

    # Predict
    if args.model_type == "dnn":
        labels, proba = predict_keras(model_file, X)
    else:
        labels, proba = predict_sklearn(model_file, X)

    # Append predictions and save
    df["predicted_label"] = labels
    df["predicted_proba"] = proba
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    logger.info(f"Predictions saved to {args.output}")


if __name__ == "__main__":
    main()
