#!/usr/bin/env python3
# File: src/classification/predict.py

"""
Load a trained GA4 conversion prediction model and run inference on new data.

Usage:
  # Use defaults (expects etl/data/processed/ga4_training_data.csv under project root)
  python predict.py

  # Override input data, model file, and output path
  python predict.py \
    --data /path/to/new_data.csv \
    --model-file /path/to/models/rf_model.pkl \
    --output /path/to/predictions.csv
"""

import argparse
import logging
from pathlib import Path
from typing import Union

import joblib
import numpy as np
import pandas as pd

# ─── Setup Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def load_data(csv_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load feature data from CSV for inference.

    Args:
        csv_path (Union[str, Path]): Path to the CSV file containing features.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"❌ Input data not found at: {path}")
    logger.info(f"Loading inference data from {path}")
    return pd.read_csv(path)


def preprocess_features(df: pd.DataFrame) -> np.ndarray:
    """
    Prepare feature matrix for model prediction: drop ID/label, encode categoricals, fill missing.

    Args:
        df (pd.DataFrame): Raw DataFrame containing columns:
            - user_pseudo_id (optional)
            - converted (optional label)
            - num_pageviews, num_addtocart, num_sessions (numeric)
            - device_type, traffic_source, country (categorical)

    Returns:
        np.ndarray: Feature matrix with shape (n_samples, 6).
    """
    df_clean = df.copy()

    # Drop identifier column
    if "user_pseudo_id" in df_clean.columns:
        df_clean.drop(columns=["user_pseudo_id"], inplace=True)
        logger.debug("Dropped 'user_pseudo_id' column")

    # Drop label column if present
    if "converted" in df_clean.columns:
        df_clean.drop(columns=["converted"], inplace=True)
        logger.debug("Dropped 'converted' label column")

    # Fill missing values
    df_clean.fillna("unknown", inplace=True)

    # Encode categorical features
    for col in ["device_type", "traffic_source", "country"]:
        if col in df_clean.columns:
            df_clean[col], _ = pd.factorize(df_clean[col])
            logger.debug(f"Factorized column '{col}'")

    # Final feature matrix
    X = df_clean.values
    logger.info(f"Preprocessed features: shape {X.shape}")
    return X


def main() -> None:
    """
    Parse args, load model & data, run predictions, and save results.
    """
    # Defaults relative to project root
    project_root = Path(__file__).resolve().parent.parent.parent
    default_data = project_root / "etl" / "data" / "processed" / "ga4_training_data.csv"
    default_model = project_root / "models" / "xgb_model.pkl"
    default_output = project_root / "models" / "predictions.csv"

    parser = argparse.ArgumentParser(
        description="Run inference with a trained GA4 conversion model."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=default_data,
        help="Path to input CSV features (default: etl/data/processed/ga4_training_data.csv)"
    )
    parser.add_argument(
        "--model-file",
        type=Path,
        default=default_model,
        help="Path to trained model pickle (default: models/xgb_model.pkl)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="Path to save predictions CSV (default: models/predictions.csv)"
    )
    args = parser.parse_args()

    # Load features
    df = load_data(args.data)
    X = preprocess_features(df)

    # Load model
    if not args.model_file.is_file():
        raise FileNotFoundError(f"❌ Model file not found at: {args.model_file}")
    logger.info(f"Loading model from {args.model_file}")
    model = joblib.load(args.model_file)

    # Predict probabilities and classes
    logger.info("Running predictions...")
    # Some models may not have predict_proba (but ours do)
    try:
        proba: np.ndarray = model.predict_proba(X)[:, 1]
    except AttributeError:
        logger.info("Model does not support predict_proba; using predict() for labels only")
        labels = model.predict(X)
        proba = labels.astype(float)
    labels = (proba >= 0.5).astype(int)

    # Attach predictions to DataFrame
    result = df.copy()
    result["predicted_proba"] = proba
    result["predicted_label"] = labels

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    logger.info(f"✅ Predictions saved to {args.output}")


if __name__ == "__main__":
    main()
