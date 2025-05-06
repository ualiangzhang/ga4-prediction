#!/usr/bin/env python3
# File: src/classification/train.py

"""
Train a conversion prediction model on GA4 user features.

Usage:
  # Use defaults (expects etl/data/processed/ga4_training_data.csv under project root)
  python train.py

  # Override data path, model type, and output directory
  python train.py \
    --data /full/path/to/ga4_training_data.csv \
    --model rf \
    --output-dir /full/path/to/models
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Tuple, Union, Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# ─── Setup Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ─── Default Hyperparameters ─────────────────────────────────────────────────────
DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {
    "xgb": {
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "random_state": 42
    },
    "rf": {
        "n_estimators": 100,
        "max_depth": None,
        "random_state": 42
    },
    "lr": {
        "solver": "liblinear",
        "C": 1.0,
        "random_state": 42
    }
}


# ─── Data Loading & Preprocessing ────────────────────────────────────────────────
def load_data(csv_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load the processed GA4 training data from a CSV file.

    Args:
        csv_path (Union[str, Path]): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"❌ Training data not found at: {path}\n"
            "Please run the ETL step to generate ga4_training_data.csv"
        )
    logger.info(f"Loading data from {path}")
    return pd.read_csv(path)


def preprocess_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess the DataFrame: drop ID, fill missing, encode categoricals, split into X/y.

    Args:
        df (pd.DataFrame): Raw DataFrame with columns:
            - user_pseudo_id (optional)
            - numeric: num_pageviews, num_addtocart, num_sessions
            - categorical: device_type, traffic_source, country
            - label: converted

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            X: Feature matrix (n_samples, n_features)
            y: Label vector (n_samples,)
    """
    df_clean = df.copy()

    # Drop identifier column
    if "user_pseudo_id" in df_clean:
        df_clean.drop(columns=["user_pseudo_id"], inplace=True)

    # Fill missing values
    df_clean.fillna("unknown", inplace=True)

    # Encode categorical features
    for col in ["device_type", "traffic_source", "country"]:
        if col in df_clean.columns:
            df_clean[col], _ = pd.factorize(df_clean[col])

    # Separate X and y
    X = df_clean.drop(columns=["converted"]).values
    y = df_clean["converted"].astype(int).values

    logger.info(f"Preprocessed data: X shape = {X.shape}, y shape = {y.shape}")
    return X, y


# ─── Model Factory ────────────────────────────────────────────────────────────────
def get_model(model_name: str, params: Dict[str, Any]):
    """
    Instantiate a classifier based on model_name and provided parameters.

    Args:
        model_name (str): One of 'xgb', 'rf', 'lr'.
        params (dict): Hyperparameters for the chosen model.

    Returns:
        An untrained sklearn-compatible classifier.

    Raises:
        ValueError: If model_name is not supported.
    """
    name = model_name.lower()
    if name == "xgb":
        return XGBClassifier(**params)
    if name == "rf":
        return RandomForestClassifier(**params)
    if name == "lr":
        return LogisticRegression(**params)
    raise ValueError(f"Unsupported model '{model_name}'. Choose from 'xgb', 'rf', 'lr'.")


# ─── Evaluation ──────────────────────────────────────────────────────────────────
def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> None:
    """
    Evaluate the model on test data and log classification metrics.

    Args:
        model: Trained classifier (supports predict & predict_proba).
        X_test (np.ndarray): Test feature matrix.
        y_test (np.ndarray): True labels.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    logger.info("=== Classification Report ===")
    logger.info("\n" + classification_report(y_test, y_pred, digits=4))

    auc = roc_auc_score(y_test, y_proba)
    logger.info(f"ROC AUC Score: {auc:.4f}")


# ─── Main Routine ────────────────────────────────────────────────────────────────
def main() -> None:
    """
    Parse command-line args, load & preprocess data, train chosen model, evaluate,
    and save both the trained model and its parameters to disk.
    """
    # Resolve project root and defaults
    project_root = Path(__file__).resolve().parent.parent.parent
    default_data = project_root / "etl" / "data" / "processed" / "ga4_training_data.csv"
    default_output_dir = project_root / "models"

    parser = argparse.ArgumentParser(
        description="Train a GA4 conversion prediction model."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=default_data,
        help="Path to CSV training data (default: etl/data/processed/ga4_training_data.csv)"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(DEFAULT_PARAMS.keys()),
        default="xgb",
        help="Model to train: xgb|rf|lr (default: xgb)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="Directory to save trained models and parameters (default: models/)"
    )
    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load & preprocess data
    df = load_data(args.data)
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Prepare model and parameters
    params = DEFAULT_PARAMS[args.model]
    model = get_model(args.model, params)
    logger.info(f"Training '{args.model}' with parameters: {params}")

    # Train and evaluate
    model.fit(X_train, y_train)
    evaluate_model(model, X_test, y_test)

    # Save model and params
    model_path = args.output_dir / f"{args.model}_model.pkl"
    params_path = args.output_dir / f"{args.model}_params.json"

    joblib.dump(model, model_path)
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

    logger.info(f"✅ Model saved to {model_path}")
    logger.info(f"✅ Parameters saved to {params_path}")


if __name__ == "__main__":
    main()
