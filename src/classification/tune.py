#!/usr/bin/env python3
# File: src/classification/tune.py

"""
Hyperparameter tuning for GA4 conversion‐prediction models using Optuna.

Supports tuning:
  - XGBoost ("xgb")
  - RandomForest ("rf")
  - LogisticRegression ("lr")
  - Keras DNN ("dnn")

Pipeline:
  1. Load & preprocess data
  2. Train/test split
  3. SMOTE to 10% minority
  4. Hybrid sampling (undersample → oversample)
  5. Optuna objective: sample hyperparameters, train model on training set,
     evaluate PR AUC on test set
  6. Report & save best parameters

Usage:
  python tune.py --model xgb --trials 50 --output best_xgb_params.json
"""

from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import optuna
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from xgboost import XGBClassifier

# ─── Logging Setup ────────────────────────────────────────────────────────────────
logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────────
SMOTE_RATIO = 0.1  # upsample minority to 10% of majority


def load_data(csv_path: Path) -> pd.DataFrame:
    """
    Load processed GA4 features CSV.

    Args:
        csv_path: Path to CSV file.

    Returns:
        DataFrame with features and 'converted' label.
    """
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} rows from {csv_path}")
    return df


def preprocess(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encode categorical features and split into X and y.

    Args:
        df: Raw DataFrame.

    Returns:
        X: Feature matrix.
        y: Label vector.
    """
    df = df.copy().fillna("unknown")
    for col in ("device_type", "traffic_source", "country"):
        df[col], _ = pd.factorize(df[col])
    X = df.drop(columns=["user_pseudo_id", "converted"], errors="ignore").values
    y = df["converted"].astype(int).values
    return X, y


def hybrid_sample(X: np.ndarray, y: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hybrid sampling: undersample majority to 2× minority, then oversample minority.

    Args:
        X: Feature matrix.
        y: Labels.
        seed: Random seed.

    Returns:
        X_res, y_res: Resampled feature matrix and labels.
    """
    mask = y == 1
    X_pos, y_pos = X[mask], y[mask]
    X_neg, y_neg = X[~mask], y[~mask]

    # 1) Undersample majority to 2× minority
    target = min(len(y_neg), 2 * len(y_pos))
    X_neg_down, y_neg_down = resample(X_neg, y_neg,
                                      replace=False,
                                      n_samples=target,
                                      random_state=seed)
    # 2) Oversample minority to match
    X_pos_up, y_pos_up = resample(X_pos, y_pos,
                                  replace=True,
                                  n_samples=len(y_neg_down),
                                  random_state=seed)
    X_comb = np.vstack([X_neg_down, X_pos_up])
    y_comb = np.concatenate([y_neg_down, y_pos_up])
    perm = np.random.RandomState(seed).permutation(len(y_comb))
    return X_comb[perm], y_comb[perm]


def objective(trial: optuna.trial.Trial,
              X_train: np.ndarray, y_train: np.ndarray,
              X_test: np.ndarray, y_test: np.ndarray,
              model_name: str) -> float:
    """
    Optuna objective: sample hyperparameters, train, and evaluate PR AUC.

    Args:
        trial: Optuna trial object.
        X_train, y_train: Training data.
        X_test, y_test: Test data.
        model_name: One of 'xgb','rf','lr','dnn'.

    Returns:
        average precision score on test set.
    """
    # Sample and train model according to model_name
    if model_name == "xgb":
        params: Dict[str, Any] = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 0.5),
            "random_state": 42,
            "use_label_encoder": False,
            "eval_metric": "logloss"
        }
        # Compute scale_pos_weight
        neg, pos = np.bincount(y_train)
        params["scale_pos_weight"] = neg / pos
        model = XGBClassifier(**params)
    elif model_name == "rf":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "class_weight": "balanced",
            "random_state": 42
        }
        model = RandomForestClassifier(**params)
    elif model_name == "lr":
        params = {
            "C": trial.suggest_loguniform("C", 1e-4, 10.0),
            "solver": "liblinear",
            "class_weight": "balanced",
            "random_state": 42
        }
        model = LogisticRegression(**params)
    else:  # DNN via Keras
        try:
            import tensorflow as tf
            from tensorflow.keras import Sequential
            from tensorflow.keras.layers import Dense, Dropout
        except ImportError:
            raise RuntimeError("TensorFlow required for DNN tuning")
        # Sample architecture
        n_layers = trial.suggest_int("n_layers", 1, 3)
        hidden_units = [
            trial.suggest_int(f"n_units_l{i}", 16, 128, log=True)
            for i in range(n_layers)
        ]
        dropout = trial.suggest_uniform("dropout", 0.0, 0.5)
        lr = trial.suggest_loguniform("learning_rate", 1e-4, 1e-1)
        params = {"hidden_units": hidden_units, "dropout": dropout, "learning_rate": lr}
        # Build model
        tf.random.set_seed(42)
        model = Sequential()
        inp_dim = X_train.shape[1]
        for units in hidden_units:
            model.add(Dense(units, activation="relu", input_shape=(inp_dim,)))
            model.add(Dropout(dropout))
            inp_dim = units
        model.add(Dense(1, activation="sigmoid"))
        model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                      loss="binary_crossentropy")
        # Compute class weights
        cw = {0: len(y_train) / np.sum(y_train == 0), 1: len(y_train) / np.sum(y_train == 1)}
        model.fit(X_train, y_train,
                  validation_split=0.1,
                  epochs=trial.suggest_int("epochs", 10, 50),
                  batch_size=trial.suggest_int("batch_size", 16, 128),
                  class_weight=cw,
                  verbose=0)
        y_proba = model.predict(X_test).ravel()
        return average_precision_score(y_test, y_proba)

    # Train and evaluate for sklearn models
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]
    return average_precision_score(y_test, y_proba)


def main() -> None:
    """
    Parse arguments, prepare data, and run Optuna study.
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    default_data = project_root / "etl" / "data" / "processed" / "ga4_training_data.csv"

    parser = argparse.ArgumentParser(description="Optuna hyperparameter tuning")
    parser.add_argument("--model", choices=["xgb", "rf", "lr", "dnn"], default="dnn")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--data", type=Path, default=default_data)
    parser.add_argument("--output", type=Path, help="Path to save best params JSON")
    args = parser.parse_args()

    # Load & preprocess
    df = load_data(args.data)
    X, y = preprocess(df)

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # SMOTE to 10% minority
    sm = SMOTE(sampling_strategy=SMOTE_RATIO, random_state=42)
    X_sm, y_sm = sm.fit_resample(X_train, y_train)
    logger.info(f"Post-SMOTE class counts: {np.bincount(y_sm)}")

    # Hybrid sampling
    X_res, y_res = hybrid_sample(X_sm, y_sm, seed=42)
    logger.info(f"Post-Hybrid class counts: {np.bincount(y_res)}")

    # Run Optuna study
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(lambda t: objective(t, X_res, y_res, X_test, y_test, args.model),
                   n_trials=args.trials)

    # Output results
    logger.info(f"Best PR AUC: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")

    # Save best parameters
    if args.output:
        args.output.write_text(json.dumps(study.best_params, indent=2), encoding="utf-8")
        logger.info(f"Saved best params to {args.output}")


if __name__ == "__main__":
    main()
