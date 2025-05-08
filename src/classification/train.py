#!/usr/bin/env python3
# File: src/classification/train.py

"""
Train a GA4 conversion-prediction model (XGB, RF, LR, or Keras DNN)
using SMOTE + hybrid sampling, plus class weighting,
with comprehensive evaluation.

When --model dnn is specified, TensorFlow is imported on-demand.
"""

from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef
)
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from xgboost import XGBClassifier

# ─── Logging Setup ────────────────────────────────────────────────────────────────
logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────────
SMOTE_RATIO = 0.1  # upsample minority to 10% of majority

# ─── Default Hyperparameters ──────────────────────────────────────────────────────
DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {
    "xgb": {
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "n_estimators": 236,
        "max_depth": 3,
        "learning_rate": 0.44928043200594214,
        "random_state": 42
    },
    "rf": {
        "n_estimators": 292,
        "max_depth": 5,
        "class_weight": "balanced",
        "random_state": 42
    },
    "lr": {
        "solver": "liblinear",
        "C": 0.0074593432857265485,
        "class_weight": "balanced",
        "random_state": 42
    },
    "dnn": {
        "hidden_units": [64],
        "dropout": 0.1207472651192055,
        "epochs": 50,
        "batch_size": 32,
        "learning_rate": 0.0007597392269837131,
        "random_state": 42
    }
}


def load_params(model_name: str, param_file: Path | None) -> Dict[str, Any]:
    """
    Load hyperparameters from JSON or fallback to DEFAULT_PARAMS.
    """
    if param_file and param_file.is_file():
        params = json.loads(param_file.read_text(encoding="utf-8"))
        logger.info(f"Loaded params from {param_file}")
    else:
        params = DEFAULT_PARAMS[model_name].copy()
        logger.info("Using default parameters")
    return params


def load_data(csv_path: Union[str, Path]) -> pd.DataFrame:
    """
    Read processed GA4 training CSV.
    """
    p = Path(csv_path)
    if not p.is_file():
        raise FileNotFoundError(f"Training file not found: {p}")
    df = pd.read_csv(p)
    logger.info(f"Loaded {len(df)} rows from {p}")
    return df


def preprocess(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encode categorical columns and split into features X and label y.
    """
    df = df.copy()
    df.fillna("unknown", inplace=True)
    for col in ("device_type", "traffic_source", "country"):
        df[col], _ = pd.factorize(df[col])
    X = df.drop(columns=["user_pseudo_id", "converted"], errors="ignore").values
    y = df["converted"].astype(int).values
    return X, y


def hybrid_sample(
        X: np.ndarray, y: np.ndarray, random_state: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hybrid sampling:
      1) Undersample majority to 2× minority
      2) Oversample minority to match majority
    """
    pos_mask = y == 1
    X_pos, y_pos = X[pos_mask], y[pos_mask]
    X_neg, y_neg = X[~pos_mask], y[~pos_mask]

    # 1) Undersample majority to 2× minority
    n_target = 2 * len(y_pos)
    X_neg_down, y_neg_down = resample(
        X_neg, y_neg,
        replace=False,
        n_samples=min(n_target, len(y_neg)),
        random_state=random_state
    )

    # 2) Oversample minority to match
    X_pos_up, y_pos_up = resample(
        X_pos, y_pos,
        replace=True,
        n_samples=len(y_neg_down),
        random_state=random_state
    )

    X_comb = np.vstack([X_neg_down, X_pos_up])
    y_comb = np.concatenate([y_neg_down, y_pos_up])

    # Shuffle
    perm = np.random.RandomState(random_state).permutation(len(y_comb))
    return X_comb[perm], y_comb[perm]


def evaluate_all(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> None:
    """
    Print comprehensive evaluation:
      - classification report
      - confusion matrix
      - ROC AUC
      - Precision-Recall AUC
      - balanced accuracy
      - Matthews correlation coefficient
    """
    logger.info("=== Classification Report ===")
    logger.info("\n" + classification_report(y_true, y_pred, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    logger.info(f"Confusion Matrix:\n{cm}")

    roc = roc_auc_score(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    logger.info(f"ROC AUC:            {roc:.4f}")
    logger.info(f"Precision-Recall AUC: {pr_auc:.4f}")
    logger.info(f"Balanced Accuracy:  {bal_acc:.4f}")
    logger.info(f"Matthews Corrcoef:  {mcc:.4f}")


def train_dnn_keras(
        X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
        params: Dict[str, Any]
) -> Any:
    """
    Train a Keras DNN with class weights.
    TensorFlow is imported on-demand here.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
    except ImportError as e:
        raise RuntimeError("TensorFlow is required for --model dnn") from e

    tf.random.set_seed(params["random_state"])
    model = Sequential()
    dims = X_train.shape[1]
    # Hidden layers
    for units in params["hidden_units"]:
        model.add(Dense(units, activation="relu", input_shape=(dims,)))
        model.add(Dropout(params["dropout"]))
        dims = units
    # Output layer
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(params["learning_rate"]),
        loss="binary_crossentropy"
    )

    # Class weights
    class_weights = {
        0: len(y_train) / np.sum(y_train == 0),
        1: len(y_train) / np.sum(y_train == 1)
    }

    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        class_weight=class_weights,
        callbacks=[es],
        verbose=1
    )
    return model


def main() -> None:
    """
    Main training routine.
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    default_data = project_root / "etl" / "data" / "processed" / "ga4_training_data.csv"
    default_out = project_root / "models" / "classification"

    parser = argparse.ArgumentParser(description="GA4 classification training")
    parser.add_argument("--data", type=Path, default=default_data)
    parser.add_argument("--model", choices=["xgb", "rf", "lr", "dnn"], default="xgb")
    parser.add_argument("--output-dir", type=Path, default=default_out)
    parser.add_argument("--param-file", type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    params = load_params(args.model, args.param_file)

    # 1) Load & preprocess
    df = load_data(args.data)
    X, y = preprocess(df)

    # 2) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=params["random_state"]
    )
    logger.info(f"Train dist. before SMOTE: {np.bincount(y_train)}")

    # 3) SMOTE up to SMOTE_RATIO of majority
    sm = SMOTE(sampling_strategy=SMOTE_RATIO, random_state=params["random_state"])
    X_sm, y_sm = sm.fit_resample(X_train, y_train)
    logger.info(f"After SMOTE (ratio={SMOTE_RATIO}): {np.bincount(y_sm)}")

    # 4) Hybrid sampling on SMOTE output
    X_res, y_res = hybrid_sample(X_sm, y_sm, params["random_state"])
    logger.info(f"After Hybrid sampling: {np.bincount(y_res)}")

    # 5) Train the chosen model
    if args.model == "dnn":
        # further split for validation
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_res, y_res, test_size=0.1, stratify=y_res, random_state=params["random_state"]
        )
        logger.info("Training Keras DNN…")
        model = train_dnn_keras(X_tr, y_tr, X_val, y_val, params)
        y_proba = model.predict(X_test).ravel()
        y_pred = (y_proba >= 0.5).astype(int)
        model_path = args.output_dir / "dnn_model.h5"
        model.save(model_path)
    else:
        if args.model == "xgb":
            neg, pos = np.bincount(y_res)
            params["scale_pos_weight"] = neg / pos
        clf = (
            XGBClassifier(**params)
            if args.model == "xgb"
            else (
                RandomForestClassifier(**params) if args.model == "rf"
                else LogisticRegression(**params)
            )
        )
        logger.info(f"Training {args.model.upper()}…")
        clf.fit(X_res, y_res)
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]
        model_path = args.output_dir / f"{args.model}_model.pkl"
        joblib.dump(clf, model_path)

    # 6) Comprehensive evaluation
    evaluate_all(y_test, y_pred, y_proba)

    # 7) Save parameters
    params_out = args.output_dir / f"{args.model}_params.json"
    params_out.write_text(json.dumps(params, indent=2), encoding="utf-8")
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Params saved to {params_out}")


if __name__ == "__main__":
    main()
