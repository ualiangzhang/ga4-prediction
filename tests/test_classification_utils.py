"""
Unit tests for classification preprocessing and sklearn inference utilities.
"""

import numpy as np
import pandas as pd
from pathlib import Path

import pytest

from src.classification.predict import preprocess_features, predict_sklearn


def test_preprocess_features() -> None:
    """
    preprocess_features should:
      - drop 'user_pseudo_id'
      - fill missing
      - factorize categorical columns
      - return numeric numpy array of shape (n,6)
    """
    df = pd.DataFrame([
        {
            "user_pseudo_id": "u1",
            "num_pageviews": 5,
            "num_addtocart": 1,
            "num_sessions": 2,
            "device_type": "mobile",
            "traffic_source": "google",
            "country": "US"
        },
        {
            "user_pseudo_id": "u2",
            "num_pageviews": 3,
            "num_addtocart": 0,
            "num_sessions": 1,
            "device_type": "desktop",
            "traffic_source": None,
            "country": None
        },
    ])
    X = preprocess_features(df)
    assert X.shape == (2, 6)
    assert np.issubdtype(X.dtype, np.integer) or np.issubdtype(X.dtype, np.floating)


def test_predict_sklearn(tmp_classification_model: Path) -> None:
    """
    predict_sklearn should return labels & probabilities of correct shape.
    """
    # 4 samples, 6 features
    X = np.random.rand(4, 6)
    labels, proba = predict_sklearn(tmp_classification_model, X)

    assert labels.shape == (4,)
    assert proba.shape == (4,)
    assert set(labels.tolist()) <= {0, 1}
    assert np.all((proba >= 0.0) & (proba <= 1.0))
