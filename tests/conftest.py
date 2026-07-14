"""Shared pytest fixtures for the IPO GMP predictor tests."""

import json
import pickle
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def _row_defaults(**overrides) -> dict:
    base = {
        "gmp_rs": 50.0, "gmp_pct": 25.0, "gmp_day1": 20.0, "gmp_day2": 40.0,
        "qib_subscription": 30.0, "nii_subscription": 60.0, "retail_subscription": 10.0,
        "total_subscription": 30.0, "issue_price": 200.0, "lot_size": 50,
        "issue_size_cr": 500.0, "category": "Mainboard", "sector": "Technology",
        "close_date": "2025-01-10", "listing_date": "2025-01-22",
    }
    base.update(overrides)
    return base


@pytest.fixture
def make_ipo_row():
    """Factory fixture: build a single-row DataFrame of raw (pre-feature-engineering) IPO fields."""
    def _make(**overrides) -> pd.DataFrame:
        return pd.DataFrame([_row_defaults(**overrides)])
    return _make


@pytest.fixture
def make_predict_input():
    """Factory fixture: build a predict_one()-shaped input dict."""
    def _make(**overrides) -> dict:
        base = {
            "gmp_rs": 50, "issue_price": 200, "gmp_day1": 25, "gmp_day2": 40,
            "total_subscription": 30, "qib_subscription": 20, "nii_subscription": 50,
            "retail_subscription": 8, "issue_size_cr": 500, "lot_size": 50,
            "category": "Mainboard", "sector": "Technology",
            "close_date": "2025-01-10", "listing_date": "2025-01-22",
        }
        base.update(overrides)
        return base
    return _make


@pytest.fixture
def trained_models():
    """Load the trained regressor/classifier/meta, or skip the test if not trained yet."""
    from model import MODEL_DIR

    try:
        with open(MODEL_DIR / "regressor.pkl", "rb") as f:
            regressor = pickle.load(f)
        with open(MODEL_DIR / "classifier.pkl", "rb") as f:
            classifier = pickle.load(f)
        with open(MODEL_DIR / "meta.json") as f:
            meta = json.load(f)
    except FileNotFoundError:
        pytest.skip("Models not trained yet. Run python src/model.py first.")
    return regressor, classifier, meta
