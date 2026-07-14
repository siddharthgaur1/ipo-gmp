"""
Tests for IPO GMP Predictor
Run: pytest tests/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import pandas as pd


# ── model.py tests ───────────────────────────────────────────────
class TestFeatureEngineering:
    def test_gmp_momentum_positive(self, make_ipo_row):
        from model import engineer_features
        result = engineer_features(make_ipo_row(gmp_day1=20.0, gmp_rs=60.0))
        assert result["gmp_momentum"].iloc[0] > 0, "GMP increasing should give positive momentum"

    def test_gmp_momentum_negative(self, make_ipo_row):
        from model import engineer_features
        result = engineer_features(make_ipo_row(gmp_day1=80.0, gmp_rs=40.0))
        assert result["gmp_momentum"].iloc[0] < 0, "Falling GMP should give negative momentum"

    def test_is_sme_flag(self, make_ipo_row):
        from model import engineer_features
        df_sme = engineer_features(make_ipo_row(category="SME"))
        df_main = engineer_features(make_ipo_row(category="Mainboard"))
        assert df_sme["is_sme"].iloc[0] == 1
        assert df_main["is_sme"].iloc[0] == 0

    def test_days_to_listing(self, make_ipo_row):
        from model import engineer_features
        result = engineer_features(make_ipo_row(close_date="2025-01-10", listing_date="2025-01-22"))
        assert result["days_to_listing"].iloc[0] == 12

    def test_log_transforms_positive(self, make_ipo_row):
        from model import engineer_features
        df = engineer_features(make_ipo_row(issue_price=500, issue_size_cr=2000))
        assert df["issue_price_log"].iloc[0] > 0
        assert df["issue_size_cr_log"].iloc[0] > 0

    def test_sector_encoding(self, make_ipo_row):
        from model import engineer_features, SECTOR_MAP
        for sector in SECTOR_MAP:
            df = engineer_features(make_ipo_row(sector=sector))
            assert df["sector_encoded"].iloc[0] == SECTOR_MAP[sector]

    def test_all_features_present(self, make_ipo_row):
        from model import engineer_features, FEATURE_COLS
        df = engineer_features(make_ipo_row())
        for col in FEATURE_COLS:
            assert col in df.columns, f"Missing feature: {col}"

    def test_momentum_clipped(self, make_ipo_row):
        from model import engineer_features
        # Edge case: huge GMP jump
        df = engineer_features(make_ipo_row(gmp_day1=1.0, gmp_rs=1000.0))
        assert df["gmp_momentum"].iloc[0] <= 5.0, "Momentum should be clipped at 5"


class TestPrediction:
    def test_prediction_returns_dict(self, make_predict_input, trained_models):
        from model import predict_one
        reg, clf, meta = trained_models
        result = predict_one(make_predict_input(), reg, clf, meta)
        assert isinstance(result, dict)
        assert "predicted_gain_pct" in result
        assert "positive_prob" in result
        assert "signal" in result

    def test_high_gmp_predicts_positive(self, make_predict_input, trained_models):
        from model import predict_one
        reg, clf, meta = trained_models
        result = predict_one(make_predict_input(gmp_rs=120, total_subscription=80), reg, clf, meta)
        assert result["predicted_gain_pct"] > 0, "High GMP should predict positive return"
        assert result["positive_prob"] > 50

    def test_negative_gmp_predicts_loss(self, make_predict_input, trained_models):
        from model import predict_one
        reg, clf, meta = trained_models
        result = predict_one(make_predict_input(gmp_rs=-30, total_subscription=1.5), reg, clf, meta)
        assert result["predicted_gain_pct"] < 0, "Negative GMP should predict loss"

    def test_confidence_band_order(self, make_predict_input, trained_models):
        from model import predict_one
        reg, clf, meta = trained_models
        result = predict_one(make_predict_input(), reg, clf, meta)
        assert result["lower_band"] < result["predicted_gain_pct"] < result["upper_band"]

    def test_prob_in_range(self, make_predict_input, trained_models):
        from model import predict_one
        reg, clf, meta = trained_models
        result = predict_one(make_predict_input(), reg, clf, meta)
        assert 0 <= result["positive_prob"] <= 100

    def test_signal_values(self, make_predict_input, trained_models):
        from model import predict_one
        reg, clf, meta = trained_models
        result = predict_one(make_predict_input(), reg, clf, meta)
        assert result["signal"] in ["STRONG BUY", "BUY", "NEUTRAL", "AVOID"]

    def test_sme_prediction_works(self, make_predict_input, trained_models):
        from model import predict_one
        reg, clf, meta = trained_models
        result = predict_one(make_predict_input(category="SME", issue_price=80, gmp_rs=20), reg, clf, meta)
        assert result["predicted_gain_pct"] is not None


class TestSeedData:
    def test_db_exists(self):
        from model import DB_PATH
        assert DB_PATH.exists(), "Database not seeded. Run python scripts/seed_data.py"

    def test_db_has_rows(self):
        import sqlite3
        from model import DB_PATH
        conn = sqlite3.connect(DB_PATH)
        count = conn.execute("SELECT COUNT(*) FROM ipos").fetchone()[0]
        conn.close()
        assert count >= 100, f"Expected at least 100 IPOs, got {count}"

    def test_required_columns(self):
        import sqlite3
        from model import DB_PATH
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT * FROM ipos LIMIT 5", conn)
        conn.close()
        required = ["gmp_rs", "gmp_pct", "issue_price", "listing_gain_pct",
                    "total_subscription", "qib_subscription", "category", "sector"]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_listing_gain_distribution(self):
        import sqlite3
        from model import DB_PATH
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT listing_gain_pct FROM ipos", conn)
        conn.close()
        # Should have both positive and negative listings
        assert (df["listing_gain_pct"] > 0).sum() > 0
        assert (df["listing_gain_pct"] < 0).sum() > 0
        # Mean should be in realistic range
        assert -20 < df["listing_gain_pct"].mean() < 100
