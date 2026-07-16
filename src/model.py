"""
IPO GMP Predictor — ML Model
Trains an XGBoost model to predict listing day return (%) from pre-listing signals.

Trains and evaluates against scripts/seed_data.py's synthetic dataset, where
listing_gain_pct is generated as a noisy function of gmp_pct — so the metrics
here measure how well the model recovers that known synthetic relationship,
not real-world predictive accuracy. See README's "Data" section before
quoting these numbers as validated performance.

Features used:
  - GMP (₹ and %)
  - Subscription rates (QIB, NII, Retail, Total)
  - Issue price, lot size, issue size
  - Category (Mainboard/SME)
  - Sector
  - Days from close to listing
  - GMP momentum (day1 → day2 → final)

Outputs:
  - Predicted listing gain %
  - Confidence band (±1 std of residuals on training set)
  - Direction probability (prob of positive listing)
  - Feature importances
  - Backtest metrics (accuracy, MAE, directional accuracy)
"""

import json
import pickle
import sqlite3
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore")

DB_PATH   = Path(__file__).resolve().parent.parent / "data" / "ipo_gmp.db"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "gmp_pct",
    "gmp_rs",
    "gmp_momentum",        # (gmp_final - gmp_day1) / gmp_day1
    "total_subscription",
    "qib_subscription",
    "nii_subscription",
    "retail_subscription",
    "sub_qib_nii_ratio",   # QIB / NII — smart vs retail money
    "issue_price",
    "issue_price_log",     # log-transform to handle wide range
    "issue_size_cr_log",
    "lot_size",
    "days_to_listing",
    "is_sme",
    "sector_encoded",
]

SECTORS = [
    "Technology", "Financial Services", "Healthcare", "Consumer Goods",
    "Infrastructure", "Manufacturing", "Real Estate", "Chemicals",
    "Auto & Auto Components", "Retail", "Telecom", "Energy",
    "Agriculture", "Media & Entertainment", "Defence",
]
SECTOR_MAP = {s: i for i, s in enumerate(SECTORS)}


def load_data() -> pd.DataFrame:
    """Load all IPOs with a known listing price from the seeded SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM ipos WHERE listing_price IS NOT NULL ORDER BY listing_date", conn)
    conn.close()
    df["listing_date"] = pd.to_datetime(df["listing_date"])
    df["open_date"]    = pd.to_datetime(df["open_date"])
    df["close_date"]   = pd.to_datetime(df["close_date"])
    df["allotment_date"] = pd.to_datetime(df["allotment_date"])
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive GMP momentum, subscription ratios, log-scaled size features, and
    categorical encodings on top of the raw IPO columns. Returns a copy."""
    df = df.copy()

    # GMP momentum: how much GMP moved from day1 to final
    df["gmp_momentum"] = np.where(
        df["gmp_day1"].abs() > 0.1,
        (df["gmp_rs"] - df["gmp_day1"]) / df["gmp_day1"].abs(),
        0.0,
    )
    df["gmp_momentum"] = df["gmp_momentum"].clip(-5, 5)

    # Subscription ratios
    df["sub_qib_nii_ratio"] = (df["qib_subscription"] + 0.01) / (df["nii_subscription"] + 0.01)

    # Log transforms
    df["issue_price_log"]  = np.log1p(df["issue_price"])
    df["issue_size_cr_log"] = np.log1p(df["issue_size_cr"])

    # Days from close to listing
    df["listing_date"] = pd.to_datetime(df["listing_date"], errors="coerce")
    df["close_date"]   = pd.to_datetime(df["close_date"], errors="coerce")
    df["days_to_listing"] = (df["listing_date"] - df["close_date"]).dt.days.fillna(12).clip(0, 30)

    # Categoricals
    df["is_sme"] = (df["category"] == "SME").astype(int)
    df["sector_encoded"] = df["sector"].map(SECTOR_MAP).fillna(0).astype(int)

    return df


def build_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Engineer features and split into (X, y_regression, y_classification, feature_df)."""
    df = engineer_features(df)
    X = df[FEATURE_COLS].fillna(0)
    y_reg  = df["listing_gain_pct"]
    y_cls  = (df["listing_gain_pct"] > 0).astype(int)
    return X, y_reg, y_cls, df


def train(n_splits: int = 5) -> dict[str, Any]:
    """Train the XGBoost regressor + classifier with time-series CV, then fit
    final models on all data and persist them (+ metrics) to MODEL_DIR.

    Returns:
        The metadata dict written to models/meta.json (CV metrics, residual
        std, feature importances, backtest results).
    """
    print("Loading data…")
    df = load_data()
    X, y_reg, y_cls, df_feat = build_dataset(df)
    n = len(X)
    print(f"  {n} IPOs loaded, {X.shape[1]} features")

    # ── Time-series cross-validation ─────────────────────────────
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=max(20, n // 10))

    reg_maes, reg_r2s, dir_accs = [], [], []

    # Regressor
    regressor = XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )

    # Classifier (for direction probability)
    classifier = XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
    )

    print("Running time-series CV…")
    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X)):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y_reg.iloc[tr_idx], y_reg.iloc[te_idx]

        regressor.fit(X_tr, y_tr)
        preds = regressor.predict(X_te)

        mae = mean_absolute_error(y_te, preds)
        r2  = r2_score(y_te, preds)
        dir_acc = np.mean((preds > 0) == (y_te > 0))

        reg_maes.append(mae)
        reg_r2s.append(r2)
        dir_accs.append(dir_acc)
        print(f"  Fold {fold+1}: MAE={mae:.1f}%  R²={r2:.3f}  DirAcc={dir_acc:.1%}")

    print(f"\nCV averages → MAE: {np.mean(reg_maes):.1f}%  R²: {np.mean(reg_r2s):.3f}  DirAcc: {np.mean(dir_accs):.1%}")

    # ── Final fit on all data ─────────────────────────────────────
    print("\nFitting final models on full data…")
    regressor.fit(X, y_reg)
    classifier.fit(X, y_cls)

    # Residuals for confidence bands
    residuals = y_reg.values - regressor.predict(X)
    residual_std = float(np.std(residuals))

    # Feature importances
    importances = dict(zip(FEATURE_COLS, regressor.feature_importances_.tolist()))

    # ── Backtest: rolling 3-month out-of-sample ───────────────────
    df_feat2 = df_feat.copy()
    df_feat2["pred_gain"] = regressor.predict(X)
    df_feat2["pred_dir"]  = classifier.predict_proba(X)[:, 1]
    df_feat2["actual_gain"] = y_reg.values

    backtest = _backtest(df_feat2)

    # ── Save everything ───────────────────────────────────────────
    with open(MODEL_DIR / "regressor.pkl", "wb") as f:
        pickle.dump(regressor, f)
    with open(MODEL_DIR / "classifier.pkl", "wb") as f:
        pickle.dump(classifier, f)

    meta = {
        "cv_mae": round(float(np.mean(reg_maes)), 2),
        "cv_r2": round(float(np.mean(reg_r2s)), 4),
        "cv_dir_acc": round(float(np.mean(dir_accs)), 4),
        "residual_std": round(residual_std, 2),
        "n_train": n,
        "feature_cols": FEATURE_COLS,
        "importances": importances,
        "backtest": backtest,
    }
    with open(MODEL_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ Models saved to {MODEL_DIR}")
    print(f"   Residual std (confidence band): ±{residual_std:.1f}%")
    return meta


def _backtest(df: pd.DataFrame) -> dict:
    """
    Simple backtest: for each month, predict direction on that month's IPOs
    using only prior data. Track P&L of: 'apply for every IPO with pred_gain > 10%'.
    """
    df = df.sort_values("listing_date").copy()
    df["month"] = df["listing_date"].dt.to_period("M")
    months = sorted(df["month"].unique())

    # Only backtest on last 40% of data (enough training data exists)
    cutoff = months[int(len(months) * 0.6)]
    test_months = [m for m in months if m >= cutoff]

    results = []
    for m in test_months:
        month_ipos = df[df["month"] == m]
        # "apply" signal: predicted gain > 10% and direction prob > 0.6
        signal = (month_ipos["pred_gain"] > 10) & (month_ipos["pred_dir"] > 0.60)
        applied = month_ipos[signal]
        all_ipos = month_ipos

        if len(applied) > 0:
            avg_return_signal = applied["actual_gain"].mean()
        else:
            avg_return_signal = 0.0

        results.append({
            "month": str(m),
            "total_ipos": len(all_ipos),
            "signal_ipos": len(applied),
            "avg_return_all": round(all_ipos["actual_gain"].mean(), 2),
            "avg_return_signal": round(avg_return_signal, 2),
        })

    df_bt = pd.DataFrame(results)
    signal_ipos = df_bt[df_bt["signal_ipos"] > 0]

    return {
        "months_tested": len(results),
        "avg_return_all_ipos": round(df_bt["avg_return_all"].mean(), 2),
        "avg_return_signal_ipos": round(signal_ipos["avg_return_signal"].mean(), 2) if len(signal_ipos) else 0.0,
        "monthly": results,
    }


def predict_one(
    input_dict: dict[str, Any],
    regressor: XGBRegressor | None = None,
    classifier: XGBClassifier | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Predict listing gain for a single new IPO.

    Args:
        input_dict: Must have gmp_rs, issue_price, total_subscription,
            qib_subscription, nii_subscription, retail_subscription,
            issue_size_cr, lot_size, category, sector, close_date,
            listing_date. gmp_day1/gmp_day2 are optional (derived from
            gmp_rs if absent).
        regressor: Trained XGBRegressor, or None to load from MODEL_DIR.
        classifier: Trained XGBClassifier, or None to load from MODEL_DIR.
        meta: Training metadata (for residual_std), or None to load from MODEL_DIR.

    Returns:
        predicted_gain_pct, lower_band, upper_band, positive_prob, signal.

    Raises:
        FileNotFoundError: if regressor/classifier/meta are None and no
            trained model exists yet at MODEL_DIR (run `python src/model.py` first).
    """
    if regressor is None:
        with open(MODEL_DIR / "regressor.pkl", "rb") as f:
            regressor = pickle.load(f)
    if classifier is None:
        with open(MODEL_DIR / "classifier.pkl", "rb") as f:
            classifier = pickle.load(f)
    if meta is None:
        with open(MODEL_DIR / "meta.json") as f:
            meta = json.load(f)

    row = pd.DataFrame([input_dict])
    row["gmp_pct"] = row["gmp_rs"] / row["issue_price"] * 100
    row["gmp_day1"] = row.get("gmp_day1", row["gmp_rs"] * 0.5)
    row["gmp_day2"] = row.get("gmp_day2", row["gmp_rs"] * 0.8)
    row = engineer_features(row)

    X = row[FEATURE_COLS].fillna(0)
    pred_gain = float(regressor.predict(X)[0])
    pred_prob = float(classifier.predict_proba(X)[0, 1])
    std = meta["residual_std"]

    return {
        "predicted_gain_pct": round(pred_gain, 2),
        "lower_band": round(pred_gain - std, 2),
        "upper_band": round(pred_gain + std, 2),
        "positive_prob": round(pred_prob * 100, 1),
        "signal": "STRONG BUY" if pred_gain > 20 and pred_prob > 0.75
                  else "BUY" if pred_gain > 10 and pred_prob > 0.60
                  else "NEUTRAL" if pred_gain > -5
                  else "AVOID",
    }


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")  # avoid UnicodeEncodeError on Windows' default cp1252 console
    meta = train()
    print("\nFeature importances:")
    for feat, imp in sorted(meta["importances"].items(), key=lambda x: -x[1]):
        bar = "█" * int(imp * 40)
        print(f"  {feat:25s} {bar} {imp:.3f}")
    print(f"\nBacktest: signal avg return {meta['backtest']['avg_return_signal_ipos']:.1f}% "
          f"vs all-IPO avg {meta['backtest']['avg_return_all_ipos']:.1f}%")
