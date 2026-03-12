"""
trend.py — ML-Based NIFTY Regime Detector
==========================================
Place this file at: src/regime/trend.py

What this file does:
    Reads NIFTY OHLCV CSV → computes 22 technical features →
    trains RF + XGBoost → predicts tomorrow's market regime →
    saves nifty_regime.csv used by ema_strategy.py as trade gate.

Label definition (NO look-ahead bias):
    UP       : next_day_return >  +0.2%  → label = 2
    SIDEWAYS : abs(next_day_return) ≤ 0.2% → label = 1
    DOWN     : next_day_return < -0.2%  → label = 0

    shift(-1) is used ONLY for label creation — never for features.
    This is correct supervised ML, not look-ahead bias.

Outputs:
    data/processed/nifty_regime.csv     ← strategy reads this
    results/reports/nifty_model_report.txt
    results/plots/nifty_regime_plot.png

Run:
    python -m src.regime.trend
"""

import os
import sys
import warnings
import logging

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")   # non-interactive — works in all environments
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb

# ── Project root on path — works regardless of run location ──────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from config import CONFIG

warnings.filterwarnings("ignore")

# ── Logger ───────────────────────────────────────────────────────
log = logging.getLogger(__name__)
if not log.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        fmt     = "%(asctime)s  [%(levelname)s]  %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
    ))
    log.addHandler(_handler)
    log.setLevel(logging.INFO)
    log.propagate = False   # stop root logger printing a second copy

# ═══════════════════════════════════════════════════════════════════
# STEP 1 — LOAD NIFTY DATA
# ═══════════════════════════════════════════════════════════════════

def load_nifty(path: str) -> pd.DataFrame:
    """
    Load and validate NIFTY CSV saved by fetcher.py.

    Checks:
        - File exists
        - Required columns present
        - Datetime parses correctly
        - No NaN in OHLC
        - Meets minimum row requirement from CONFIG
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n[trend.py] NIFTY data not found: {path}"
            f"\nRun first: python -m src.data.fetcher --nifty-only"
        )

    df = pd.read_csv(path)

    # Normalize column names — strip whitespace
    df.columns = [c.strip() for c in df.columns]

    # Handle "datetime" vs "Datetime" spelling difference
    col_lower = {c.lower(): c for c in df.columns}
    if "datetime" in col_lower and col_lower["datetime"] != "Datetime":
        df = df.rename(columns={col_lower["datetime"]: "Datetime"})

    # Parse Datetime
    df["Datetime"] = pd.to_datetime(df["Datetime"])

    # Sort ascending — critical for time-series correctness
    df = df.sort_values("Datetime").reset_index(drop=True)

    # Validate required columns
    required = ["Datetime", "Open", "High", "Low", "Close"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[trend.py] NIFTY CSV missing columns: {missing}")

    # Drop rows where OHLC is NaN
    before = len(df)
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).reset_index(drop=True)
    if len(df) < before:
        log.warning(f"[load_nifty] Dropped {before - len(df)} rows with NaN OHLC")

    # Minimum rows check — need enough for indicator warm-up
    if len(df) < CONFIG.data.min_rows_required:
        raise ValueError(
            f"[trend.py] Only {len(df)} rows — need {CONFIG.data.min_rows_required}.\n"
            f"Set earlier start_date in CONFIG.data.start_date"
        )

    log.info(
        f"[load_nifty] ✅ {len(df):,} rows | "
        f"{df['Datetime'].min().date()} → {df['Datetime'].max().date()}"
    )
    return df


# ═══════════════════════════════════════════════════════════════════
# STEP 2 — FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 22 technical features from NIFTY OHLCV.

    ══════════════════════════════════════════════════
    ⚠️  STRICT RULE: NO shift(-1) in this function.
        Every feature uses ONLY current or past data.
        shift(-1) belongs exclusively in create_labels().
    ══════════════════════════════════════════════════

    Features (22 total):
        Group 1 — Returns        : Return_1d/3d/5d/10d
        Group 2 — Candle         : HL_range, Close_vs_Open, Upper_wick, Lower_wick
        Group 3 — EMA Crossovers : EMA_9_21_cross, EMA_21_50_cross
        Group 4 — EMA Distance   : Close_vs_EMA9/21/50
        Group 5 — RSI            : RSI_14
        Group 6 — MACD           : MACD, MACD_sig, MACD_hist
        Group 7 — Bollinger      : BB_width, BB_pct
        Group 8 — ATR            : ATR_pct
        Group 9 — Stochastic     : Stoch_K  (replaces redundant ROC_10)
        Group 10 — Volatility    : Volatility_10
    """
    d = df.copy()

    # ── Group 1: Returns ────────────────────────────────────────
    # pct_change(n) = (Close_today - Close_n_days_ago) / Close_n_days_ago
    # Uses only past data ✅
    d["Return_1d"]  = d["Close"].pct_change(1)
    d["Return_3d"]  = d["Close"].pct_change(3)
    d["Return_5d"]  = d["Close"].pct_change(5)
    d["Return_10d"] = d["Close"].pct_change(10)

    # ── Group 2: Candle Structure ────────────────────────────────
    # All computed from today's OHLC — no future data ✅
    d["HL_range"]      = (d["High"] - d["Low"]) / d["Close"]
    d["Close_vs_Open"] = (d["Close"] - d["Open"]) / d["Open"]

    body_top        = d[["Open", "Close"]].max(axis=1)
    body_bot        = d[["Open", "Close"]].min(axis=1)
    d["Upper_wick"] = (d["High"] - body_top) / d["Close"]
    d["Lower_wick"] = (body_bot  - d["Low"])  / d["Close"]

    # ── Group 3 & 4: EMA ────────────────────────────────────────
    # EWM uses adjust=False — standard exponential weighting
    # Each EMA value uses only past + current data ✅
    for span in [9, 21, 50]:
        d[f"EMA_{span}"] = d["Close"].ewm(span=span, adjust=False).mean()

    # Binary crossover signals
    d["EMA_9_21_cross"]  = (d["EMA_9"]  > d["EMA_21"]).astype(int)
    d["EMA_21_50_cross"] = (d["EMA_21"] > d["EMA_50"]).astype(int)

    # Price distance from EMA (normalised — comparable across price levels)
    d["Close_vs_EMA9"]  = (d["Close"] - d["EMA_9"])  / d["EMA_9"]
    d["Close_vs_EMA21"] = (d["Close"] - d["EMA_21"]) / d["EMA_21"]
    d["Close_vs_EMA50"] = (d["Close"] - d["EMA_50"]) / d["EMA_50"]

    # ── Group 5: RSI 14 ──────────────────────────────────────────
    # Uses only past price changes ✅
    delta = d["Close"].diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    # EWM with com=13 is equivalent to span=14 for RSI calculation
    avg_g = gain.ewm(com=13, adjust=False).mean()
    avg_l = loss.ewm(com=13, adjust=False).mean()
    # Replace 0 with NaN to avoid division by zero
    rs    = avg_g / avg_l.replace(0, np.nan)
    d["RSI_14"] = 100 - (100 / (1 + rs))

    # ── Group 6: MACD (12, 26, 9) ───────────────────────────────
    # Uses only past data via EWM ✅
    ema12          = d["Close"].ewm(span=12, adjust=False).mean()
    ema26          = d["Close"].ewm(span=26, adjust=False).mean()
    d["MACD"]      = ema12 - ema26
    d["MACD_sig"]  = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_hist"] = d["MACD"] - d["MACD_sig"]

    # ── Group 7: Bollinger Bands (20, 2) ────────────────────────
    # Rolling mean and std use only past + current data ✅
    bb_mid        = d["Close"].rolling(20).mean()
    bb_std        = d["Close"].rolling(20).std()
    bb_upper      = bb_mid + 2 * bb_std
    bb_lower      = bb_mid - 2 * bb_std
    bb_range      = bb_upper - bb_lower
    d["BB_width"] = bb_range / bb_mid
    # 1e-9 guard prevents division by zero when bands are extremely tight
    d["BB_pct"]   = (d["Close"] - bb_lower) / (bb_range + 1e-9)

    # ── Group 8: ATR 14 (normalised) ────────────────────────────
    # shift(1) = previous close — this is correct, uses past data ✅
    prev_close = d["Close"].shift(1)
    tr = pd.concat([
        (d["High"] - d["Low"]),                  # intraday range
        (d["High"] - prev_close).abs(),           # gap up + move
        (d["Low"]  - prev_close).abs(),           # gap down + move
    ], axis=1).max(axis=1)
    atr14          = tr.ewm(com=13, adjust=False).mean()
    # Normalise by price — makes ATR comparable across different price levels
    d["ATR_pct"]   = atr14 / d["Close"]

    # ── Group 9: Stochastic %K (14-period) ──────────────────────
    # Replaces ROC_10 which was redundant with Return_10d
    # %K = where is today's close within the 14-day High-Low range
    # Uses only rolling past data ✅
    low14          = d["Low"].rolling(14).min()
    high14         = d["High"].rolling(14).max()
    hl_range14     = high14 - low14
    # 1e-9 guard for flat market periods where high == low
    d["Stoch_K"]   = (d["Close"] - low14) / (hl_range14 + 1e-9) * 100

    # ── Group 10: Rolling Volatility ────────────────────────────
    # Std of daily returns over 10 days — uses only past returns ✅
    d["Volatility_10"] = d["Return_1d"].rolling(10).std()

    # ── Drop intermediate EMA columns not needed as features ────
    # Keep EMA values for crossover + distance calculation only
    # They are NOT fed directly into the model
    d = d.drop(columns=["EMA_9", "EMA_21", "EMA_50"], errors="ignore")

    return d


# ═══════════════════════════════════════════════════════════════════
# STEP 3 — LABEL CREATION
# ═══════════════════════════════════════════════════════════════════

def create_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create next-day regime label for supervised ML training.

    ══════════════════════════════════════════════════════════
    WHY shift(-1) IS CORRECT HERE (not look-ahead bias):

        Features at day t  = computed from data UP TO day t
        Label at day t     = what happened at day t+1

        The model learns: "given today's signals, predict tomorrow"
        This is standard supervised ML — identical to how weather
        forecasting, credit scoring, and all production ML works.

        Look-ahead bias would be:
            Using tomorrow's price to compute today's FEATURES ❌
            We NEVER do that. Features are strictly past data.
    ══════════════════════════════════════════════════════════

    Encoding:
        UP       = 2  (next_day_return >  +threshold)
        SIDEWAYS = 1  (abs(next_day_return) <= threshold)
        DOWN     = 0  (next_day_return <  -threshold)

    The ±0.2% threshold filters noise — random tiny moves
    become SIDEWAYS rather than polluting UP/DOWN labels.

    Last row always dropped — no t+1 data exists for it.
    """
    d         = df.copy()
    threshold = CONFIG.regime.label_threshold   # 0.002 = ±0.2%

    # Next day return — shift(-1) for LABELS ONLY
    # This is the "answer key" the model learns to predict
    next_return = d["Close"].pct_change().shift(-1)

    d["label"] = np.select(
        condlist  = [next_return >  threshold,    # UP   = 2
                     next_return < -threshold],   # DOWN = 0
        choicelist= [2, 0],
        default   = 1,                            # SIDEWAYS = 1
    )

    # Drop last row — shift(-1) makes it NaN, no valid label
    d = d.iloc[:-1].copy().reset_index(drop=True)

    # Log distribution — helps diagnose class imbalance
    dist = d["label"].value_counts().sort_index()
    total = len(d)
    log.info(
        f"[create_labels] Threshold ±{threshold*100:.1f}% | "
        f"DOWN={dist.get(0,0):,} ({dist.get(0,0)/total*100:.1f}%)  "
        f"SIDEWAYS={dist.get(1,0):,} ({dist.get(1,0)/total*100:.1f}%)  "
        f"UP={dist.get(2,0):,} ({dist.get(2,0)/total*100:.1f}%)"
    )
    return d


# ═══════════════════════════════════════════════════════════════════
# STEP 4 — FEATURE LIST & PREPARE X, y
# ═══════════════════════════════════════════════════════════════════

# Explicit feature list — exactly 22 features
# Any change here must be intentional and documented
FEATURE_COLS = [
    # Group 1: Returns (4)
    "Return_1d", "Return_3d", "Return_5d", "Return_10d",
    # Group 2: Candle structure (4)
    "HL_range", "Close_vs_Open", "Upper_wick", "Lower_wick",
    # Group 3: EMA crossovers (2)
    "EMA_9_21_cross", "EMA_21_50_cross",
    # Group 4: EMA distance (3)
    "Close_vs_EMA9", "Close_vs_EMA21", "Close_vs_EMA50",
    # Group 5: RSI (1)
    "RSI_14",
    # Group 6: MACD (3)
    "MACD", "MACD_sig", "MACD_hist",
    # Group 7: Bollinger (2)
    "BB_width", "BB_pct",
    # Group 8: ATR (1)
    "ATR_pct",
    # Group 9: Stochastic (1)
    "Stoch_K",
    # Group 10: Volatility (1)
    "Volatility_10",
]


def prepare_Xy(df: pd.DataFrame):
    """
    Package features and labels for ML training.

    Actions:
        1. Select only FEATURE_COLS that exist in df
           (guards against any compute_features() failures)
        2. Retain Datetime + Close for output CSV alignment
        3. Drop NaN rows from indicator warm-up period
           (first ~50 rows will have NaN from rolling windows)

    Returns:
        X        : feature DataFrame  (n_samples × 22)
        y        : label Series       (n_samples,)
        df_clean : full DataFrame with Datetime retained
                   used to align predictions with dates
    """
    # Only use feature cols that actually exist in df
    # (defensive — avoids KeyError if a feature failed to compute)
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]

    missing_feats = set(FEATURE_COLS) - set(feat_cols)
    if missing_feats:
        log.warning(f"[prepare_Xy] Missing features (will be skipped): {missing_feats}")

    # Retain Datetime and Close for output alignment
    keep = feat_cols + ["label", "Datetime", "Close"]
    keep = [c for c in keep if c in df.columns]

    # Drop rows with any NaN — indicator warm-up period
    before = len(df)
    d      = df[keep].dropna().reset_index(drop=True)
    dropped = before - len(d)

    if dropped > 0:
        log.info(f"[prepare_Xy] Dropped {dropped:,} NaN rows (indicator warm-up)")

    X = d[feat_cols].copy()
    y = d["label"].copy()

    # Sanity check — X should never contain NaN after dropna
    nan_count = X.isnull().sum().sum()
    if nan_count > 0:
        raise ValueError(
            f"[prepare_Xy] X still contains {nan_count} NaN values after dropna. "
            f"Check compute_features() for issues."
        )

    log.info(
        f"[prepare_Xy] ✅ {len(d):,} samples | "
        f"{X.shape[1]} features | "
        f"Date range: {d['Datetime'].min().date()} → {d['Datetime'].max().date()}"
    )
    return X, y, d


# ═══════════════════════════════════════════════════════════════════
# STEP 5 — WALK-FORWARD CROSS VALIDATION
# ═══════════════════════════════════════════════════════════════════

def cross_validate(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    Walk-forward TimeSeriesSplit cross-validation.

    Why TimeSeriesSplit (not random KFold):
        Random KFold would allow training on 2020 data to predict 2018.
        TimeSeriesSplit always trains on past, validates on future.
        This is the only honest way to validate a time-series model.

    Fold structure (n_splits=5):
        Fold 1: Train 2015-2017 → Validate 2018
        Fold 2: Train 2015-2018 → Validate 2019
        Fold 3: Train 2015-2019 → Validate 2020
        Fold 4: Train 2015-2020 → Validate 2021
        Fold 5: Train 2015-2021 → Validate 2022-2025

    Metric: Macro F1
        Treats UP, DOWN, SIDEWAYS equally.
        A model that only predicts the majority class gets penalized.

    Returns:
        avg_rf_f1  : mean macro F1 across folds — Random Forest
        avg_xgb_f1 : mean macro F1 across folds — XGBoost
    """
    n_splits = CONFIG.regime.n_cv_splits
    tscv     = TimeSeriesSplit(n_splits=n_splits)
    rf_f1s   = []
    xgb_f1s  = []

    log.info(f"[cross_validate] Walk-forward CV | {n_splits} folds...")

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_tr,  X_val = X.iloc[tr_idx],  X.iloc[val_idx]
        y_tr,  y_val = y.iloc[tr_idx],  y.iloc[val_idx]

        # Validate fold sizes are reasonable
        if len(X_tr) < 100 or len(X_val) < 20:
            log.warning(f"  Fold {fold}: small split — Train={len(X_tr)}, Val={len(X_val)}")

        # Train both models fresh each fold
        rf  = RandomForestClassifier(**CONFIG.regime.rf_params)
        xgm = xgb.XGBClassifier(**CONFIG.regime.xgb_params)

        rf.fit(X_tr, y_tr)
        xgm.fit(X_tr, y_tr)

        # Score on held-out validation fold
        rf_f1  = f1_score(y_val, rf.predict(X_val),  average="macro", zero_division=0)
        xgb_f1 = f1_score(y_val, xgm.predict(X_val), average="macro", zero_division=0)

        rf_f1s.append(rf_f1)
        xgb_f1s.append(xgb_f1)

        log.info(
            f"  Fold {fold}/{n_splits} | "
            f"Train={len(tr_idx):,}  Val={len(val_idx):,} | "
            f"RF F1={rf_f1:.4f}  XGB F1={xgb_f1:.4f}"
        )

    avg_rf  = float(np.mean(rf_f1s))
    avg_xgb = float(np.mean(xgb_f1s))
    winner  = "RF" if avg_rf >= avg_xgb else "XGB"

    log.info("─" * 55)
    log.info(f"  RF  avg macro F1 : {avg_rf:.4f}")
    log.info(f"  XGB avg macro F1 : {avg_xgb:.4f}")
    log.info(f"  Winner           : {winner}")
    log.info("─" * 55)

    return avg_rf, avg_xgb


# ═══════════════════════════════════════════════════════════════════
# STEP 6 — FINAL TRAINING ON FULL DATASET
# ═══════════════════════════════════════════════════════════════════

def train_final(X: pd.DataFrame, y: pd.Series, model_name: str):
    """
    Train the winning model on the complete dataset.

    Why train on full data after CV:
        CV was used to COMPARE models honestly.
        Now we train on ALL available data to get the most
        informed model for generating predictions.
        More data = better learned patterns.

    Args:
        X          : full feature matrix
        y          : full label vector
        model_name : "RF" or "XGB" (winner from cross_validate)

    Returns trained model object.
    """
    log.info(f"[train_final] Training final {model_name} on {len(X):,} samples...")

    if model_name == "RF":
        model = RandomForestClassifier(**CONFIG.regime.rf_params)
    else:
        model = xgb.XGBClassifier(**CONFIG.regime.xgb_params)

    model.fit(X, y)
    log.info(f"[train_final] ✅ Training complete")
    return model


# ═══════════════════════════════════════════════════════════════════
# STEP 7 — GENERATE REGIME PREDICTIONS
# ═══════════════════════════════════════════════════════════════════

def generate_predictions(
    model,
    X        : pd.DataFrame,
    df_clean : pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate predicted regime + probabilities for all historical dates.

    Output columns:
        Datetime         : date (YYYY-MM-DD)
        Close            : NIFTY close price that day
        predicted_regime : "UP" / "SIDEWAYS" / "DOWN"
        prob_DOWN        : model's confidence for DOWN (0-1)
        prob_SIDEWAYS    : model's confidence for SIDEWAYS (0-1)
        prob_UP          : model's confidence for UP (0-1)

    CRITICAL — how strategy uses this:
        ema_strategy.py merges this CSV with stock data on Datetime.
        For each day:
            predicted_regime == "UP"  → check EMA crossover → enter if signal
            predicted_regime == "SIDEWAYS" → no trade
            predicted_regime == "DOWN"     → no trade (long-only system)

    Probability columns (prob_UP etc.) are saved for future use:
        - Confidence filtering: only trade when prob_UP > 0.6
        - Position sizing: larger size when model is more confident
        - These are not used in v1 but available for enhancement
    """
    label_map = CONFIG.regime.regime_labels   # {0:"DOWN", 1:"SIDEWAYS", 2:"UP"}

    preds = model.predict(X)
    proba = model.predict_proba(X)   # shape: (n_samples, 3)
    # Column order from sklearn: sorted by class label
    # class 0=DOWN → col 0, class 1=SIDEWAYS → col 1, class 2=UP → col 2

    regime_df = pd.DataFrame({
        "Datetime"         : df_clean["Datetime"].values,
        "Close"            : df_clean["Close"].values,
        "predicted_regime" : [label_map[int(p)] for p in preds],
        "prob_DOWN"        : proba[:, 0].round(4),
        "prob_SIDEWAYS"    : proba[:, 1].round(4),
        "prob_UP"          : proba[:, 2].round(4),
    })

    # Verify probabilities sum to ~1.0 (sanity check)
    prob_sum = regime_df[["prob_DOWN", "prob_SIDEWAYS", "prob_UP"]].sum(axis=1)
    if not np.allclose(prob_sum, 1.0, atol=0.01):
        log.warning("[generate_predictions] Some probability rows don't sum to 1.0")

    # Log regime distribution
    dist  = regime_df["predicted_regime"].value_counts()
    total = len(regime_df)
    log.info("[generate_predictions] Predicted regime distribution:")
    for r in ["UP", "SIDEWAYS", "DOWN"]:
        n   = dist.get(r, 0)
        pct = n / total * 100
        log.info(f"  {r:<10} : {n:>5,}  ({pct:.1f}%)")

    return regime_df


# ═══════════════════════════════════════════════════════════════════
# STEP 8 — SAVE ALL OUTPUTS
# ═══════════════════════════════════════════════════════════════════

def save_regime_csv(regime_df: pd.DataFrame):
    """
    Save nifty_regime.csv — primary output consumed by ema_strategy.py.
    Datetime saved as YYYY-MM-DD string for clean CSV reading.
    """
    path = CONFIG.paths.nifty_regime_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)

    out = regime_df.copy()
    out["Datetime"] = pd.to_datetime(out["Datetime"]).dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False)

    log.info(f"[save] ✅ Regime CSV  → {path}  ({len(out):,} rows)")


def save_model_report(
    model      ,
    X          : pd.DataFrame,
    y          : pd.Series,
    model_name : str,
    avg_rf     : float,
    avg_xgb    : float,
):
    """
    Save model performance report.
    Includes CV scores (honest) + in-sample report (fit quality only).
    """
    path = CONFIG.paths.model_report_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # In-sample report — shows how well model fits training data
    # Do NOT use this to judge live performance — use CV F1 for that
    preds  = model.predict(X)
    report = classification_report(
        y, preds,
        target_names = ["DOWN", "SIDEWAYS", "UP"],
        zero_division = 0,
    )

    # Feature importance (top 10)
    if hasattr(model, "feature_importances_"):
        feat_names   = X.columns.tolist()
        importances  = model.feature_importances_
        feat_imp     = sorted(
            zip(feat_names, importances),
            key=lambda x: x[1], reverse=True
        )[:10]
        imp_lines = ["", "  Top 10 Feature Importances:"]
        for fname, imp in feat_imp:
            bar = "█" * int(imp * 200)
            imp_lines.append(f"  {fname:<20} {imp:.4f}  {bar}")
    else:
        imp_lines = []

    content = "\n".join([
        "=" * 62,
        "  NIFTY REGIME PREDICTION — MODEL REPORT",
        "=" * 62,
        f"  Best model selected  : {model_name}",
        f"  Random Forest CV F1  : {avg_rf:.4f}",
        f"  XGBoost CV F1        : {avg_xgb:.4f}",
        f"  Label threshold      : ±{CONFIG.regime.label_threshold*100:.1f}%",
        f"  CV folds             : {CONFIG.regime.n_cv_splits}",
        f"  Training samples     : {len(X):,}",
        "",
        "  ⚠️  Use CV F1 scores to judge model quality.",
        "     In-sample report below shows FIT QUALITY only.",
        "",
        "  In-Sample Classification Report:",
        report,
    ] + imp_lines + [
        "",
        "=" * 62,
    ])

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    print("\n" + content)
    log.info(f"[save] ✅ Model report → {path}")


def save_regime_plot(regime_df: pd.DataFrame):
    """
    Two-panel chart:
        Top    : NIFTY close price with dots coloured by predicted regime
        Bottom : Regime bar (UP=+1 green, SIDEWAYS=0 gray, DOWN=-1 red)
    """
    path = CONFIG.paths.regime_plot_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)

    color_map = {
        "UP"      : "#27ae60",   # green
        "DOWN"    : "#e74c3c",   # red
        "SIDEWAYS": "#95a5a6",   # gray
    }

    plot_df = regime_df.copy()
    plot_df["Datetime"] = pd.to_datetime(plot_df["Datetime"])

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize           = (18, 10),
        gridspec_kw       = {"height_ratios": [3, 1]},
        sharex            = True,
    )
    fig.patch.set_facecolor("#f8f9fa")

    # ── Top panel: price line + regime dots ─────────────────────
    ax1.set_facecolor("#f8f9fa")
    ax1.plot(
        plot_df["Datetime"], plot_df["Close"],
        color="#2c3e50", linewidth=0.8, alpha=0.4, zorder=1,
    )
    for regime, grp in plot_df.groupby("predicted_regime"):
        ax1.scatter(
            grp["Datetime"], grp["Close"],
            c     = color_map[regime],
            label = regime,
            s     = 8,
            alpha = 0.85,
            zorder= 3,
        )
    ax1.set_title(
        "NIFTY 50 — ML Predicted Regime  (UP = Green | SIDEWAYS = Gray | DOWN = Red)",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax1.set_ylabel("NIFTY Close", fontsize=11)
    ax1.legend(markerscale=4, loc="upper left", framealpha=0.9)
    ax1.grid(True, alpha=0.25, linestyle="--")
    ax1.spines[["top", "right"]].set_visible(False)

    # ── Bottom panel: regime bar ─────────────────────────────────
    ax2.set_facecolor("#f8f9fa")
    regime_num = plot_df["predicted_regime"].map({"DOWN": -1, "SIDEWAYS": 0, "UP": 1})
    bar_colors = plot_df["predicted_regime"].map(color_map)
    ax2.bar(
        plot_df["Datetime"], regime_num,
        color = bar_colors,
        alpha = 0.75,
        width = 1.5,
    )
    ax2.axhline(0, color="#2c3e50", linewidth=0.8, linestyle="--")
    ax2.set_ylabel("Regime", fontsize=10)
    ax2.set_yticks([-1, 0, 1])
    ax2.set_yticklabels(["DOWN", "FLAT", "UP"], fontsize=9)
    ax2.set_ylim(-1.5, 1.5)
    ax2.grid(True, alpha=0.2, linestyle="--")
    ax2.spines[["top", "right"]].set_visible(False)

    # ── X-axis formatting ────────────────────────────────────────
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=40, ha="right", fontsize=9)

    plt.tight_layout(h_pad=0.5)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    log.info(f"[save] ✅ Regime plot  → {path}")


# ═══════════════════════════════════════════════════════════════════
# MAIN PIPELINE — single entry point
# ═══════════════════════════════════════════════════════════════════

def run_trend_detection() -> pd.DataFrame:
    """
    Full NIFTY regime detection pipeline.

    Called by:
        main.py                → as part of full pipeline
        python -m src.regime.trend  → standalone run

    Returns:
        regime_df : DataFrame with Datetime + predicted_regime
                    This is consumed by ema_strategy.py
    """
    log.info("=" * 62)
    log.info("  NIFTY Regime Detection — Pipeline Starting")
    log.info("=" * 62)

    # Step 1: Load
    df = load_nifty(CONFIG.paths.nifty_raw_path())

    # Step 2: Features — past data only
    log.info("[Pipeline] Step 2/8 — Computing features...")
    df = compute_features(df)

    # Step 3: Labels — shift(-1) for labels only
    log.info("[Pipeline] Step 3/8 — Creating labels...")
    df = create_labels(df)

    # Step 4: Prepare X, y
    log.info("[Pipeline] Step 4/8 — Preparing X, y...")
    X, y, df_clean = prepare_Xy(df)

    # Step 5: Cross-validation
    log.info("[Pipeline] Step 5/8 — Walk-forward cross-validation...")
    avg_rf, avg_xgb = cross_validate(X, y)

    # Step 6: Select best model
    best_name = "RF" if avg_rf >= avg_xgb else "XGB"
    log.info(f"[Pipeline] Step 6/8 — Best model: {best_name}")

    # Step 7: Final training on full data
    log.info("[Pipeline] Step 7/8 — Training final model...")
    final_model = train_final(X, y, best_name)

    # Step 8: Generate predictions for all dates
    log.info("[Pipeline] Step 8/8 — Generating predictions...")
    regime_df = generate_predictions(final_model, X, df_clean)

    # Save all outputs
    save_regime_csv(regime_df)
    save_model_report(final_model, X, y, best_name, avg_rf, avg_xgb)
    save_regime_plot(regime_df)

    log.info("=" * 62)
    log.info("  NIFTY Regime Detection — Pipeline Complete ✅")
    log.info("=" * 62)

    return regime_df


# ═══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    regime_df = run_trend_detection()

    print(f"\n{'═'*62}")
    print(f"  Sample Output — Last 10 Rows")
    print(f"{'═'*62}")
    print(regime_df[["Datetime","Close","predicted_regime",
                      "prob_DOWN","prob_SIDEWAYS","prob_UP"]].tail(10).to_string(index=False))

    print(f"\n  Files saved:")
    print(f"  ✅ {CONFIG.paths.nifty_regime_path()}")
    print(f"  ✅ {CONFIG.paths.model_report_path()}")
    print(f"  ✅ {CONFIG.paths.regime_plot_path()}")
    print(f"{'═'*62}\n")