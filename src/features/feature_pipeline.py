"""
feature_pipeline.py — Stock Feature Engineering Pipeline
=========================================================
Place this file at: src/features/feature_pipeline.py

What this file does:
    Reads each stock's raw OHLCV CSV (from data/raw/) →
    computes the same 22 technical features used in trend.py →
    drops NaN warm-up rows → saves clean {SYMBOL}_features.csv
    to data/processed/ for consumption by tuner and backtester.

Feature set (22 features — identical logic to trend.py):
    Group 1  — Returns        : Return_1d/3d/5d/10d
    Group 2  — Candle         : HL_range, Close_vs_Open, Upper_wick, Lower_wick
    Group 3  — EMA Crossovers : EMA_9_21_cross, EMA_21_50_cross
    Group 4  — EMA Distance   : Close_vs_EMA9/21/50
    Group 5  — RSI            : RSI_14
    Group 6  — MACD           : MACD, MACD_sig, MACD_hist
    Group 7  — Bollinger      : BB_width, BB_pct
    Group 8  — ATR            : ATR_pct
    Group 9  — Stochastic     : Stoch_K
    Group 10 — Volatility     : Volatility_10

Design principles:
    - Zero look-ahead bias: every feature uses only past/current data
    - Mirrors trend.py compute_features() exactly — same indicator math
    - NaN rows (indicator warm-up) dropped before saving
    - Processes all stocks in STOCK_UNIVERSE automatically
    - Fails loudly per-stock, continues to next (no silent data loss)

Outputs (per stock):
    data/processed/{SYMBOL}_features.csv

Run:
    python -m src.features.feature_pipeline
    python feature_pipeline.py   (from project root)
"""

import os
import sys
import logging
import warnings

import numpy as np
import pandas as pd

# ── Project root on path ─────────────────────────────────────────
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
    log.propagate = False

# ── Stock universe (same as fetch_stocks.py) ─────────────────────
# Add / remove symbols here as needed.
# Tokens not needed here — we read from already-fetched CSVs.
STOCK_UNIVERSE = [
    "RELIANCE",
    "INFY",
    "TCS",
    "HDFCBANK",
    "ICICIBANK",
    "SBIN",
    "WIPRO",
    "AXISBANK",
    "KOTAKBANK",
]

# ── Explicit feature column list (22 features) ───────────────────
# Must stay in sync with any changes to compute_features() below.
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
    # Group 7: Bollinger Bands (2)
    "BB_width", "BB_pct",
    # Group 8: ATR (1)
    "ATR_pct",
    # Group 9: Stochastic (1)
    "Stoch_K",
    # Group 10: Volatility (1)
    "Volatility_10",
]


# ═══════════════════════════════════════════════════════════════════
# STEP 1 — LOAD STOCK DATA
# ═══════════════════════════════════════════════════════════════════

def load_stock(symbol: str) -> pd.DataFrame:
    """
    Load and validate a stock's raw OHLCV CSV saved by fetcher.py.

    Checks:
        - File exists (raises FileNotFoundError with helpful message)
        - Required OHLC columns present
        - Datetime parses correctly
        - No NaN in OHLC
        - Meets minimum row requirement from CONFIG

    Args:
        symbol : Stock symbol string (e.g. "RELIANCE")

    Returns:
        Clean DataFrame sorted ascending by Datetime.
    """
    path = CONFIG.paths.stock_raw_path(symbol)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n[feature_pipeline] Raw data not found: {path}"
            f"\nRun first: python fetch_stocks.py"
        )

    df = pd.read_csv(path)

    # Normalize column names
    df.columns = [c.strip() for c in df.columns]

    # Handle "datetime" vs "Datetime" spelling difference
    col_lower = {c.lower(): c for c in df.columns}
    if "datetime" in col_lower and col_lower["datetime"] != "Datetime":
        df = df.rename(columns={col_lower["datetime"]: "Datetime"})

    # Parse and sort by Datetime
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.sort_values("Datetime").reset_index(drop=True)

    # Validate required columns
    required = ["Datetime", "Open", "High", "Low", "Close"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[feature_pipeline] {symbol} CSV missing columns: {missing}")

    # Drop NaN OHLC rows
    before = len(df)
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).reset_index(drop=True)
    dropped = before - len(df)
    if dropped > 0:
        log.warning(f"[{symbol}] Dropped {dropped} rows with NaN OHLC")

    # Minimum rows check
    if len(df) < CONFIG.data.min_rows_required:
        raise ValueError(
            f"[feature_pipeline] {symbol} has only {len(df)} rows — "
            f"need {CONFIG.data.min_rows_required}. "
            f"Extend CONFIG.data.start_date or re-fetch."
        )

    log.info(
        f"[{symbol}] Loaded {len(df):,} rows | "
        f"{df['Datetime'].min().date()} → {df['Datetime'].max().date()}"
    )
    return df


# ═══════════════════════════════════════════════════════════════════
# STEP 2 — FEATURE ENGINEERING (mirrors trend.py exactly)
# ═══════════════════════════════════════════════════════════════════

def compute_features(df: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
    """
    Compute 22 technical features from stock OHLCV data.

    ══════════════════════════════════════════════════
    ⚠️  STRICT RULE: NO shift(-1) anywhere here.
        Every feature uses ONLY current or past data.
        This function is identical in logic to
        trend.py's compute_features() — same math,
        applied to individual stocks instead of NIFTY.
    ══════════════════════════════════════════════════

    Args:
        df     : Raw OHLCV DataFrame (sorted ascending)
        symbol : Symbol string for log messages only

    Returns:
        DataFrame with all 22 feature columns appended.
        Intermediate EMA columns dropped (EMA_9/21/50).
    """
    tag = f"[{symbol}]" if symbol else "[compute_features]"
    d   = df.copy()

    # ── Group 1: Returns ────────────────────────────────────────
    # pct_change(n) = (Close_today − Close_n_days_ago) / Close_n_days_ago
    # Uses only past data ✅
    d["Return_1d"]  = d["Close"].pct_change(1)
    d["Return_3d"]  = d["Close"].pct_change(3)
    d["Return_5d"]  = d["Close"].pct_change(5)
    d["Return_10d"] = d["Close"].pct_change(10)

    # ── Group 2: Candle Structure ────────────────────────────────
    # Computed from today's OHLC only — no future data ✅
    d["HL_range"]      = (d["High"] - d["Low"]) / d["Close"]
    d["Close_vs_Open"] = (d["Close"] - d["Open"]) / d["Open"]

    body_top        = d[["Open", "Close"]].max(axis=1)
    body_bot        = d[["Open", "Close"]].min(axis=1)
    d["Upper_wick"] = (d["High"] - body_top) / d["Close"]
    d["Lower_wick"] = (body_bot  - d["Low"])  / d["Close"]

    # ── Group 3 & 4: EMA ────────────────────────────────────────
    # EWM uses adjust=False — standard exponential weighting ✅
    for span in [9, 21, 50]:
        d[f"EMA_{span}"] = d["Close"].ewm(span=span, adjust=False).mean()

    # Binary crossover signals
    d["EMA_9_21_cross"]  = (d["EMA_9"]  > d["EMA_21"]).astype(int)
    d["EMA_21_50_cross"] = (d["EMA_21"] > d["EMA_50"]).astype(int)

    # Price distance from EMA (normalised)
    d["Close_vs_EMA9"]  = (d["Close"] - d["EMA_9"])  / d["EMA_9"]
    d["Close_vs_EMA21"] = (d["Close"] - d["EMA_21"]) / d["EMA_21"]
    d["Close_vs_EMA50"] = (d["Close"] - d["EMA_50"]) / d["EMA_50"]

    # ── Group 5: RSI 14 ──────────────────────────────────────────
    # Uses only past price changes ✅
    delta = d["Close"].diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_g = gain.ewm(com=13, adjust=False).mean()
    avg_l = loss.ewm(com=13, adjust=False).mean()
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
    # Rolling window uses only past + current data ✅
    bb_mid        = d["Close"].rolling(20).mean()
    bb_std        = d["Close"].rolling(20).std()
    bb_upper      = bb_mid + 2 * bb_std
    bb_lower      = bb_mid - 2 * bb_std
    bb_range      = bb_upper - bb_lower
    d["BB_width"] = bb_range / bb_mid
    d["BB_pct"]   = (d["Close"] - bb_lower) / (bb_range + 1e-9)

    # ── Group 8: ATR 14 (normalised) ────────────────────────────
    # shift(1) = previous close — uses past data only ✅
    prev_close = d["Close"].shift(1)
    tr = pd.concat([
        (d["High"] - d["Low"]),
        (d["High"] - prev_close).abs(),
        (d["Low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr14        = tr.ewm(com=13, adjust=False).mean()
    d["ATR_pct"] = atr14 / d["Close"]

    # ── Group 9: Stochastic %K (14-period) ──────────────────────
    # %K = position of close within 14-day High-Low range ✅
    low14        = d["Low"].rolling(14).min()
    high14       = d["High"].rolling(14).max()
    hl_range14   = high14 - low14
    d["Stoch_K"] = (d["Close"] - low14) / (hl_range14 + 1e-9) * 100

    # ── Group 10: Rolling Volatility ────────────────────────────
    # Std of daily returns over 10 days — past data only ✅
    d["Volatility_10"] = d["Return_1d"].rolling(10).std()

    # ── Drop intermediate EMA columns ───────────────────────────
    d = d.drop(columns=["EMA_9", "EMA_21", "EMA_50"], errors="ignore")

    log.info(f"{tag} Features computed — {len(FEATURE_COLS)} feature columns ready")
    return d


# ═══════════════════════════════════════════════════════════════════
# STEP 3 — CLEAN: DROP NaN WARM-UP ROWS
# ═══════════════════════════════════════════════════════════════════

def drop_warmup_nans(df: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
    """
    Drop rows where any feature column is NaN.

    Why NaNs exist:
        Rolling windows (BB=20, Stoch=14, Volatility=10) and
        pct_change (Return_10d) produce NaN for the first N rows
        while indicators "warm up". These rows cannot be used
        for training or backtesting and must be removed.

    Approximately 50 rows dropped (largest window = BB at 20,
    but Return_10d requires 10 rows, Volatility_10 needs 11).

    Args:
        df     : DataFrame after compute_features()
        symbol : Symbol string for log messages

    Returns:
        Clean DataFrame with no NaN in any FEATURE_COLS column.
    """
    tag = f"[{symbol}]" if symbol else "[drop_warmup_nans]"

    # Only drop on feature columns that exist in df
    feat_cols_present = [c for c in FEATURE_COLS if c in df.columns]
    missing_cols = set(FEATURE_COLS) - set(feat_cols_present)
    if missing_cols:
        log.warning(f"{tag} Missing feature columns: {missing_cols}")

    before   = len(df)
    df_clean = df.dropna(subset=feat_cols_present).reset_index(drop=True)
    dropped  = before - len(df_clean)

    log.info(
        f"{tag} Dropped {dropped:,} warm-up NaN rows | "
        f"Clean rows: {len(df_clean):,} | "
        f"Date range: {df_clean['Datetime'].min().date()} → {df_clean['Datetime'].max().date()}"
    )

    # Final sanity check — no NaN should remain
    nan_count = df_clean[feat_cols_present].isnull().sum().sum()
    if nan_count > 0:
        raise ValueError(
            f"{tag} {nan_count} NaN values remain after dropna. "
            f"Check compute_features() for issues."
        )

    return df_clean


# ═══════════════════════════════════════════════════════════════════
# STEP 4 — SAVE FEATURES CSV
# ═══════════════════════════════════════════════════════════════════

def save_features(df: pd.DataFrame, symbol: str):
    """
    Save clean features DataFrame to data/processed/{SYMBOL}_features.csv.

    Columns saved:
        Datetime  : YYYY-MM-DD string (for clean CSV reading)
        Open/High/Low/Close/Volume : original OHLCV retained
        [22 feature columns]       : all computed features

    Args:
        df     : Clean DataFrame from drop_warmup_nans()
        symbol : Stock symbol string
    """
    path = CONFIG.paths.stock_features_path(symbol)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    out = df.copy()
    out["Datetime"] = pd.to_datetime(out["Datetime"]).dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False)

    log.info(f"[{symbol}] ✅ Saved → {path}  ({len(out):,} rows, {len(FEATURE_COLS)} features)")


# ═══════════════════════════════════════════════════════════════════
# STEP 5 — PER-SYMBOL PIPELINE
# ═══════════════════════════════════════════════════════════════════

def process_symbol(symbol: str) -> bool:
    """
    Run the full feature pipeline for a single stock symbol.

    Steps:
        1. Load raw CSV from data/raw/
        2. Compute 22 technical features
        3. Drop NaN warm-up rows
        4. Save clean features CSV to data/processed/

    Args:
        symbol : Stock symbol string (e.g. "RELIANCE")

    Returns:
        True if successful, False if any step failed.
        Failure is logged — pipeline continues to next symbol.
    """
    log.info(f"{'─'*55}")
    log.info(f"  Processing: {symbol}")
    log.info(f"{'─'*55}")

    try:
        # Step 1: Load
        df = load_stock(symbol)

        # Step 2: Features
        df = compute_features(df, symbol=symbol)

        # Step 3: Drop warm-up NaNs
        df = drop_warmup_nans(df, symbol=symbol)

        # Step 4: Save
        save_features(df, symbol)

        return True

    except FileNotFoundError as e:
        log.error(f"[{symbol}] ❌ File not found: {e}")
        log.error(f"[{symbol}]    Run: python fetch_stocks.py first")
        return False

    except ValueError as e:
        log.error(f"[{symbol}] ❌ Validation error: {e}")
        return False

    except Exception as e:
        log.error(f"[{symbol}] ❌ Unexpected error: {e}", exc_info=True)
        return False


# ═══════════════════════════════════════════════════════════════════
# MAIN PIPELINE — processes all stocks
# ═══════════════════════════════════════════════════════════════════

def run_feature_pipeline(symbols: list = None) -> dict:
    """
    Run feature engineering pipeline for all stocks in STOCK_UNIVERSE.

    Called by:
        main.py                                  → as part of full pipeline
        python -m src.features.feature_pipeline  → standalone run

    Args:
        symbols : Optional override list. Defaults to STOCK_UNIVERSE.

    Returns:
        results : dict of {symbol: True/False} — success/failure per stock
    """
    symbols = symbols or STOCK_UNIVERSE

    log.info("=" * 55)
    log.info("  Stock Feature Pipeline — Starting")
    log.info(f"  Symbols : {symbols}")
    log.info(f"  Total   : {len(symbols)}")
    log.info("=" * 55)

    results   = {}
    succeeded = []
    failed    = []

    for symbol in symbols:
        symbol = symbol.upper()
        ok = process_symbol(symbol)
        results[symbol] = ok
        if ok:
            succeeded.append(symbol)
        else:
            failed.append(symbol)

    # ── Summary ─────────────────────────────────────────────────
    log.info("=" * 55)
    log.info("  Stock Feature Pipeline — Complete")
    log.info("=" * 55)
    log.info(f"  ✅ Succeeded ({len(succeeded)}) : {succeeded}")
    if failed:
        log.warning(f"  ❌ Failed    ({len(failed)}) : {failed}")
        log.warning("     Check logs above for per-symbol errors.")
    log.info("=" * 55)

    return results


# ═══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    results = run_feature_pipeline()

    # Print final output paths for succeeded symbols
    print(f"\n{'═'*55}")
    print(f"  Output Files")
    print(f"{'═'*55}")
    for symbol, ok in results.items():
        if ok:
            path = CONFIG.paths.stock_features_path(symbol)
            print(f"  ✅ {path}")
        else:
            print(f"  ❌ {symbol} — failed (see logs above)")
    print(f"{'═'*55}\n")