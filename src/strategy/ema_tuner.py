"""
ema_tuner.py — EMA Crossover Strategy Hyperparameter Tuner
===========================================================
Place this file at: src/strategy/ema_tuner.py

What this file does:
    Reads each stock's features CSV (from data/processed/) →
    splits into Train / Val / Test (60/20/20, time-ordered) →
    runs ~19,800 parameter combinations on Train set →
    selects best params on Val set (with overfitting guard) →
    saves full ranked results + stores best params in CONFIG →
    Test set is NEVER touched here — locked for final evaluation.

Parameter search space (from CONFIG.tuning):
    EMA pairs     : ~330 valid (fast, slow) combinations
    Stop Loss     : 1%, 2%, 3%, 4%, 5%
    Risk-Reward   : 1:2, 1:3, 1:4  →  TP = SL × RR
    Retest tol    : None, 1%, 1.5%, 2%
    Total         : ~19,800 combinations per stock

Entry logic:
    1. EMA crossover detected (fast EMA crosses above slow EMA)
    2. Wait for retest — price pulls back within retest_tol% of fast EMA
    3. If retest happens within retest_max_bars → enter at next open
    4. If retest_tol is None → no retest required, but still wait
       for price to pull back and re-touch fast EMA within retest_max_bars
       If no touch → skip trade entirely
    5. Exit on SL hit, TP hit, or end of data

Scoring:
    Score = Sharpe − lambda × Max Drawdown
    Hard filters: max_drawdown ≤ 15%, min_trades ≥ 10
    Overfitting guard: val_sharpe ≥ 0.5 × train_sharpe

Outputs (per stock):
    results/reports/{SYMBOL}_ema_tuning.csv   ← all combinations ranked
    CONFIG.tuning.best_params[SYMBOL]         ← best params in memory

Run:
    python -m src.strategy.ema_tuner
    python ema_tuner.py   (from project root)
"""

import os
import sys
import logging
import warnings
import time

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

# ── Stock universe ───────────────────────────────────────────────
STOCK_UNIVERSE = [
    "RELIANCE", "INFY", "TCS", "HDFCBANK", "ICICIBANK",
    "SBIN", "WIPRO", "AXISBANK", "KOTAKBANK",
]


# ═══════════════════════════════════════════════════════════════════
# STEP 1 — LOAD FEATURES & SPLIT
# ═══════════════════════════════════════════════════════════════════

def load_features(symbol: str) -> pd.DataFrame:
    """
    Load clean features CSV produced by feature_pipeline.py.

    Args:
        symbol : Stock symbol (e.g. "RELIANCE")

    Returns:
        DataFrame sorted ascending by Datetime.
    """
    path = CONFIG.paths.stock_features_path(symbol)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n[ema_tuner] Features not found: {path}"
            f"\nRun first: python -m src.features.feature_pipeline"
        )

    df = pd.read_csv(path, parse_dates=["Datetime"])
    df = df.sort_values("Datetime").reset_index(drop=True)

    log.info(
        f"[{symbol}] Loaded {len(df):,} rows | "
        f"{df['Datetime'].min().date()} → {df['Datetime'].max().date()}"
    )
    return df


def time_split(df: pd.DataFrame, symbol: str = "") -> tuple:
    """
    Split data into Train / Val / Test — strictly time-ordered.

    Ratios from CONFIG:
        Train : 60%  →  model learns here
        Val   : 20%  →  best params selected here
        Test  : 20%  →  LOCKED, never used in tuning

    Why time-ordered (not random):
        Random split would allow training on 2022 data
        to predict 2018 — that is look-ahead bias.
        Time split ensures train is always before val/test.

    Returns:
        (df_train, df_val, df_test) — three DataFrames
    """
    n       = len(df)
    n_train = int(n * CONFIG.tuning.train_ratio)
    n_val   = int(n * CONFIG.tuning.val_ratio)

    df_train = df.iloc[:n_train].copy().reset_index(drop=True)
    df_val   = df.iloc[n_train : n_train + n_val].copy().reset_index(drop=True)
    df_test  = df.iloc[n_train + n_val :].copy().reset_index(drop=True)

    tag = f"[{symbol}]" if symbol else "[time_split]"
    log.info(
        f"{tag} Split → "
        f"Train={len(df_train):,} ({df_train['Datetime'].min().date()}→{df_train['Datetime'].max().date()}) | "
        f"Val={len(df_val):,} ({df_val['Datetime'].min().date()}→{df_val['Datetime'].max().date()}) | "
        f"Test={len(df_test):,} ({df_test['Datetime'].min().date()}→{df_test['Datetime'].max().date()})"
    )
    return df_train, df_val, df_test


# ═══════════════════════════════════════════════════════════════════
# STEP 2 — SINGLE BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════

def run_backtest(
    df         : pd.DataFrame,
    ema_fast   : int,
    ema_slow   : int,
    sl         : float,
    tp         : float,
    retest_tol : float | None,
) -> dict:
    """
    Simulate EMA crossover strategy on a single data slice.

    Entry logic:
        1. Detect bullish EMA crossover:
               fast EMA crosses ABOVE slow EMA
               (yesterday fast < slow, today fast > slow)
        2. After crossover, wait for retest:
               Price must pull back and touch within retest_tol%
               of the fast EMA within retest_max_bars days.
               If retest_tol is None → must touch fast EMA exactly
               (within 0% — i.e. Low ≤ fast EMA ≤ High).
               If no retest within retest_max_bars → skip trade.
        3. Enter at next day's Open after retest bar confirmed.
        4. Exit rules (checked each day after entry):
               SL hit : Close ≤ entry_price × (1 − sl)  → exit at Close
               TP hit : Close ≥ entry_price × (1 + tp)  → exit at Close
               End    : Last row of data → exit at Close

    Position sizing:
        One trade at a time. Full capital deployed per trade.
        Transaction cost applied on entry AND exit.

    Args:
        df         : Feature DataFrame (train or val slice)
        ema_fast   : Fast EMA period
        ema_slow   : Slow EMA period
        sl         : Stop loss fraction (e.g. 0.02 = 2%)
        tp         : Take profit fraction (e.g. 0.06 = 6%)
        retest_tol : Retest tolerance fraction or None

    Returns:
        dict with backtest metrics:
            n_trades, win_rate, sharpe, max_drawdown,
            total_return, score, avg_trade_return
    """
    # ── Compute EMAs on this slice ───────────────────────────────
    # Recompute fresh on each slice — avoids data leakage between
    # train/val boundaries where rolling state would carry over.
    fast_ema = df["Close"].ewm(span=ema_fast, adjust=False).mean()
    slow_ema = df["Close"].ewm(span=ema_slow, adjust=False).mean()

    cost     = CONFIG.backtest.transaction_cost_pct
    max_bars = CONFIG.tuning.retest_max_bars

    capital       = CONFIG.backtest.initial_capital
    equity        = capital
    peak_equity   = capital
    max_drawdown  = 0.0

    trade_returns = []
    in_trade      = False
    entry_price   = 0.0
    wait_retest   = False   # True = crossover seen, waiting for retest
    bars_waiting  = 0       # How many bars since crossover

    n = len(df)

    for i in range(1, n):
        close     = df["Close"].iloc[i]
        low       = df["Low"].iloc[i]
        high      = df["High"].iloc[i]
        fast_val  = fast_ema.iloc[i]
        fast_prev = fast_ema.iloc[i - 1]
        slow_prev = slow_ema.iloc[i - 1]

        # ── If in a trade: check SL / TP ────────────────────────
        if in_trade:
            sl_price = entry_price * (1 - sl)
            tp_price = entry_price * (1 + tp)
            exit_price = None

            if close <= sl_price:
                exit_price = close    # SL hit
            elif close >= tp_price:
                exit_price = close    # TP hit
            elif i == n - 1:
                exit_price = close    # End of data

            if exit_price is not None:
                # Apply round-trip transaction cost
                ret = (exit_price / entry_price) - 1 - 2 * cost
                trade_returns.append(ret)

                equity      *= (1 + ret)
                peak_equity  = max(peak_equity, equity)
                dd           = (peak_equity - equity) / peak_equity
                max_drawdown = max(max_drawdown, dd)

                in_trade    = False
                entry_price = 0.0
                # Don't continue — allow crossover check on same bar
            else:
                continue    # Still in trade — skip signal detection

        # ── Not in trade: check for crossover ───────────────────
        if not wait_retest:
            # Bullish crossover: yesterday fast < slow, today fast > slow
            crossed = (
                fast_prev <= slow_prev and
                fast_ema.iloc[i] > slow_ema.iloc[i]
            )
            if crossed:
                wait_retest  = True
                bars_waiting = 0
            continue

        # ── Waiting for retest ───────────────────────────────────
        if wait_retest:
            bars_waiting += 1

            # Check if retest condition is met on this bar
            if retest_tol is None:
                # Must touch fast EMA: Low ≤ fast EMA ≤ High
                retest_hit = (low <= fast_val <= high)
            else:
                # Price must come within retest_tol% of fast EMA
                # i.e. Low ≤ fast_val × (1 + retest_tol)
                # and Close is still above fast EMA (trend intact)
                upper_band  = fast_val * (1 + retest_tol)
                retest_hit  = (low <= upper_band) and (close > fast_val * (1 - retest_tol))

            if retest_hit:
                # Enter at next bar's open (i+1)
                if i + 1 < n:
                    entry_price  = df["Open"].iloc[i + 1]
                    entry_price *= (1 + cost)   # entry cost
                    in_trade     = True
                wait_retest  = False
                bars_waiting = 0

            elif bars_waiting >= max_bars:
                # Retest window expired — skip this trade
                wait_retest  = False
                bars_waiting = 0

    # ── Compute metrics ─────────────────────────────────────────
    n_trades = len(trade_returns)

    if n_trades < CONFIG.tuning.min_trades:
        # Not enough trades — return sentinel values
        return {
            "n_trades"        : n_trades,
            "win_rate"        : 0.0,
            "sharpe"          : -99.0,
            "max_drawdown"    : 1.0,
            "total_return"    : 0.0,
            "avg_trade_return": 0.0,
            "score"           : -99.0,
        }

    returns_arr  = np.array(trade_returns)
    win_rate     = float(np.mean(returns_arr > 0))
    total_return = float(np.prod(1 + returns_arr) - 1)
    avg_ret      = float(np.mean(returns_arr))

    # Annualised Sharpe — daily returns approximation
    # Using per-trade returns as daily proxy since we hold 1 trade at a time
    rf_daily = CONFIG.backtest.risk_free_rate / CONFIG.backtest.annualization_factor
    excess   = returns_arr - rf_daily
    if returns_arr.std() < 1e-9:
        sharpe = 0.0
    else:
        sharpe = float(
            np.sqrt(CONFIG.backtest.annualization_factor) *
            excess.mean() / returns_arr.std()
        )

    score = CONFIG.tuning.compute_score(sharpe, max_drawdown)

    return {
        "n_trades"        : n_trades,
        "win_rate"        : round(win_rate, 4),
        "sharpe"          : round(sharpe, 4),
        "max_drawdown"    : round(max_drawdown, 4),
        "total_return"    : round(total_return, 4),
        "avg_trade_return": round(avg_ret, 4),
        "score"           : round(score, 4),
    }


# ═══════════════════════════════════════════════════════════════════
# STEP 3 — GRID SEARCH OVER ALL COMBINATIONS
# ═══════════════════════════════════════════════════════════════════

def grid_search(
    df_train : pd.DataFrame,
    df_val   : pd.DataFrame,
    symbol   : str = "",
) -> pd.DataFrame:
    """
    Exhaustive grid search over all parameter combinations.

    Flow per combination:
        1. Run backtest on TRAIN set
        2. Apply hard filters (max_drawdown ≤ 15%, min_trades ≥ 10)
        3. Run backtest on VAL set for the same params
        4. Apply overfitting guard (val_sharpe ≥ 0.5 × train_sharpe)
        5. Compute val score → store results

    Args:
        df_train : Training data slice
        df_val   : Validation data slice
        symbol   : Symbol for logging

    Returns:
        DataFrame of all results sorted by val_score descending.
        Includes both passing and filtered-out combinations
        (filtered rows have val_score = NaN for transparency).
    """
    combos   = CONFIG.tuning.get_all_combinations()
    tag      = f"[{symbol}]" if symbol else "[grid_search]"
    n_combos = len(combos)

    log.info(f"{tag} Starting grid search — {n_combos:,} combinations...")
    t_start  = time.time()

    rows = []

    for i, params in enumerate(combos):
        ema_fast   = params["ema_fast"]
        ema_slow   = params["ema_slow"]
        sl         = params["sl"]
        rr         = params["rr"]
        tp         = params["tp"]
        retest_tol = params["retest_tol"]

        # ── Train backtest ───────────────────────────────────────
        train_res = run_backtest(df_train, ema_fast, ema_slow, sl, tp, retest_tol)

        row = {
            "ema_fast"         : ema_fast,
            "ema_slow"         : ema_slow,
            "sl_pct"           : round(sl * 100, 1),
            "rr"               : rr,
            "tp_pct"           : round(tp * 100, 1),
            "retest_tol"       : retest_tol,
            # Train metrics
            "train_trades"     : train_res["n_trades"],
            "train_sharpe"     : train_res["sharpe"],
            "train_drawdown"   : train_res["max_drawdown"],
            "train_return"     : train_res["total_return"],
            "train_winrate"    : train_res["win_rate"],
            "train_score"      : train_res["score"],
            # Val metrics (filled below if passes filters)
            "val_trades"       : np.nan,
            "val_sharpe"       : np.nan,
            "val_drawdown"     : np.nan,
            "val_return"       : np.nan,
            "val_winrate"      : np.nan,
            "val_score"        : np.nan,
            "passed_filters"   : False,
        }

        # ── Hard filters on train ────────────────────────────────
        dd_ok     = train_res["max_drawdown"] <= CONFIG.tuning.max_drawdown_limit
        trades_ok = train_res["n_trades"]     >= CONFIG.tuning.min_trades
        sharpe_ok = train_res["sharpe"]       > 0

        if not (dd_ok and trades_ok and sharpe_ok):
            rows.append(row)
            continue

        # ── Val backtest ─────────────────────────────────────────
        val_res = run_backtest(df_val, ema_fast, ema_slow, sl, tp, retest_tol)

        row["val_trades"]   = val_res["n_trades"]
        row["val_sharpe"]   = val_res["sharpe"]
        row["val_drawdown"] = val_res["max_drawdown"]
        row["val_return"]   = val_res["total_return"]
        row["val_winrate"]  = val_res["win_rate"]
        row["val_score"]    = val_res["score"]

        # ── Overfitting guard ────────────────────────────────────
        # Val Sharpe must be at least 50% of train Sharpe
        overfit_ok = (
            val_res["sharpe"] >=
            CONFIG.tuning.overfitting_threshold * train_res["sharpe"]
        )
        val_dd_ok = val_res["max_drawdown"] <= CONFIG.tuning.max_drawdown_limit

        if overfit_ok and val_dd_ok and val_res["n_trades"] >= CONFIG.tuning.min_trades:
            row["passed_filters"] = True

        rows.append(row)

        # ── Progress log every 2000 combos ──────────────────────
        if (i + 1) % 2000 == 0:
            elapsed = time.time() - t_start
            pct     = (i + 1) / n_combos * 100
            log.info(f"{tag}   {i+1:>6,}/{n_combos:,} ({pct:.0f}%) — {elapsed:.0f}s elapsed")

    elapsed = time.time() - t_start
    results = pd.DataFrame(rows)

    # Sort: passed filters first, then by val_score descending
    results = results.sort_values(
        ["passed_filters", "val_score"],
        ascending=[False, False],
    ).reset_index(drop=True)

    n_passed = results["passed_filters"].sum()
    log.info(
        f"{tag} Grid search complete | {elapsed:.1f}s | "
        f"{n_passed:,}/{n_combos:,} combinations passed all filters"
    )
    return results


# ═══════════════════════════════════════════════════════════════════
# STEP 4 — SELECT BEST PARAMS
# ═══════════════════════════════════════════════════════════════════

def select_best(results: pd.DataFrame, symbol: str = "") -> dict | None:
    """
    Select the best parameter combination from grid search results.

    Best = highest val_score among combinations that passed all filters.

    Args:
        results : Full grid search results DataFrame
        symbol  : Symbol for logging

    Returns:
        dict of best params, or None if no combination passed filters.
    """
    tag     = f"[{symbol}]" if symbol else "[select_best]"
    passing = results[results["passed_filters"] == True]

    if passing.empty:
        log.warning(
            f"{tag} ⚠️  No combination passed all filters. "
            f"Consider relaxing CONFIG.tuning constraints."
        )
        return None

    best_row = passing.iloc[0]   # already sorted by val_score desc

    best_params = {
        "ema_fast"  : int(best_row["ema_fast"]),
        "ema_slow"  : int(best_row["ema_slow"]),
        "sl"        : float(best_row["sl_pct"]) / 100,
        "rr"        : int(best_row["rr"]),
        "tp"        : float(best_row["tp_pct"]) / 100,
        "retest_tol": best_row["retest_tol"] if not pd.isna(best_row["retest_tol"]) else None,
        # Val performance at these params
        "val_sharpe"    : float(best_row["val_sharpe"]),
        "val_drawdown"  : float(best_row["val_drawdown"]),
        "val_score"     : float(best_row["val_score"]),
        "val_trades"    : int(best_row["val_trades"]),
        "train_sharpe"  : float(best_row["train_sharpe"]),
    }

    log.info(f"{tag} ✅ Best params selected:")
    log.info(f"  EMA ({best_params['ema_fast']}, {best_params['ema_slow']}) | "
             f"SL {best_params['sl']*100:.1f}% | "
             f"RR 1:{best_params['rr']} | "
             f"TP {best_params['tp']*100:.1f}% | "
             f"Retest {best_params['retest_tol']}")
    log.info(f"  Val  → Sharpe={best_params['val_sharpe']:.4f}  "
             f"DD={best_params['val_drawdown']*100:.1f}%  "
             f"Score={best_params['val_score']:.4f}  "
             f"Trades={best_params['val_trades']}")
    log.info(f"  Train → Sharpe={best_params['train_sharpe']:.4f}")

    return best_params


# ═══════════════════════════════════════════════════════════════════
# STEP 5 — SAVE RESULTS
# ═══════════════════════════════════════════════════════════════════

def save_results(results: pd.DataFrame, symbol: str):
    """
    Save full ranked grid search results to CSV.
    Includes ALL ~19,800 combinations — passed and filtered.
    'passed_filters' column shows which ones are usable.
    """
    path = CONFIG.paths.tuning_results_path(symbol)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    results.to_csv(path, index=False)
    log.info(f"[{symbol}] ✅ Tuning results → {path}  ({len(results):,} rows)")


# ═══════════════════════════════════════════════════════════════════
# STEP 6 — PER-SYMBOL TUNING PIPELINE
# ═══════════════════════════════════════════════════════════════════

def tune_symbol(symbol: str) -> dict | None:
    """
    Run full tuning pipeline for one stock symbol.

    Steps:
        1. Load features CSV
        2. Split into Train / Val / Test
        3. Grid search on Train → Val
        4. Select best params
        5. Save full results CSV
        6. Store best params in CONFIG

    Args:
        symbol : Stock symbol string (e.g. "RELIANCE")

    Returns:
        Best params dict, or None if tuning failed / no params found.
    """
    log.info(f"{'═'*60}")
    log.info(f"  Tuning: {symbol}")
    log.info(f"{'═'*60}")

    try:
        # Step 1: Load
        df = load_features(symbol)

        # Step 2: Split
        df_train, df_val, df_test = time_split(df, symbol=symbol)

        # Step 3: Grid search
        results = grid_search(df_train, df_val, symbol=symbol)

        # Step 4: Select best
        best_params = select_best(results, symbol=symbol)

        # Step 5: Save full results
        save_results(results, symbol)

        # Step 6: Store in CONFIG
        if best_params is not None:
            CONFIG.tuning.save_best_params(symbol, best_params)

        return best_params

    except FileNotFoundError as e:
        log.error(f"[{symbol}] ❌ {e}")
        return None

    except Exception as e:
        log.error(f"[{symbol}] ❌ Unexpected error: {e}", exc_info=True)
        return None


# ═══════════════════════════════════════════════════════════════════
# MAIN — runs all stocks
# ═══════════════════════════════════════════════════════════════════

def run_tuning(symbols: list = None) -> dict:
    """
    Run EMA tuning pipeline for all stocks in STOCK_UNIVERSE.

    Called by:
        main.py                          → full pipeline
        python -m src.strategy.ema_tuner → standalone

    Args:
        symbols : Optional override. Defaults to STOCK_UNIVERSE.

    Returns:
        dict of {symbol: best_params_dict or None}
    """
    symbols = symbols or STOCK_UNIVERSE

    log.info("=" * 60)
    log.info("  EMA Tuner — Starting")
    log.info(f"  Symbols   : {symbols}")
    log.info(f"  Total     : {len(symbols)}")
    log.info(f"  Combos/stock : {len(CONFIG.tuning.get_all_combinations()):,}")
    log.info("=" * 60)

    all_results = {}
    succeeded   = []
    failed      = []

    for symbol in symbols:
        symbol = symbol.upper()
        best   = tune_symbol(symbol)
        all_results[symbol] = best

        if best is not None:
            succeeded.append(symbol)
        else:
            failed.append(symbol)

    # ── Summary ─────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("  EMA Tuner — Complete")
    log.info("=" * 60)
    log.info(f"  ✅ Tuned  ({len(succeeded)}) : {succeeded}")
    if failed:
        log.warning(f"  ❌ Failed ({len(failed)}) : {failed}")
    log.info("=" * 60)

    # Print best params summary table
    print(f"\n{'═'*70}")
    print(f"  {'Symbol':<12} {'EMA':^10} {'SL':>5} {'RR':>5} {'TP':>5} "
          f"{'ValSharpe':>10} {'ValDD':>8} {'Trades':>7}")
    print(f"  {'─'*66}")
    for sym, bp in all_results.items():
        if bp:
            print(
                f"  {sym:<12} "
                f"({bp['ema_fast']},{bp['ema_slow']}){'':<3} "
                f"{bp['sl']*100:.0f}%{'':<2} "
                f"1:{bp['rr']}{'':<2} "
                f"{bp['tp']*100:.0f}%{'':<2} "
                f"{bp['val_sharpe']:>10.4f} "
                f"{bp['val_drawdown']*100:>7.1f}% "
                f"{bp['val_trades']:>7}"
            )
        else:
            print(f"  {sym:<12} ❌ No valid params found")
    print(f"{'═'*70}\n")

    return all_results


# ═══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    results = run_tuning()

    print(f"\n{'═'*60}")
    print(f"  Tuning Results saved to: results/reports/")
    print(f"  Best params stored in CONFIG.tuning.best_params")
    print(f"  Next step: python -m src.strategy.ema_strategy")
    print(f"{'═'*60}\n")