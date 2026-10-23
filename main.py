"""
main.py — Master Pipeline Orchestrator
=======================================
Place at: D:/AI-ML/regime_trading_system/main.py  (project root)

WHAT THIS FILE DOES
-------------------
Runs the full trading system pipeline in the correct order:

  Step 1 — trend.py         : Train NIFTY regime ML model
                               Predicts UP / SIDEWAYS / DOWN for each day
                               Output: data/processed/nifty_regime.csv

  Step 2 — ema_tuner.py     : Grid search best EMA params per stock
                               Runs on Train+Val sets ONLY (test locked)
                               Output: results/reports/{SYMBOL}_ema_tuning.csv

  Step 3 — ema_strategy.py  : Final backtest on unseen Test set
                               Uses best params from Step 2
                               Output: results/trades/, plots/, reports/

RETRY LOGIC
-----------
Each step is retried once on failure before being marked as failed.
This handles transient issues (file locks, memory spikes, etc).
A failure in Step 1 (regime) aborts the entire run — Steps 2 and 3
depend on the regime CSV being present.
A failure in Step 2 for a stock skips Step 3 for that stock only.

RUN
---
Full pipeline, all 9 stocks:
    python main.py

Single stock (all 3 steps for that stock only):
    python main.py --symbol RELIANCE

Skip steps already done (regime model already trained):
    python main.py --skip-regime
    python main.py --symbol RELIANCE --skip-regime

Force re-run everything even if outputs exist:
    python main.py --force

NOTES
-----
- fetch_stocks and feature_pipeline are NOT run here.
  They are run separately (once) as data preparation steps.
  This pipeline assumes data/processed/{SYMBOL}_features.csv
  and data/raw/ are already populated.
- Run order: trend → tuner → strategy. Never skip this order.
- CONFIG is shared in memory across all steps in the same run,
  so tuner's best_params are available to strategy immediately
  without disk reads.
"""

import os
import sys
import time
import logging
import argparse
import traceback
from datetime import datetime

# ── Project root on path ─────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CONFIG

# ── Logging setup ────────────────────────────────────────────────
log = logging.getLogger("main")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(
        "%(asctime)s  [%(levelname)s]  %(message)s",
        "%Y-%m-%d %H:%M:%S",
    ))
    log.addHandler(_h)
    log.setLevel(logging.INFO)
    log.propagate = False

# ── Stock universe ────────────────────────────────────────────────
STOCK_UNIVERSE = [
    "RELIANCE", "INFY", "TCS", "HDFCBANK", "ICICIBANK",
    "SBIN", "WIPRO", "AXISBANK", "KOTAKBANK",
]

MAX_RETRIES = 1   # retry once on failure, then skip


# ════════════════════════════════════════════════════════════════
# UTILITIES
# ════════════════════════════════════════════════════════════════

def banner(title: str, width: int = 66):
    """Print a prominent section banner to console."""
    print()
    print("═" * width)
    print(f"  {title}")
    print("═" * width)


def step_header(step_num: int, title: str, width: int = 66):
    """Print a step header."""
    print()
    print("─" * width)
    print(f"  STEP {step_num}  ──  {title}")
    print("─" * width)


def format_elapsed(seconds: float) -> str:
    """Format elapsed time as mm:ss or hh:mm:ss."""
    seconds = int(seconds)
    h, rem  = divmod(seconds, 3600)
    m, s    = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def run_with_retry(fn, label: str, retries: int = MAX_RETRIES):
    """
    Call fn(). On exception, retry up to `retries` times.
    Returns (result, success: bool).
    
    WHY RETRY:
    Transient failures are common in long-running pipelines —
    file handle contention, memory spikes on large grid searches,
    temporary I/O errors. One retry catches ~95% of these without
    masking real bugs (which will fail on retry too).
    """
    for attempt in range(1 + retries):
        try:
            result = fn()
            return result, True
        except Exception as e:
            if attempt < retries:
                log.warning(
                    f"[{label}] Attempt {attempt+1} failed: {e}  "
                    f"— retrying in 3 seconds..."
                )
                time.sleep(3)
            else:
                log.error(
                    f"[{label}] Failed after {attempt+1} attempt(s): {e}"
                )
                log.debug(traceback.format_exc())
    return None, False


def check_prerequisites(symbols: list) -> bool:
    """
    Verify that required input files exist before starting.
    Fails fast with clear error messages rather than cryptic
    KeyErrors deep inside the pipeline.

    Checks:
    - data/processed/{SYMBOL}_features.csv  (for each stock)
    - data/processed/nifty_regime.csv       (only if --skip-regime used)
    """
    missing = []
    for symbol in symbols:
        fp = CONFIG.paths.stock_features_path(symbol)
        if not os.path.exists(fp):
            missing.append(fp)

    if missing:
        print()
        log.error("PREREQUISITE CHECK FAILED — missing input files:")
        for f in missing:
            log.error(f"  ✗  {f}")
        log.error(
            "\nFix: run these first:\n"
            "  python -m src.data.fetch_stocks\n"
            "  python -m src.features.feature_pipeline"
        )
        return False

    log.info(f"Prerequisites OK — all {len(symbols)} feature files found")
    return True


def check_regime_file() -> bool:
    """Check if NIFTY regime CSV already exists (for --skip-regime)."""
    return os.path.exists(CONFIG.paths.nifty_regime_path())


# ════════════════════════════════════════════════════════════════
# STEP 1 — NIFTY REGIME MODEL
# ════════════════════════════════════════════════════════════════

def run_step_regime(force: bool = False) -> bool:
    """
    Train NIFTY regime ML model and generate regime predictions.

    WHY THIS RUNS FIRST:
    The regime CSV (nifty_regime.csv) is the gate keeper for all
    stock trades. The tuner (Step 2) needs it to evaluate whether
    crossover signals occurred on UP/SIDEWAYS days. The strategy
    (Step 3) needs it to apply the DOWN-block gate. Nothing works
    without it.

    WHY WE CHECK IF IT EXISTS:
    Training the ML model takes ~1-2 minutes. If regime CSV already
    exists and --force is not set, we skip re-training to save time.
    Use --force or --skip-regime=False to re-train from scratch.

    Returns True on success, False on failure.
    """
    regime_path = CONFIG.paths.nifty_regime_path()

    if not force and os.path.exists(regime_path):
        log.info(f"Regime file exists: {regime_path}")
        log.info("Skipping regime re-training. Use --force to retrain.")
        return True

    log.info("Training NIFTY regime model...")

    def _run():
        from src.regime.trend import run_regime_pipeline
        run_regime_pipeline()

    _, ok = run_with_retry(_run, label="NIFTY Regime")

    if ok:
        if os.path.exists(regime_path):
            log.info(f"Regime model trained. Output: {regime_path}")
        else:
            log.error("Regime function completed but output file not found.")
            ok = False

    return ok


# ════════════════════════════════════════════════════════════════
# STEP 2 — EMA TUNER
# ════════════════════════════════════════════════════════════════

def run_step_tuner(symbols: list, force: bool = False) -> dict:
    """
    Run EMA parameter grid search for each stock.

    Returns dict of {symbol: best_params or None}.
    Symbols that fail tuning are excluded from Step 3.

    WHY PER-SYMBOL RETRY:
    Grid search on one stock failing (e.g. corrupted features CSV)
    shouldn't block the other 8 stocks. Each is retried independently.
    Failed symbols are tracked and excluded from Step 3.

    WHY FORCE CHECK:
    If tuning CSV already exists for a symbol AND --force is not set,
    we skip that symbol. Useful when re-running after a partial failure
    — only re-tunes the stocks that failed previously.
    """
    from src.strategy.ema_tuner import tune_symbol

    tuner_results = {}
    succeeded     = []
    skipped       = []
    failed        = []

    for symbol in symbols:
        symbol    = symbol.upper()
        csv_path  = CONFIG.paths.tuning_results_path(symbol)

        # Skip if already tuned and --force not set
        if not force and os.path.exists(csv_path):
            log.info(f"[{symbol}] Tuning CSV exists — skipping. Use --force to retune.")
            # Load best params from CSV into CONFIG memory
            # so Step 3 can read from CONFIG without disk read
            try:
                import pandas as pd
                df      = pd.read_csv(csv_path)
                passing = df[df["passed_filters"] == True]
                if not passing.empty:
                    r          = passing.sort_values("val_score", ascending=False).iloc[0]
                    retest_tol = float(r["retest_tol"]) if not pd.isna(r["retest_tol"]) else None
                    params = {
                        "ema_fast"    : int(r["ema_fast"]),
                        "ema_slow"    : int(r["ema_slow"]),
                        "sl"          : float(r["sl_pct"]) / 100,
                        "rr"          : int(r["rr"]),
                        "tp"          : float(r["tp_pct"]) / 100,
                        "retest_tol"  : retest_tol,
                        "val_sharpe"  : float(r["val_sharpe"]),
                        "val_drawdown": float(r["val_drawdown"]),
                        "val_score"   : float(r["val_score"]),
                        "val_trades"  : int(r["val_trades"]),
                        "train_sharpe": float(r["train_sharpe"]),
                    }
                    CONFIG.tuning.save_best_params(symbol, params)
                    tuner_results[symbol] = params
                    skipped.append(symbol)
                    continue
                else:
                    log.warning(f"[{symbol}] CSV exists but no passing combos — re-tuning.")
            except Exception as e:
                log.warning(f"[{symbol}] Could not read existing CSV ({e}) — re-tuning.")

        # Run tuner for this symbol
        def _tune(sym=symbol):
            return tune_symbol(sym)

        result, ok = run_with_retry(_tune, label=symbol)

        if ok and result is not None:
            tuner_results[symbol] = result
            succeeded.append(symbol)
        else:
            tuner_results[symbol] = None
            failed.append(symbol)
            log.error(
                f"[{symbol}] Tuning failed — this stock will be excluded "
                f"from the final backtest."
            )

    log.info(
        f"Tuner summary — "
        f"Succeeded: {len(succeeded)} | Skipped: {len(skipped)} | Failed: {len(failed)}"
    )
    if failed:
        log.warning(f"Failed symbols: {failed}")

    return tuner_results


# ════════════════════════════════════════════════════════════════
# STEP 3 — EMA STRATEGY (FINAL BACKTEST)
# ════════════════════════════════════════════════════════════════

def run_step_strategy(symbols: list) -> dict:
    """
    Run final backtest on locked Test set for each stock.

    Only runs for symbols where tuning succeeded (params available).
    Each symbol is retried once on failure.

    Returns dict of {symbol: metrics or None}.
    """
    from src.strategy.ema_strategy import run_symbol

    strategy_results = {}
    succeeded        = []
    failed           = []

    for symbol in symbols:
        symbol = symbol.upper()

        def _backtest(sym=symbol):
            return run_symbol(sym)

        result, ok = run_with_retry(_backtest, label=symbol)

        if ok and result is not None:
            strategy_results[symbol] = result
            succeeded.append(symbol)
        else:
            strategy_results[symbol] = None
            failed.append(symbol)

    log.info(
        f"Strategy summary — "
        f"Succeeded: {len(succeeded)} | Failed: {len(failed)}"
    )
    if failed:
        log.warning(f"Failed symbols: {failed}")

    return strategy_results


# ════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════════════

def print_final_summary(
    symbols          : list,
    tuner_results    : dict,
    strategy_results : dict,
    pipeline_start   : float,
):
    """
    Print a clean end-to-end summary showing what succeeded,
    what failed, and the final performance numbers for every stock.
    """
    elapsed = format_elapsed(time.time() - pipeline_start)

    banner("PIPELINE COMPLETE — FINAL SUMMARY")

    print(f"\n  Total time   : {elapsed}")
    print(f"  Symbols run  : {len(symbols)}")
    print(
        f"  Tuned OK     : "
        f"{sum(1 for v in tuner_results.values() if v is not None)}"
    )
    print(
        f"  Backtested OK: "
        f"{sum(1 for v in strategy_results.values() if v is not None)}"
    )

    # Performance table
    print(f"\n{'═'*82}")
    print(
        f"  {'Symbol':<12} {'Return':>8} {'B&H':>8} {'Alpha':>8} "
        f"{'Sharpe':>8} {'MaxDD':>7} {'WinRate':>8} {'Trades':>7}"
    )
    print(f"  {'─'*78}")

    for sym in symbols:
        sym = sym.upper()
        m   = strategy_results.get(sym)
        t   = tuner_results.get(sym)

        if m is not None:
            alpha = m["total_return"] - m["buy_hold_return"]
            print(
                f"  {sym:<12} "
                f"{m['total_return']:>+7.1f}%  "
                f"{m['buy_hold_return']:>+6.1f}%  "
                f"{alpha:>+6.1f}%  "
                f"{m['sharpe']:>7.2f}  "
                f"{m['max_drawdown']:>6.1f}%  "
                f"{m['win_rate']:>7.1f}%  "
                f"{m['total_trades']:>6}"
            )
        elif t is None:
            print(f"  {sym:<12}  ✗ Tuning failed")
        else:
            print(f"  {sym:<12}  ✗ Backtest failed")

    print(f"{'═'*82}")

    # Output file locations
    print(f"\n  OUTPUT FILES")
    print(f"  {'─'*40}")
    print(f"  Trade logs   : results/trades/")
    print(f"  Equity plots : results/plots/")
    print(f"  Reports      : results/reports/")
    print(f"  Regime CSV   : {CONFIG.paths.nifty_regime_path()}")
    print()


# ════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ════════════════════════════════════════════════════════════════

def main():
    # ── Argument parsing ─────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description=(
            "Regime-Aware EMA Trading System - Full Pipeline\n"
            "Runs: NIFTY Regime Model -> EMA Tuner -> EMA Strategy Backtest"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help=(
            "Run pipeline for a single stock only. "
            "e.g. --symbol RELIANCE"
        ),
    )
    parser.add_argument(
        "--skip-regime",
        action="store_true",
        default=False,
        help=(
            "Skip Step 1 (regime model training). "
            "Use when nifty_regime.csv already exists from a previous run."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help=(
            "Force re-run all steps even if output files already exist. "
            "Without this flag, existing tuning CSVs and regime CSV are reused."
        ),
    )
    args = parser.parse_args()

    # ── Determine symbols ─────────────────────────────────────────
    if args.symbol:
        symbols = [args.symbol.upper()]
        log.info(f"Single-symbol mode: {symbols[0]}")
    else:
        symbols = STOCK_UNIVERSE
        log.info(f"Full pipeline mode: {len(symbols)} stocks")

    # ── Create output directories ─────────────────────────────────
    CONFIG.paths.create_all_dirs()

    pipeline_start = time.time()

    banner(
        f"REGIME-AWARE EMA TRADING SYSTEM  —  "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )
    print(f"  Symbols      : {symbols}")
    print(f"  Skip regime  : {args.skip_regime}")
    print(f"  Force re-run : {args.force}")
    print(f"  Combos/stock : {len(CONFIG.tuning.get_all_combinations()):,}")

    # ── Prerequisite check ────────────────────────────────────────
    if not check_prerequisites(symbols):
        sys.exit(1)

    # ════════════════════════════════════════════════════════════
    # STEP 1 — NIFTY REGIME MODEL
    # ════════════════════════════════════════════════════════════
    step_header(1, "NIFTY REGIME MODEL  (trend.py)")

    if args.skip_regime:
        if check_regime_file():
            log.info("--skip-regime set. Using existing regime file.")
            regime_ok = True
        else:
            log.error(
                "--skip-regime set but regime file not found: "
                f"{CONFIG.paths.nifty_regime_path()}\n"
                "Remove --skip-regime and re-run."
            )
            sys.exit(1)
    else:
        t0        = time.time()
        regime_ok = run_step_regime(force=args.force)
        log.info(f"Step 1 elapsed: {format_elapsed(time.time() - t0)}")

    if not regime_ok:
        log.error(
            "STEP 1 FAILED — Cannot proceed without NIFTY regime predictions.\n"
            "Steps 2 and 3 both depend on nifty_regime.csv."
        )
        sys.exit(1)

    # ════════════════════════════════════════════════════════════
    # STEP 2 — EMA TUNER
    # ════════════════════════════════════════════════════════════
    step_header(2, "EMA PARAMETER TUNER  (ema_tuner.py)")

    t0             = time.time()
    tuner_results  = run_step_tuner(symbols, force=args.force)
    log.info(f"Step 2 elapsed: {format_elapsed(time.time() - t0)}")

    # Only backtest stocks where tuning succeeded
    strategy_symbols = [
        sym for sym in symbols
        if tuner_results.get(sym.upper()) is not None
    ]

    if not strategy_symbols:
        log.error(
            "STEP 2: All symbols failed tuning. "
            "Nothing to backtest. Check feature CSVs and config."
        )
        print_final_summary(symbols, tuner_results, {}, pipeline_start)
        sys.exit(1)

    excluded = set(symbols) - set(strategy_symbols)
    if excluded:
        log.warning(
            f"Excluding from Step 3 (tuning failed): "
            f"{sorted(excluded)}"
        )

    # ════════════════════════════════════════════════════════════
    # STEP 3 — EMA STRATEGY (FINAL BACKTEST)
    # ════════════════════════════════════════════════════════════
    step_header(3, "EMA STRATEGY — FINAL TEST SET BACKTEST  (ema_strategy.py)")

    t0               = time.time()
    strategy_results = run_step_strategy(strategy_symbols)
    log.info(f"Step 3 elapsed: {format_elapsed(time.time() - t0)}")

    # Fill in None for excluded symbols
    for sym in symbols:
        if sym.upper() not in strategy_results:
            strategy_results[sym.upper()] = None

    # ════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ════════════════════════════════════════════════════════════
    print_final_summary(symbols, tuner_results, strategy_results, pipeline_start)

    # Exit code — non-zero if any symbol failed completely
    any_failed = any(
        strategy_results.get(s.upper()) is None for s in symbols
    )
    sys.exit(1 if any_failed else 0)


if __name__ == "__main__":
    main()

