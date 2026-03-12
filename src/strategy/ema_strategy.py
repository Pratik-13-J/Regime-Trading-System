"""
ema_strategy.py  --  EMA Crossover Strategy : Final Backtest on Test Set
Place at: src/strategy/ema_strategy.py

PURPOSE
-------
Final evaluation on the LOCKED Test set (last 20%). These rows
were never touched during tuning or validation. This file gives
the most honest estimate of real live trading performance.

PIPELINE POSITION
-----------------
fetch_stocks -> feature_pipeline -> trend (NIFTY regime) -> ema_tuner -> ema_strategy (HERE)

ENTRY LOGIC  (must be IDENTICAL to ema_tuner.run_backtest)
----------------------------------------------------------
1. Regime gate  : DOWN -> skip everything. UP / SIDEWAYS -> proceed.
2. EMA crossover: fast EMA crosses above slow EMA (trend shifts bullish).
3. Retest wait  : price must pull back to fast EMA within retest_max_bars.
                  retest_tol=None -> candle physically touches EMA (Low<=EMA<=High).
                  retest_tol=0.01 -> price comes within 1% of EMA, close still above.
                  No retest within max_bars -> trade skipped entirely.
4. Entry at NEXT bar Open (after retest confirmed).
5. Exit: SL hit (Close <= entry*(1-sl)) / TP hit (Close >= entry*(1+tp)) / EOD.
   Transaction cost applied on BOTH entry and exit legs.

REGIME GATE DETAIL
------------------
DOWN     : No new entries. Cancel any pending retest. Active trades continue
           (exit via SL/TP -- do not panic-sell on regime change).
UP       : All signals active.
SIDEWAYS : All signals active -- individual stocks can trend even when NIFTY is flat.

OUTPUTS per stock
-----------------
results/trades/{SYMBOL}_trade_log.csv
results/plots/{SYMBOL}_equity_curve.png
results/reports/{SYMBOL}_performance.txt

RUN
---
python -m src.strategy.ema_strategy                    # all stocks
python -m src.strategy.ema_strategy --symbol RELIANCE  # one stock
"""

import os, sys, logging, warnings, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from config import CONFIG

warnings.filterwarnings("ignore")

log = logging.getLogger(__name__)
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(
        "%(asctime)s  [%(levelname)s]  %(message)s", "%Y-%m-%d %H:%M:%S"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)
    log.propagate = False

STOCK_UNIVERSE = [
    "RELIANCE", "INFY", "TCS", "HDFCBANK", "ICICIBANK",
    "SBIN", "WIPRO", "AXISBANK", "KOTAKBANK",
]


# ================================================================
# STEP 1  LOAD BEST PARAMS
# ================================================================
# WHY TWO SOURCES:
# Pipeline mode (main.py)  -> tuner stores params in CONFIG memory
#                             same Python session, no disk IO needed.
# Standalone mode (restart)-> CONFIG empty, fall back to tuning CSV.
# ================================================================

def load_best_params(symbol):
    # Priority 1: CONFIG memory (tuner ran this session)
    params = CONFIG.tuning.get_best_params(symbol)
    if params is not None:
        log.info(f"[{symbol}] Params from CONFIG (in-memory)")
        return params

    # Priority 2: Tuning results CSV (standalone / post-restart)
    csv_path = CONFIG.paths.tuning_results_path(symbol)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"[{symbol}] No params found anywhere.\n"
            f"  CONFIG is empty (tuner not run this session).\n"
            f"  CSV not found: {csv_path}\n"
            f"  Fix: run python -m src.strategy.ema_tuner first."
        )

    log.info(f"[{symbol}] Loading params from CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    # passed_filters=True means all 4 filters passed:
    # max_drawdown <= 15%, min_trades >= 10, sharpe > 0,
    # val_sharpe >= 0.5 * train_sharpe (overfitting guard)
    passing = df[df["passed_filters"] == True]
    if passing.empty:
        raise ValueError(
            f"[{symbol}] Tuning CSV exists but zero combos passed all filters.\n"
            f"  Relax CONFIG.tuning constraints and re-run ema_tuner."
        )

    # Best = highest val_score = Sharpe - lambda * MaxDrawdown
    r = passing.sort_values("val_score", ascending=False).iloc[0]

    # retest_tol is stored as NaN in CSV when its value was None.
    # Must convert NaN back to Python None, not float('nan').
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
    log.info(
        f"[{symbol}] EMA({params['ema_fast']},{params['ema_slow']}) "
        f"SL={params['sl']*100:.1f}% TP={params['tp']*100:.1f}% "
        f"Retest={params['retest_tol']}"
    )
    return params


# ================================================================
# STEP 2  LOAD AND MERGE DATA
# ================================================================
# WHY MERGE STOCK FEATURES WITH NIFTY REGIME:
# The regime gate (DOWN=block) requires NIFTY regime on every
# trading day. Stock features CSV has OHLCV + 22 indicators.
# NIFTY regime CSV has predicted_regime per date from trend.py.
# We need both on the same row.
#
# WHY INNER JOIN (not left join):
# If a date exists in stock CSV but not in regime CSV, we don't
# know the regime. Safer to exclude that day entirely.
#
# WHY .dt.normalize():
# Different CSVs may have different time components on timestamps.
# Stock: "2021-05-03 00:00:00", Regime: "2021-05-03 09:15:00".
# These do NOT match on a join. normalize() strips time -> midnight.
#
# WHY drop_duplicates on regime:
# Protects against accidental duplicate rows in the regime CSV
# (e.g. from re-running trend.py that appended rows). Without this,
# a join on a date with 2 regime rows would create 2 stock rows.
# ================================================================

def load_and_merge(symbol):
    # Load stock features
    fp = CONFIG.paths.stock_features_path(symbol)
    if not os.path.exists(fp):
        raise FileNotFoundError(
            f"[{symbol}] Features not found: {fp}\n"
            f"  Run feature_pipeline first."
        )
    stock = pd.read_csv(fp, parse_dates=["Datetime"])
    stock["Datetime"] = pd.to_datetime(stock["Datetime"]).dt.normalize()
    stock = stock.sort_values("Datetime").reset_index(drop=True)

    # Load NIFTY regime
    rp = CONFIG.paths.nifty_regime_path()
    if not os.path.exists(rp):
        raise FileNotFoundError(
            f"[{symbol}] NIFTY regime not found: {rp}\n"
            f"  Run trend.py first."
        )
    regime = pd.read_csv(rp, parse_dates=["Datetime"])
    regime["Datetime"] = pd.to_datetime(regime["Datetime"]).dt.normalize()
    regime = (
        regime[["Datetime", "predicted_regime"]]
        .drop_duplicates(subset="Datetime", keep="first")
    )

    # Inner join on date
    merged  = pd.merge(stock, regime, on="Datetime", how="inner")
    merged  = merged.sort_values("Datetime").reset_index(drop=True)
    dropped = len(stock) - len(merged)
    if dropped > 0:
        log.warning(f"[{symbol}] {dropped} rows dropped (stock dates with no NIFTY regime)")

    log.info(
        f"[{symbol}] Merged {len(merged):,} rows | "
        f"{merged['Datetime'].min().date()} -> {merged['Datetime'].max().date()}"
    )
    return merged


# ================================================================
# STEP 3  EXTRACT TEST SET
# ================================================================
# WHY SAME SPLIT MATH AS ema_tuner.py:
# The tuner used int(n*0.60) and int(n*0.20) with floor rounding.
# Using the same formula here guarantees the rows here are exactly
# the same rows the tuner never touched. Any different rounding
# would risk evaluating on rows the tuner already optimised on.
#
# Note: merged n may differ slightly from tuner's n (inner join
# dropped a few rows). This is minor and unavoidable.
# ================================================================

def get_test_set(df, symbol=""):
    n       = len(df)
    n_train = int(n * CONFIG.tuning.train_ratio)   # 60%
    n_val   = int(n * CONFIG.tuning.val_ratio)     # 20%
    df_test = df.iloc[n_train + n_val:].copy().reset_index(drop=True)
    log.info(
        f"[{symbol}] Test set: {len(df_test):,} rows | "
        f"{df_test['Datetime'].min().date()} -> {df_test['Datetime'].max().date()}"
    )
    return df_test


# ================================================================
# STEP 4  BACKTEST ENGINE
# ================================================================
# STATE MACHINE
# -------------
# idle     : waiting for crossover  (in_trade=False, wait_retest=False)
# watching : crossover seen, waiting for retest  (wait_retest=True)
# in_trade : position open, monitoring SL/TP  (in_trade=True)
#
# Transitions:
#   idle     --[crossover, regime != DOWN]--> watching
#   watching --[retest hit, i+1 exists]-----> in_trade
#   watching --[bars >= max_bars]-----------> idle (timeout, skip trade)
#   watching --[regime turns DOWN]----------> idle (cancel setup)
#   in_trade --[SL / TP / EOD]-------------> idle (and possibly watching same bar)
#
# BAR-BY-BAR FLOW
# ---------------
# [A] If in_trade: check SL/TP/EOD
#     If exit triggered: record trade, reset state, FALL THROUGH to [B][C][D][E]
#     If no exit: append equity, CONTINUE to next bar
# [B] Append equity to curve
# [C] Regime gate: DOWN -> continue (skip D and E)
# [D] If not wait_retest: check for crossover, then continue
# [E] If wait_retest: check retest, handle timeout
#
# WHY FALL THROUGH ON EXIT (not continue):
# On the bar we exit a trade, we immediately check for a new crossover.
# Realistic -- if a position closes at end of day and a new crossover
# formed same day, you'd spot it and start watching for retest.
#
# WHY ENTER AT NEXT BAR OPEN:
# Retest is confirmed only at Close. You can't enter at same bar Close
# because the bar hasn't closed when you decide to enter. Next bar Open
# is the earliest realistic and honest entry time.
#
# WHY SL BEFORE TP:
# If both trigger same bar (e.g. gap scenario), SL wins -- conservative.
# ================================================================

def run_backtest(df, params, symbol=""):
    ema_fast   = params["ema_fast"]
    ema_slow   = params["ema_slow"]
    sl         = params["sl"]
    tp         = params["tp"]
    retest_tol = params["retest_tol"]
    cost       = CONFIG.backtest.transaction_cost_pct
    max_bars   = CONFIG.tuning.retest_max_bars

    # Recompute EMAs fresh on test slice only.
    # WHY: EMA is recursive -- each value depends on all prior values.
    # If we used EMAs from the full dataset, test-bar EMAs would contain
    # information from training period (look-ahead from train into test).
    # Fresh computation on test slice avoids this leakage.
    fast_ema = df["Close"].ewm(span=ema_fast, adjust=False).mean()
    slow_ema = df["Close"].ewm(span=ema_slow, adjust=False).mean()

    initial_capital = CONFIG.backtest.initial_capital
    equity          = initial_capital
    peak_equity     = initial_capital

    trades        = []   # list of dicts, one per completed trade
    equity_curve  = []   # list of (date, equity), one per bar

    # State variables
    in_trade      = False   # currently holding a position
    entry_price   = 0.0     # cost-adjusted entry price
    entry_date    = None    # pd.Timestamp of entry bar
    retest_regime = None    # regime on retest confirmation bar
    entry_regime  = None    # regime on actual entry bar (next day)
    wait_retest   = False   # crossover seen, waiting for retest
    bars_waiting  = 0       # bars elapsed since crossover

    n   = len(df)
    tag = f"[{symbol}]" if symbol else "[run_backtest]"

    for i in range(1, n):

        # Read today's data
        date      = df["Datetime"].iloc[i]
        close     = df["Close"].iloc[i]
        low       = df["Low"].iloc[i]
        high      = df["High"].iloc[i]
        regime    = df["predicted_regime"].iloc[i]
        fast_val  = fast_ema.iloc[i]
        fast_prev = fast_ema.iloc[i - 1]
        slow_prev = slow_ema.iloc[i - 1]

        # --------------------------------------------------------
        # [A] IN TRADE: check SL / TP / end-of-data
        # --------------------------------------------------------
        if in_trade:
            sl_price   = entry_price * (1 - sl)
            tp_price   = entry_price * (1 + tp)
            exit_price = None
            exit_type  = None

            # SL checked before TP -- conservative priority
            if close <= sl_price:
                exit_price, exit_type = close, "SL"
            elif close >= tp_price:
                exit_price, exit_type = close, "TP"
            elif i == n - 1:
                exit_price, exit_type = close, "EOD"

            if exit_price is not None:
                # Return after round-trip transaction costs
                ret = (exit_price / entry_price) - 1 - 2 * cost
                pnl = equity * ret
                equity *= (1 + ret)
                peak_equity = max(peak_equity, equity)

                trades.append({
                    "entry_date"   : entry_date.date(),
                    "exit_date"    : date.date(),
                    "hold_days"    : (date - entry_date).days,
                    # Both regime fields for audit trail:
                    # retest_regime = regime when retest signal confirmed
                    # entry_regime  = regime when we actually entered next day
                    # They can differ if regime changed overnight
                    "retest_regime": retest_regime,
                    "entry_regime" : entry_regime,
                    "entry_price"  : round(entry_price, 2),
                    "exit_price"   : round(exit_price, 2),
                    "exit_type"    : exit_type,
                    "return_pct"   : round(ret * 100, 4),
                    "pnl"          : round(pnl, 2),
                    "equity"       : round(equity, 2),
                })

                # Reset all trade state
                in_trade      = False
                entry_price   = 0.0
                entry_date    = None
                retest_regime = None
                entry_regime  = None
                # FALL THROUGH -- allow crossover check on same bar
            else:
                # Still in trade -- update equity curve, skip signals
                equity_curve.append((date, equity))
                continue   # <-- only fires when STILL in trade

        # --------------------------------------------------------
        # [B] APPEND EQUITY
        # Fires when: not in trade | just exited (fell through)
        # Does NOT fire when: still in trade (caught by continue above)
        # --------------------------------------------------------
        equity_curve.append((date, equity))

        # --------------------------------------------------------
        # [C] REGIME GATE -- block only on DOWN
        # --------------------------------------------------------
        if regime == "DOWN":
            # Cancel pending retest setup -- market is falling,
            # the crossover signal is now invalidated
            if wait_retest:
                wait_retest  = False
                bars_waiting = 0
            continue   # skip signal detection

        # Regime is UP or SIDEWAYS -- proceed with signals

        # --------------------------------------------------------
        # [D] CROSSOVER DETECTION  (idle state only)
        # --------------------------------------------------------
        if not wait_retest:
            # Bullish crossover:
            # Yesterday fast <= slow (bearish/flat momentum)
            # Today fast > slow (momentum shifted bullish)
            # <= on left catches exact crossover including equal case
            if fast_prev <= slow_prev and fast_ema.iloc[i] > slow_ema.iloc[i]:
                wait_retest  = True
                bars_waiting = 0
            # Whether crossover or not, do NOT check retest same bar
            continue

        # --------------------------------------------------------
        # [E] RETEST WAIT  (watching state)
        # --------------------------------------------------------
        if wait_retest:
            bars_waiting += 1

            if retest_tol is None:
                # Strict: candle must physically touch the EMA line
                # If Low <= EMA <= High, the candle passed through EMA
                retest_hit = (low <= fast_val <= high)
            else:
                # Relaxed: price must come within retest_tol% of EMA
                # upper_band = the "close enough" boundary above EMA
                # Cond 1: Low reached or went below upper_band
                # Cond 2: Close still above EMA*(1-tol) -> trend intact
                upper_band = fast_val * (1 + retest_tol)
                retest_hit = (
                    (low <= upper_band) and
                    (close > fast_val * (1 - retest_tol))
                )

            if retest_hit:
                # Enter at NEXT bar Open -- earliest honest entry
                # Guard: if retest is on the last bar, skip entry
                if i + 1 < n:
                    raw_open      = df["Open"].iloc[i + 1]
                    entry_price   = raw_open * (1 + cost)   # entry cost applied
                    entry_date    = pd.Timestamp(df["Datetime"].iloc[i + 1])
                    retest_regime = regime                               # today's regime
                    entry_regime  = df["predicted_regime"].iloc[i + 1]  # tomorrow's regime
                    in_trade      = True

                wait_retest  = False
                bars_waiting = 0

            elif bars_waiting >= max_bars:
                # Timeout: stock moved too far from EMA without retest.
                # Chasing would mean buying at extended price with poor R:R.
                # Abandon this crossover signal.
                wait_retest  = False
                bars_waiting = 0
            # If neither hit nor timed out: keep waiting next bar

    # ---- Build outputs ----------------------------------------
    if trades:
        trades_df = pd.DataFrame(trades)
    else:
        # Empty DataFrame with EXACT same schema as filled version.
        # This ensures compute_metrics and save_trade_log work the
        # same way whether 0 trades or 100 trades were generated.
        trades_df = pd.DataFrame(columns=[
            "entry_date", "exit_date", "hold_days",
            "retest_regime", "entry_regime",
            "entry_price", "exit_price", "exit_type",
            "return_pct", "pnl", "equity",
        ])

    eq_series = pd.Series(
        data  = [e for _, e in equity_curve],
        index = [d for d, _ in equity_curve],
        name  = "equity",
    )

    final_return = (equity - initial_capital) / initial_capital * 100
    log.info(
        f"{tag} Done | Trades={len(trades_df)} | "
        f"Equity=Rs{equity:,.0f} | Return={final_return:+.1f}%"
    )
    return trades_df, eq_series


# ================================================================
# STEP 5  COMPUTE METRICS
# ================================================================
# METRICS EXPLAINED
# -----------------
# total_return    : % gain/loss on capital over test period
# buy_hold_return : passive benchmark -- just hold the stock
# alpha           : total_return - buy_hold_return  (computed in report)
# sharpe          : sqrt(252) * mean_excess_return / std_returns
#                   > 1.0 good, > 2.0 excellent
# max_drawdown    : worst peak-to-trough equity drop %
# win_rate        : % of trades with positive return after costs
# avg_profit_pct  : avg return on winning trades (how much you make right)
# avg_loss_pct    : avg return on losing trades (how much you lose wrong)
# avg_hold_days   : avg calendar days per trade
# ================================================================

def compute_metrics(trades_df, eq_series, df_test):
    bh_return = (
        (df_test["Close"].iloc[-1] - df_test["Close"].iloc[0])
        / df_test["Close"].iloc[0] * 100
    )

    if len(trades_df) == 0:
        log.warning("No trades generated. Check regime coverage and params.")
        return {
            "total_return"   : 0.0,
            "sharpe"         : 0.0,
            "max_drawdown"   : 0.0,
            "win_rate"       : 0.0,
            "total_trades"   : 0,
            "avg_profit_pct" : 0.0,
            "avg_loss_pct"   : 0.0,
            "avg_hold_days"  : 0.0,
            "buy_hold_return": round(bh_return, 2),
        }

    ret_arr = trades_df["return_pct"].values / 100
    wins    = ret_arr[ret_arr > 0]
    losses  = ret_arr[ret_arr < 0]

    # Total return from equity curve
    initial_cap = CONFIG.backtest.initial_capital
    final_eq    = eq_series.iloc[-1] if len(eq_series) > 0 else initial_cap
    total_ret   = (final_eq - initial_cap) / initial_cap

    # Annualised Sharpe using per-trade returns
    rf_per_trade = CONFIG.backtest.risk_free_rate / CONFIG.backtest.annualization_factor
    excess       = ret_arr - rf_per_trade
    sharpe = (
        float(np.sqrt(CONFIG.backtest.annualization_factor) * excess.mean() / ret_arr.std())
        if ret_arr.std() > 1e-9 else 0.0
    )

    # Max drawdown from equity curve (not from trade list)
    eq_vals = eq_series.values
    if len(eq_vals) > 0:
        running_max = np.maximum.accumulate(eq_vals)
        drawdowns   = (running_max - eq_vals) / running_max
        max_dd      = float(drawdowns.max())
    else:
        max_dd = 0.0

    return {
        "total_return"   : round(total_ret * 100, 2),
        "sharpe"         : round(sharpe, 4),
        "max_drawdown"   : round(max_dd * 100, 2),
        "win_rate"       : round(float(np.mean(ret_arr > 0)) * 100, 2),
        "total_trades"   : len(trades_df),
        "avg_profit_pct" : round(float(wins.mean()   * 100), 4) if len(wins)   > 0 else 0.0,
        "avg_loss_pct"   : round(float(losses.mean() * 100), 4) if len(losses) > 0 else 0.0,
        "avg_hold_days"  : round(float(trades_df["hold_days"].mean()), 1),
        "buy_hold_return": round(bh_return, 2),
    }


# ================================================================
# STEP 6  SAVE OUTPUTS
# ================================================================

def save_trade_log(trades_df, symbol):
    """
    Save trade-by-trade CSV audit trail.
    Columns: entry_date, exit_date, hold_days, retest_regime,
    entry_regime, entry_price, exit_price, exit_type, return_pct,
    pnl, equity.
    Use this to debug: e.g. if all losses have entry_regime=SIDEWAYS,
    consider restricting entries to UP-only.
    """
    path = CONFIG.paths.trade_log_path(symbol)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    trades_df.to_csv(path, index=False)
    log.info(f"[{symbol}] Trade log -> {path}  ({len(trades_df)} trades)")


def save_equity_curve(eq_series, df_test, symbol, params, metrics):
    """
    Two-panel PNG chart.
    Top   : Strategy (blue) vs Buy-and-Hold (gray dashed), both at base 100.
    Bottom: NIFTY regime bar -- green=UP, gray=SIDEWAYS, red=DOWN.
    """
    path = CONFIG.paths.equity_curve_path(symbol)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(16, 9),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )
    fig.patch.set_facecolor("#f8f9fa")

    eq_vals = eq_series.values
    eq_norm = eq_vals / eq_vals[0] * 100 if len(eq_vals) > 0 else []
    bh      = df_test.set_index("Datetime")["Close"]
    bh_norm = bh / bh.iloc[0] * 100

    # Top panel
    ax1.set_facecolor("#f8f9fa")
    ax1.plot(eq_series.index, eq_norm, color="#2980b9", linewidth=1.8,
             label="Strategy", zorder=3)
    ax1.plot(bh.index, bh_norm, color="#95a5a6", linewidth=1.2,
             linestyle="--", label="Buy & Hold", alpha=0.85, zorder=2)
    ax1.axhline(100, color="#2c3e50", linewidth=0.6, linestyle=":", alpha=0.5)
    ax1.set_title(
        f"{symbol}  EMA({params['ema_fast']},{params['ema_slow']})  "
        f"SL={params['sl']*100:.0f}%  TP={params['tp']*100:.0f}%  "
        f"Retest={params['retest_tol']}  |  "
        f"Return={metrics['total_return']:+.1f}%  "
        f"Sharpe={metrics['sharpe']:.2f}  "
        f"MaxDD={metrics['max_drawdown']:.1f}%  "
        f"WinRate={metrics['win_rate']:.1f}%  "
        f"Trades={metrics['total_trades']}",
        fontsize=10, fontweight="bold", pad=10,
    )
    ax1.set_ylabel("Portfolio Value (Base = 100)", fontsize=10)
    ax1.legend(loc="upper left", framealpha=0.9, fontsize=9)
    ax1.grid(True, alpha=0.25, linestyle="--")
    ax1.spines[["top", "right"]].set_visible(False)

    # Bottom panel -- regime
    ax2.set_facecolor("#f8f9fa")
    regime_s = df_test.set_index("Datetime")["predicted_regime"]
    colors   = regime_s.map({"UP": "#27ae60", "SIDEWAYS": "#95a5a6", "DOWN": "#e74c3c"})
    ax2.bar(regime_s.index, 1, color=colors, alpha=0.7, width=1.5)
    ax2.legend(
        handles=[
            Patch(color="#27ae60", alpha=0.7, label="UP"),
            Patch(color="#95a5a6", alpha=0.7, label="SIDEWAYS"),
            Patch(color="#e74c3c", alpha=0.7, label="DOWN"),
        ],
        loc="upper right", fontsize=8, framealpha=0.8,
    )
    ax2.set_ylabel("Regime", fontsize=9)
    ax2.set_yticks([])
    ax2.set_ylim(0, 1.3)
    ax2.grid(False)
    ax2.spines[["top", "right", "left"]].set_visible(False)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=35, ha="right", fontsize=8)
    plt.tight_layout(h_pad=0.5)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    log.info(f"[{symbol}] Equity curve -> {path}")


def save_performance_report(metrics, params, symbol, df_test):
    """
    Human-readable TXT report. Also printed to console immediately.
    Shows: test period, params, val reference, full test results.
    """
    path  = CONFIG.paths.performance_report_path(symbol)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    alpha = metrics["total_return"] - metrics["buy_hold_return"]
    W     = 26

    lines = [
        "=" * 64,
        f"  {symbol}  --  EMA Strategy  --  TEST SET RESULTS",
        "=" * 64,
        "",
        "  TEST PERIOD",
        f"  {'From':<{W}} : {df_test['Datetime'].min().date()}",
        f"  {'To':<{W}} : {df_test['Datetime'].max().date()}",
        f"  {'Total Trading Days':<{W}} : {len(df_test):,}",
        "",
        "  PARAMETERS  (selected by tuner on val set)",
        f"  {'EMA Fast Period':<{W}} : {params['ema_fast']}",
        f"  {'EMA Slow Period':<{W}} : {params['ema_slow']}",
        f"  {'Stop Loss':<{W}} : {params['sl']*100:.1f}%",
        f"  {'Risk-Reward':<{W}} : 1:{params['rr']}",
        f"  {'Take Profit':<{W}} : {params['tp']*100:.1f}%",
        f"  {'Retest Tolerance':<{W}} : {params['retest_tol']}",
        "",
        "  VAL SET REFERENCE  (tuner optimised on this, not test)",
        f"  {'Val Sharpe':<{W}} : {params.get('val_sharpe', 'N/A')}",
        f"  {'Val Max Drawdown':<{W}} : {params.get('val_drawdown', 0)*100:.1f}%",
        f"  {'Val Total Trades':<{W}} : {params.get('val_trades', 'N/A')}",
        f"  {'Train Sharpe':<{W}} : {params.get('train_sharpe', 'N/A')}",
        "",
        "  TEST SET RESULTS  (honest unseen evaluation)",
        f"  {'Strategy Total Return':<{W}} : {metrics['total_return']:+.2f}%",
        f"  {'Buy and Hold Return':<{W}} : {metrics['buy_hold_return']:+.2f}%",
        f"  {'Alpha vs Buy and Hold':<{W}} : {alpha:+.2f}%",
        f"  {'Sharpe Ratio':<{W}} : {metrics['sharpe']:.4f}",
        f"  {'Max Drawdown':<{W}} : {metrics['max_drawdown']:.2f}%",
        f"  {'Win Rate':<{W}} : {metrics['win_rate']:.2f}%",
        f"  {'Total Trades':<{W}} : {metrics['total_trades']}",
        f"  {'Avg Profit per Trade':<{W}} : {metrics['avg_profit_pct']:+.4f}%",
        f"  {'Avg Loss per Trade':<{W}} : {metrics['avg_loss_pct']:+.4f}%",
        f"  {'Avg Hold Days':<{W}} : {metrics['avg_hold_days']:.1f} days",
        "",
        "=" * 64,
    ]
    content = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("\n" + content)
    log.info(f"[{symbol}] Performance report -> {path}")


# ================================================================
# STEP 7  PER-SYMBOL ORCHESTRATION
# ================================================================

def run_symbol(symbol):
    """
    Full pipeline for one stock: params -> merge -> test -> backtest
    -> metrics -> save. Error per symbol is caught so one broken stock
    does not kill the entire run.
    """
    log.info(f"{'─'*64}")
    log.info(f"  Backtesting: {symbol}")
    log.info(f"{'─'*64}")
    try:
        params              = load_best_params(symbol)
        df                  = load_and_merge(symbol)
        df_test             = get_test_set(df, symbol=symbol)
        trades_df, eq_series = run_backtest(df_test, params, symbol=symbol)
        metrics             = compute_metrics(trades_df, eq_series, df_test)
        save_trade_log(trades_df, symbol)
        save_equity_curve(eq_series, df_test, symbol, params, metrics)
        save_performance_report(metrics, params, symbol, df_test)
        return metrics
    except (FileNotFoundError, ValueError) as e:
        log.error(f"[{symbol}] Setup error: {e}")
        return None
    except Exception as e:
        log.error(f"[{symbol}] Unexpected error: {e}", exc_info=True)
        return None


# ================================================================
# STEP 8  MAIN ENTRY
# ================================================================

def run_strategy(symbols=None):
    """
    Run backtest for all stocks or a subset.
    Called by main.py (pipeline) or directly (standalone).
    """
    symbols = [s.upper() for s in (symbols or STOCK_UNIVERSE)]
    log.info("=" * 64)
    log.info("  EMA Strategy -- Final Test Set Evaluation")
    log.info(f"  Symbols : {symbols}")
    log.info(f"  Gate    : DOWN=blocked | UP=trade | SIDEWAYS=trade")
    log.info("=" * 64)

    all_metrics = {}
    succeeded   = []
    failed      = []

    for symbol in symbols:
        metrics = run_symbol(symbol)
        all_metrics[symbol] = metrics
        (succeeded if metrics is not None else failed).append(symbol)

    # Summary table
    print(f"\n{'='*82}")
    print(f"  {'Symbol':<12} {'Return':>8} {'B&H':>8} {'Alpha':>8} "
          f"{'Sharpe':>8} {'MaxDD':>7} {'WinRate':>8} {'Trades':>7}")
    print(f"  {'-'*78}")
    for sym, m in all_metrics.items():
        if m is not None:
            alpha = m["total_return"] - m["buy_hold_return"]
            print(
                f"  {sym:<12} {m['total_return']:>+7.1f}%  "
                f"{m['buy_hold_return']:>+6.1f}%  {alpha:>+6.1f}%  "
                f"{m['sharpe']:>7.2f}  {m['max_drawdown']:>6.1f}%  "
                f"{m['win_rate']:>7.1f}%  {m['total_trades']:>6}"
            )
        else:
            print(f"  {sym:<12} Failed -- see logs")
    print(f"{'='*82}")
    print(f"  Succeeded: {len(succeeded)} | Failed: {len(failed)}")
    if failed:
        print(f"  Failed: {failed}")
    print(f"{'='*82}\n")
    return all_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EMA Strategy -- Final Backtest on Unseen Test Set"
    )
    parser.add_argument(
        "--symbol", type=str, default=None,
        help="Single symbol e.g. RELIANCE. Omit to run all 9 stocks."
    )
    args    = parser.parse_args()
    symbols = [args.symbol.upper()] if args.symbol else None
    run_strategy(symbols=symbols)