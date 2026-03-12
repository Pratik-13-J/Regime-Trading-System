"""
config.py — Central Configuration for Regime-Aware Trading System
==================================================================
Single source of truth for ALL parameters across the pipeline.

Design:
    - Dataclasses for type safety + IDE autocomplete
    - Credentials from environment variables (never hardcoded)
    - EMA pairs generated dynamically (no hardcoded list)
    - TP always derived from SL × RR ratio (risk-reward based)
    - Validation runs at import time — bad config fails loudly

Usage:
    from config import CONFIG

    CONFIG.angel.api_key
    CONFIG.data.start_date
    CONFIG.regime.label_threshold
    CONFIG.tuning.rr_ratios
    CONFIG.backtest.initial_capital
    CONFIG.paths.stock_raw_path("RELIANCE")
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
load_dotenv()  

# ═══════════════════════════════════════════════════════════════════
# 1. ANGEL ONE API CREDENTIALS
# ═══════════════════════════════════════════════════════════════════

@dataclass
class AngelOneConfig:
    """
    Angel One SmartAPI credentials loaded from environment variables.

    Set in terminal before running:
        export ANGEL_API_KEY="your_key"
        export ANGEL_CLIENT_ID="your_client_id"
        export ANGEL_PASSWORD="your_password"
        export ANGEL_TOTP_SECRET="your_totp_secret"
    """
    api_key     : str = field(default_factory=lambda: os.environ.get("ANGEL_API_KEY",      ""))
    client_id   : str = field(default_factory=lambda: os.environ.get("ANGEL_CLIENT_ID",    ""))
    password    : str = field(default_factory=lambda: os.environ.get("ANGEL_PASSWORD",     ""))
    totp_secret : str = field(default_factory=lambda: os.environ.get("ANGEL_TOTP_SECRET",  ""))

    def validate(self):
        missing = [k for k, v in {
            "ANGEL_API_KEY"    : self.api_key,
            "ANGEL_CLIENT_ID"  : self.client_id,
            "ANGEL_PASSWORD"   : self.password,
            "ANGEL_TOTP_SECRET": self.totp_secret,
        }.items() if not v]
        if missing:
            raise EnvironmentError(
                f"\n[Config] Missing Angel One credentials: {missing}"
                f"\nSet them via: export ANGEL_API_KEY='...' etc."
            )


# ═══════════════════════════════════════════════════════════════════
# 2. DATA CONFIG
# ═══════════════════════════════════════════════════════════════════

@dataclass
class DataConfig:
    """
    Data fetching and storage parameters.

    nifty_symbol      : Angel One token for NIFTY 50 index
    exchange          : NSE / BSE
    timeframe         : Candle interval for Angel One API
    start_date        : Historical data start date
    end_date          : None = fetch until today
    min_rows_required : Minimum rows before pipeline proceeds
                        250 covers EMA warm-up + enough history
    """
    nifty_symbol      : str           = "99926000"   # Angel One NIFTY 50 token
    exchange          : str           = "NSE"
    timeframe         : str           = "ONE_DAY"
    start_date        : str           = "2015-01-01"
    end_date          : Optional[str] = None          # None = today
    date_format       : str           = "%Y-%m-%d"
    min_rows_required : int           = 250


# ═══════════════════════════════════════════════════════════════════
# 3. NIFTY REGIME (ML) CONFIG
# ═══════════════════════════════════════════════════════════════════

@dataclass
class RegimeConfig:
    """
    NIFTY ML regime detection parameters (trend.py).

    label_threshold   : Next-day return threshold for UP/DOWN vs SIDEWAYS
                        ±0.2% — returns inside this band → SIDEWAYS label

    n_cv_splits       : Walk-forward TimeSeriesSplit folds for model CV

    trade_on_regimes  : Only enter trades when NIFTY predicted regime
                        is in this list. ["UP"] = long-only system.
    """
    label_threshold  : float = 0.003    # ±0.3%
    n_cv_splits      : int   = 5

     # ── Label encoding — MUST match create_labels() in trend.py ──
    # 0=DOWN, 1=SIDEWAYS, 2=UP — never change these numbers
    regime_labels : dict = field(default_factory=lambda: {
        0 : "DOWN",
        1 : "SIDEWAYS",
        2 : "UP",
    })

    rf_params : dict = field(default_factory=lambda: {
        "n_estimators"    : 300,
        "max_depth"       : 6,
        "min_samples_split" : 40,
        "min_samples_leaf": 20,
        "class_weight"    : "balanced",
        "random_state"    : 42,
        "n_jobs"          : -1,
    })

    xgb_params : dict = field(default_factory=lambda: {
        "n_estimators"    : 300,
        "max_depth"       : 4,
        "learning_rate"   : 0.05,
        "subsample"       : 0.8,
        "colsample_bytree": 0.8,
        "eval_metric"     : "mlogloss",
        "random_state"    : 42,
        "verbosity"       : 0,
    })

    trade_on_regimes : List[str] = field(default_factory=lambda: ["UP"])


# ═══════════════════════════════════════════════════════════════════
# 4. FEATURE ENGINEERING CONFIG
# ═══════════════════════════════════════════════════════════════════

@dataclass
class FeatureConfig:
    """
    Technical indicator parameters for stock feature engineering.
    All indicators use strictly past data — no look-ahead.
    """
    ema_periods       : List[int] = field(default_factory=lambda: [9, 21, 50, 200])
    rsi_period        : int       = 14
    macd_fast         : int       = 12
    macd_slow         : int       = 26
    macd_signal       : int       = 9
    bb_period         : int       = 20
    bb_std            : float     = 2.0
    atr_period        : int       = 14
    volume_ma_period  : int       = 20
    return_periods    : List[int] = field(default_factory=lambda: [1, 3, 5, 10])
    volatility_period : int       = 10


# ═══════════════════════════════════════════════════════════════════
# 5. HYPERPARAMETER TUNING CONFIG
# ═══════════════════════════════════════════════════════════════════

@dataclass
class TuningConfig:
    """
    Hyperparameter search space and optimization logic.

    EMA PAIRS
    ---------
    All (fast, slow) pairs where:
        fast ∈ [5, 45],  slow ∈ [fast+5, 50],  gap ≥ 5
    ~330 valid pairs generated dynamically.

    STOP LOSS
    ---------
    sl_range : 2%, 3%, 5% — typical values for swing trading.

    TAKE PROFIT — Risk-Reward Based
    --------------------------------
    TP is NEVER set independently.
    TP = SL × RR ratio.
    rr_ratios : [3, 4]  →  1:3, 1:4
    Example: SL=2%, RR=1:3 → TP=6%

    RETEST TOLERANCE
    ----------------
    Price must pull back within X% of fast EMA after crossover.
    None = disabled (enter immediately on crossover).

    TOTAL COMBINATIONS
    ------------------
    ~330 EMA pairs × 5 SL × 3 RR × 4 retest = ~19,800
    (fast to run — each backtest is just vectorized pandas ops)

    OPTIMIZATION OBJECTIVE
    ----------------------
    Score = Sharpe − lambda × Max Drawdown
    Lambda = 1.0 by default.

    HARD FILTERS
    ------------
    max_drawdown_limit : 15% hard cap — very strict.
                         Protects against consecutive losing streaks.
                         (Per-trade loss capped by SL, but 10 consecutive
                          losses at 2% SL = ~18% portfolio drawdown.)
    min_trades         : 10 — discard statistically unreliable combos.

    OVERFITTING GUARD
    -----------------
    val_sharpe ≥ overfitting_threshold × train_sharpe
    Default 0.5 — val Sharpe must be at least half of train Sharpe.

    DATA SPLIT (time-ordered, no shuffle)
    --------------------------------------
    Train : 60%  →  strategy learns here
    Val   : 20%  →  best params selected here
    Test  : 20%  →  LOCKED, touched exactly once at the very end

    BEST PARAMS
    -----------
    Stored per symbol after tuning. Each stock gets its own optimal
    params since volatility profiles differ across stocks.
    """

    # ── EMA search space ──
    ema_fast_min : int = 5
    ema_fast_max : int = 45
    ema_slow_max : int = 50
    ema_min_gap  : int = 5

    # ── Stop loss ──
    sl_range : List[float] = field(
    default_factory=lambda: [0.02, 0.03, 0.05]
    )

    # ── Risk-reward ratios (TP = SL × RR) ──
    rr_ratios : List[int] = field(default_factory=lambda: [3, 4])

    # ── Retest tolerance ──
    retest_range    : List[Optional[float]] = field(
    default_factory=lambda: [None, 0.015]
    )
    retest_max_bars : int = 5

    # ── Optimization objective ──
    drawdown_penalty_lambda : float = 1.0

    # ── Hard filters ──
    max_drawdown_limit : float = 0.15   # 15% hard cap
    min_trades         : int   = 10

    # ── Overfitting guard ──
    overfitting_threshold : float = 0.5

    # ── Data split ratios ──
    train_ratio : float = 0.60
    val_ratio   : float = 0.20
    test_ratio  : float = 0.20

    # ── Best params per symbol (populated after tuning) ──
    best_params : Dict[str, Any] = field(default_factory=dict)

    # ──────────────────────────────────────────────
    # METHODS
    # ──────────────────────────────────────────────

    def generate_ema_pairs(self) -> List[tuple]:
        """
        Dynamically generate all valid (fast, slow) EMA pairs.
        fast ∈ [ema_fast_min, ema_fast_max]
        slow ∈ [fast + ema_min_gap, ema_slow_max]
        """
        pairs = []
        for fast in range(self.ema_fast_min, self.ema_fast_max + 1, 3):
            for slow in range(fast + self.ema_min_gap, self.ema_slow_max + 1, 3):
                pairs.append((fast, slow))
        return pairs

    def compute_take_profit(self, sl: float, rr: int) -> float:
        """
        TP = SL × RR.  Always derived — never set independently.
        """
        return round(sl * rr, 4)

    def get_all_combinations(self) -> List[Dict]:
        """
        Full grid of all parameter combinations for tuning.
        Returns list of dicts — each dict is one backtest run.
        """
        combos = []
        for (fast, slow) in self.generate_ema_pairs():
            for sl in self.sl_range:
                for rr in self.rr_ratios:
                    tp = self.compute_take_profit(sl, rr)
                    for retest in self.retest_range:
                        combos.append({
                            "ema_fast"  : fast,
                            "ema_slow"  : slow,
                            "sl"        : sl,
                            "rr"        : rr,
                            "tp"        : tp,
                            "retest_tol": retest,
                        })
        return combos

    def compute_score(self, sharpe: float, max_drawdown: float) -> float:
        """
        Ranking score for a parameter combination.
        Score = Sharpe − lambda × Max Drawdown
        Higher = better. Penalizes large drawdowns.
        """
        return sharpe - self.drawdown_penalty_lambda * max_drawdown

    def save_best_params(self, symbol: str, params: Dict):
        """Store best params for a symbol after tuning."""
        self.best_params[symbol.upper()] = params
        print(f"[TuningConfig] ✅ Best params saved for {symbol.upper()}: {params}")

    def get_best_params(self, symbol: str) -> Optional[Dict]:
        """Retrieve best params. Returns None if tuning hasn't run."""
        return self.best_params.get(symbol.upper(), None)

    def validate(self):
        ratio_sum = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(ratio_sum - 1.0) > 1e-6:
            raise ValueError(f"Train+Val+Test ratios must sum to 1.0. Got {ratio_sum:.4f}")
        if self.ema_fast_min < 2:
            raise ValueError("ema_fast_min must be >= 2.")
        if self.ema_slow_max > 200:
            raise ValueError("ema_slow_max must be <= 200.")
        if self.ema_min_gap < 1:
            raise ValueError("ema_min_gap must be >= 1.")
        if not (0 < self.max_drawdown_limit < 1):
            raise ValueError("max_drawdown_limit must be between 0 and 1.")
        if not (0 < self.overfitting_threshold < 1):
            raise ValueError("overfitting_threshold must be between 0 and 1.")


# ═══════════════════════════════════════════════════════════════════
# 6. BACKTEST CONFIG
# ═══════════════════════════════════════════════════════════════════

@dataclass
class BacktestConfig:
    """
    Backtest engine parameters.

    initial_capital      : Starting portfolio in INR
    transaction_cost_pct : One-way cost — applied on entry AND exit
                           0.001 = 0.1% per side = 0.2% round trip
                           Covers brokerage + STT + exchange fees
    position_size_pct    : Fraction of equity deployed per trade
                           1.0 = fully invested (one position at a time)
    annualization_factor : 252 NSE trading days
    risk_free_rate       : Annual rate for Sharpe (India 10Y G-Sec ~6.5%)
    """
    initial_capital      : float = 1_000_000.0
    transaction_cost_pct : float = 0.001
    position_size_pct    : float = 1.0
    annualization_factor : int   = 252
    risk_free_rate       : float = 0.065

    def validate(self):
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive.")
        if not (0 <= self.transaction_cost_pct < 0.01):
            raise ValueError("transaction_cost_pct seems unusually high.")
        if not (0 < self.position_size_pct <= 1.0):
            raise ValueError("position_size_pct must be between 0 and 1.")


# ═══════════════════════════════════════════════════════════════════
# 7. PATH CONFIG
# ═══════════════════════════════════════════════════════════════════

@dataclass
class PathConfig:
    """All file paths centralized. Dirs auto-created on first run."""

    raw_dir       : str = "data/raw"
    processed_dir : str = "data/processed"
    results_dir   : str = "results"
    trades_dir    : str = "results/trades"
    plots_dir     : str = "results/plots"
    reports_dir   : str = "results/reports"

    nifty_raw_file    : str = "nifty50_ohlc_daily.csv"
    nifty_regime_file : str = "nifty_regime.csv"

    def nifty_raw_path(self)    -> str:
        return os.path.join(self.raw_dir, self.nifty_raw_file)

    def nifty_regime_path(self) -> str:
        return os.path.join(self.processed_dir, self.nifty_regime_file)

    def stock_raw_path(self, symbol: str) -> str:
        return os.path.join(self.raw_dir, f"{symbol.upper()}_ohlc_daily.csv")

    def stock_features_path(self, symbol: str) -> str:
        return os.path.join(self.processed_dir, f"{symbol.upper()}_features.csv")

    def trade_log_path(self, symbol: str) -> str:
        return os.path.join(self.trades_dir, f"{symbol.upper()}_trade_log.csv")

    def equity_curve_path(self, symbol: str) -> str:
        return os.path.join(self.plots_dir, f"{symbol.upper()}_equity_curve.png")

    def regime_plot_path(self) -> str:
        return os.path.join(self.plots_dir, "nifty_regime_plot.png")

    def performance_report_path(self, symbol: str) -> str:
        return os.path.join(self.reports_dir, f"{symbol.upper()}_performance.txt")

    def tuning_results_path(self, symbol: str) -> str:
        return os.path.join(self.reports_dir, f"{symbol.upper()}_ema_tuning.csv")

    def model_report_path(self) -> str:
        return os.path.join(self.reports_dir, "nifty_model_report.txt")

    def create_all_dirs(self):
        for d in [
            self.raw_dir, self.processed_dir,
            self.results_dir, self.trades_dir,
            self.plots_dir, self.reports_dir,
        ]:
            os.makedirs(d, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
# 8. MASTER CONFIG
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Config:
    """
    Master config — single import for entire project.

        from config import CONFIG

        CONFIG.angel.api_key
        CONFIG.tuning.generate_ema_pairs()
        CONFIG.tuning.get_all_combinations()
        CONFIG.tuning.compute_score(sharpe, drawdown)
        CONFIG.tuning.save_best_params("RELIANCE", {...})
        CONFIG.tuning.get_best_params("RELIANCE")
        CONFIG.paths.stock_raw_path("RELIANCE")
    """
    angel    : AngelOneConfig = field(default_factory=AngelOneConfig)
    data     : DataConfig     = field(default_factory=DataConfig)
    regime   : RegimeConfig   = field(default_factory=RegimeConfig)
    features : FeatureConfig  = field(default_factory=FeatureConfig)
    tuning   : TuningConfig   = field(default_factory=TuningConfig)
    backtest : BacktestConfig = field(default_factory=BacktestConfig)
    paths    : PathConfig     = field(default_factory=PathConfig)

    def validate(self, check_credentials: bool = True):
        """
        Validate all config at startup.
        Pass check_credentials=False when running backtests
        on local CSV data without API connection.
        """
        if check_credentials:
            self.angel.validate()
        self.tuning.validate()
        self.backtest.validate()
        print("[Config] ✅ All configuration validated successfully.")


# ═══════════════════════════════════════════════════════════════════
# SINGLETON
# ═══════════════════════════════════════════════════════════════════

CONFIG = Config()


# ═══════════════════════════════════════════════════════════════════
# SELF-TEST — run: python config.py
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 62)
    print("   Regime-Aware Trading System — Config Verification")
    print("=" * 62)

    CONFIG.validate(check_credentials=False)

    ema_pairs  = CONFIG.tuning.generate_ema_pairs()
    all_combos = CONFIG.tuning.get_all_combinations()

    print(f"\n📅 Data range           : {CONFIG.data.start_date} → {CONFIG.data.end_date or 'today'}")
    print(f"🤖 Regime threshold     : ±{CONFIG.regime.label_threshold*100:.1f}%")
    print(f"🤖 Trade on             : {CONFIG.regime.trade_on_regimes}")

    print(f"\n🔍 EMA fast range       : {CONFIG.tuning.ema_fast_min} → {CONFIG.tuning.ema_fast_max}")
    print(f"🔍 EMA slow max         : {CONFIG.tuning.ema_slow_max}")
    print(f"🔍 Min gap              : {CONFIG.tuning.ema_min_gap}")
    print(f"🔍 Valid EMA pairs      : {len(ema_pairs)}")
    print(f"🔍 SL range             : {[f'{x*100:.0f}%' for x in CONFIG.tuning.sl_range]}")
    print(f"🔍 RR ratios            : {['1:'+str(r) for r in CONFIG.tuning.rr_ratios]}")
    print(f"🔍 Retest options       : {CONFIG.tuning.retest_range}")
    print(f"🔍 Total combinations   : {len(all_combos):,}")

    print(f"\n🎯 Drawdown penalty λ   : {CONFIG.tuning.drawdown_penalty_lambda}")
    print(f"🎯 Max DD hard limit    : {CONFIG.tuning.max_drawdown_limit*100:.0f}%  (consecutive loss protection)")
    print(f"🎯 Min trades filter    : {CONFIG.tuning.min_trades}")
    print(f"🎯 Overfit guard        : val_sharpe ≥ {CONFIG.tuning.overfitting_threshold} × train_sharpe")

    print(f"\n📊 Train / Val / Test   : "
          f"{CONFIG.tuning.train_ratio*100:.0f}% / "
          f"{CONFIG.tuning.val_ratio*100:.0f}% / "
          f"{CONFIG.tuning.test_ratio*100:.0f}%")

    print(f"\n💰 Initial capital      : ₹{CONFIG.backtest.initial_capital:,.0f}")
    print(f"💸 Transaction cost     : {CONFIG.backtest.transaction_cost_pct*100:.2f}% per side")
    print(f"📈 Risk-free rate       : {CONFIG.backtest.risk_free_rate*100:.1f}% p.a.")

    print(f"\n🧪 Score function demo  (Score = Sharpe − λ × MaxDD):")
    for sharpe, dd in [(1.5, 0.20), (1.8, 0.14), (1.6, 0.10)]:
        score  = CONFIG.tuning.compute_score(sharpe, dd)
        status = "✅ PASSES" if dd <= CONFIG.tuning.max_drawdown_limit else "❌ DISQUALIFIED (DD > 15%)"
        print(f"   Sharpe={sharpe}  DD={dd*100:.0f}%  →  Score={score:.3f}  {status}")

    print(f"\n🧪 TP derivation demo   (TP = SL × RR):")
    for sl in [0.01, 0.02, 0.03]:
        for rr in CONFIG.tuning.rr_ratios:
            tp = CONFIG.tuning.compute_take_profit(sl, rr)
            print(f"   SL={sl*100:.0f}%  ×  RR 1:{rr}  →  TP={tp*100:.0f}%")

    print(f"\n📁 Sample paths:")
    print(f"   {CONFIG.paths.nifty_raw_path()}")
    print(f"   {CONFIG.paths.stock_raw_path('RELIANCE')}")
    print(f"   {CONFIG.paths.trade_log_path('RELIANCE')}")
    print(f"   {CONFIG.paths.tuning_results_path('RELIANCE')}")

    print("\n" + "=" * 62)
    print("✅ Config ready. Run main.py --symbol RELIANCE to start.")
    print("=" * 62)