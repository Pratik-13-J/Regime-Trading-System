# Regime-Aware EMA Trading System

A machine learning-powered algorithmic trading system that combines NIFTY 50 market regime detection with optimized EMA crossover strategies for Indian equity markets.

## 🎯 Overview

This system uses a two-layer approach:

1. **ML Regime Detection** - XGBoost/RandomForest models predict daily NIFTY market state (UP/SIDEWAYS/DOWN)
2. **Regime-Gated EMA Strategy** - EMA crossover trades execute only during favorable market regimes

## 🏗️ Architecture

```
Data Fetching → Feature Engineering → Regime Prediction → Parameter Tuning → Backtesting
```

### Pipeline Components

| Step | Module                | Purpose                                         |
| ---- | --------------------- | ----------------------------------------------- |
| 1    | `fetch_stocks.py`     | Download OHLCV data via Angel One API           |
| 2    | `feature_pipeline.py` | Generate 22+ technical indicators               |
| 3    | `trend.py`            | Train ML models for NIFTY regime classification |
| 4    | `ema_tuner.py`        | Grid search optimal EMA parameters per stock    |
| 5    | `ema_strategy.py`     | Final backtest on unseen test set               |

## 📊 Features

- **Regime Classification**: 3-class prediction (UP/SIDEWAYS/DOWN) using 22 technical features
- **Walk-Forward Optimization**: Time-series cross-validation prevents look-ahead bias
- **Retest Logic**: Waits for pullback to EMA before entry (reduces false signals)
- **Risk Management**: Dynamic stop-loss/take-profit with configurable risk-reward ratios
- **Transaction Costs**: 0.15% per trade leg (realistic slippage + brokerage)
- **No Data Leakage**: Strict train/val/test split (60/20/20)

## 🚀 Quick Start

### Prerequisites

```bash
pip install pandas numpy scikit-learn xgboost matplotlib python-dotenv
```

### Setup

1. **Configure Angel One API credentials** (`.env` file):

```env
ANGEL_API_KEY=your_api_key
ANGEL_CLIENT_ID=your_client_id
ANGEL_PASSWORD=your_password
ANGEL_TOTP_SECRET=your_totp_secret
```

2. **Run full pipeline**:

```bash
python main.py
```

3. **Run specific components**:

```bash
# Single stock
python main.py --symbol RELIANCE

# Skip regime training (use existing)
python main.py --skip-regime

# Force re-run everything
python main.py --force
```

## 📁 Project Structure

```
regime_trading_system/
├── main.py                      # Master pipeline orchestrator
├── config.py                    # Centralized configuration
├── .env                         # API credentials (not committed)
│
├── src/
│   ├── data/
│   │   ├── fetcher.py          # Angel One API wrapper
│   │   └── fetch_stocks.py     # Stock data downloader
│   ├── features/
│   │   └── feature_pipeline.py # Technical indicator generator
│   ├── regime/
│   │   └── trend.py            # NIFTY ML regime detector
│   └── strategy/
│       ├── ema_tuner.py        # Parameter optimizer
│       └── ema_strategy.py     # Final backtest engine
│
├── data/
│   ├── raw/                    # OHLCV CSV files
│   └── processed/              # Feature CSVs + regime predictions
│
└── results/
    ├── trades/                 # Trade logs (entry/exit/PnL)
    ├── plots/                  # Equity curves + regime plots
    └── reports/                # Performance metrics
```

## 🔬 Strategy Logic

### Entry Conditions (ALL must be true)

1. ✅ NIFTY regime = UP or SIDEWAYS (DOWN blocks all entries)
2. ✅ EMA crossover detected (fast > slow)
3. ✅ Price retests fast EMA within `retest_max_bars` candles
4. ✅ Entry at next bar open

### Exit Conditions (FIRST triggered)

- 🛑 Stop Loss: Close ≤ Entry × (1 - SL%)
- 🎯 Take Profit: Close ≥ Entry × (1 + TP%)
- 📅 End of Day: Position closed at market close

## 📈 Performance Highlights

| Stock    | Return | B&H    | Alpha  | Sharpe | Max DD | Win Rate | Trades |
| -------- | ------ | ------ | ------ | ------ | ------ | -------- | ------ |
| RELIANCE | +5.7%  | +17.3% | -11.6% | 2.04   | 10.7%  | 36.4%    | 11     |
| INFY     | +12.8% | +11.2% | +1.6%  | 3.21   | 8.5%   | 50.0%    | 14     |
| HDFCBANK | +8.9%  | +13.1% | -4.2%  | 2.67   | 9.3%   | 42.9%    | 12     |
| WIPRO    | +6.4%  | +9.7%  | -3.3%  | 2.13   | 11.2%  | 40.0%    | 10     |

_Test set: 545 days (Dec 2023 - Feb 2026)_

## ⚙️ Configuration

Key parameters in `config.py`:

```python
# Regime Detection
LABEL_THRESHOLD = 0.002  # ±0.2% for SIDEWAYS classification

# EMA Grid Search
EMA_FAST_PERIODS = [5, 8, 13, 21]
EMA_SLOW_PERIODS = [21, 31, 50, 89]
STOP_LOSS_PCT = [0.01, 0.015, 0.02, 0.025]
RR_RATIOS = [2, 2.5, 3]

# Backtest
INITIAL_CAPITAL = 100000
TRANSACTION_COST = 0.0015  # 0.15% per leg
```

## 🧪 Manual Logic Review - Checkpoints

✅ **Data Integrity**

- OHLCV CSV files present for all 9 stocks
- Feature engineering produces 22+ indicators
- No NaN/inf values in feature matrices

✅ **ML Model**

- NIFTY regime model trains on 60% data
- Validation on 20% (hyperparameter tuning)
- Test set (20%) completely unseen

✅ **No Look-Ahead Bias**

- Labels created with `shift(-1)` for next-day returns
- Features use only historical data
- Train/val/test splits maintain temporal order

✅ **Trade Execution Logic**

- Retest confirmation before entry
- Regime gate blocks DOWN-day entries
- SL/TP calculated correctly with RR ratio

✅ **Risk Management**

- Transaction costs applied on both legs
- Position sizing within capital limits
- Drawdown tracking accurate

## 📝 Outputs

### Trade Logs (`results/trades/`)

- Entry/exit timestamps
- Prices, PnL, holding period
- Regime state at entry

### Performance Reports (`results/reports/`)

- Total return, Sharpe ratio, max drawdown
- Win rate, avg profit/loss per trade
- Comparison vs buy-and-hold

### Visualizations (`results/plots/`)

- Equity curve progression
- NIFTY regime classification timeline

## 🛠️ Dependencies

```
pandas
numpy
scikit-learn
xgboost
matplotlib
python-dotenv
```
