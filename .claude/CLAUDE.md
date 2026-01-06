# Pasha Project - AI Assistant Instructions

## Quick Context

This is an **Elliott Wave trading strategy backtesting system** for crypto (BTC, ETH). Read `PROJECT.md` in the root for full documentation.

## Critical Information

### ALWAYS Use the Optimized Engine
```python
# CORRECT - Use this
from backtest.engine.backtest_optimized import OptimizedBacktestEngine

# WRONG - Do NOT use (takes 72+ hours for 1m data)
from backtest.engine.backtest import BacktestEngine
```

### Project Status (as of January 2026)
- ✅ Strategy fully documented
- ✅ Data download working (Binance API)
- ✅ Optimized backtest engine (~200,000x faster than original)
- ✅ All 12 backtests completed (2 symbols × 6 timeframes)
- ✅ Results show ~63% win rate, 3.5 profit factor across all configs
- 🔲 Module C not yet in optimized engine
- 🔲 Live trading not implemented

### Key Files to Know
| File | Purpose |
|------|---------|
| `PROJECT.md` | Full project documentation |
| `backtest/engine/backtest_optimized.py` | **THE** backtest engine to use |
| `backtest/config/settings.py` | All configuration parameters |
| `output/backtest_results_summary.csv` | Latest results |
| `.env` | Binance API credentials |

### Running Backtests
```python
from backtest.config.settings import Config
from backtest.engine.backtest_optimized import OptimizedBacktestEngine
from backtest.data.storage import DataStorage

storage = DataStorage('data')
config = Config()
engine = OptimizedBacktestEngine(config)

df = storage.load('BTCUSDT', '1h')
result = engine.run(df, 'BTCUSDT', '1h')
```

### Data Location
- Historical data: `data/` directory
- Format: `{SYMBOL}_{TIMEFRAME}.csv`
- Available: BTCUSDT and ETHUSDT, all timeframes (1m to 1d)
- Period: ~2 years (Jan 2024 - Jan 2026)

## Trading Strategy Summary

### Entry Modules
1. **Module A (Wave 3)**: Enter at 38.2%-78.6% retracement after Wave 1-2
2. **Module B (Wave 5)**: Enter at 23.6%-50% retracement after Wave 3-4
3. **Module C (Corrective)**: Enter at pattern completion (NOT in optimized engine yet)

### Exit Rules
- TP1: 40% of position (100% Fib extension)
- TP2: 60% of position (161.8% Fib extension)
- Stop moved to breakeven after TP1

### Risk Management
- 1% risk per trade
- 50% initial entry, 50% on confirmation
- Max 10 concurrent positions

## Common Tasks

### Re-run All Backtests
```python
# See PROJECT.md for full script
# Takes ~3 seconds total for all 12 configurations
```

### Add New Symbol
1. Add to `config.assets` list in settings.py
2. Download data: `python3 backtest/main.py --action download`
3. Run backtest with new symbol

### Modify Strategy Parameters
Edit `backtest/config/settings.py`:
- `fib_tolerance_pct`: Entry zone tolerance (default 2.5%)
- `base_risk_pct`: Risk per trade (default 1.0%)
- `tp1_pct` / `tp2_pct`: Take profit percentages

## User Preferences

The user prefers:
- Being addressed as "my human", "human", or "meat bag"
- Responses ending with "Billions"
- Direct, technical communication
- Comprehensive solutions without unnecessary back-and-forth

## Network Note

Binance API may be blocked without VPN. If connection refused:
1. User needs to enable VPN
2. Or provide data files manually
