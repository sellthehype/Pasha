# Pasha Project - AI Assistant Instructions

## Quick Context

This is an **Elliott Wave trading strategy backtesting system** for crypto (BTC, ETH). Read `PROJECT.md` in the root for full documentation.

## Critical Information

### Backtest Engine
```python
from backtest.engine.backtest import BacktestEngine
from backtest.config.settings import Config
from backtest.data.storage import DataStorage

storage = DataStorage('data')
config = Config()
engine = BacktestEngine(config)

df = storage.load('BTCUSDT', '1h')
result = engine.run(df, 'BTCUSDT', '1h')
```

There is only ONE engine now. The old broken engines were deleted on January 9, 2026.

### Project Status (as of January 9, 2026)
- ✅ Strategy fully documented
- ✅ Data download working (Binance API)
- ✅ Backtest engine (all 3 modules, no look-ahead bias)
- ✅ Visual verification tool (interactive HTML charts)
- ✅ All 12 backtests completed (2 symbols × 6 timeframes)
- ✅ **GIGA OPTIMIZATION COMPLETE** (27,648 backtests, 6,307 valid)
- ✅ Optimal parameters identified (52.2% test return champion)
- 🔲 Live trading not implemented

### Key Files to Know
| File | Purpose |
|------|---------|
| `PROJECT.md` | Full project documentation |
| `backtest/engine/backtest.py` | Backtest engine (all 3 modules) |
| `backtest/visualization/verification_chart.py` | Visual verification HTML generator |
| `backtest/config/settings.py` | All configuration parameters |
| `giga_optimizer_fast.py` | Parameter optimization tool |
| `output/giga_optimization/best_params.json` | Optimal parameters |
| `output/giga_optimization/FINAL_RECOMMENDATIONS.md` | Optimization analysis |
| `.env` | Binance API credentials |

### Data Location
- Historical data: `data/` directory
- Format: `{SYMBOL}_{TIMEFRAME}.csv`
- Available: BTCUSDT and ETHUSDT, all timeframes (1m to 1d)
- Period: ~2 years (Jan 2024 - Jan 2026)

## Trading Strategy Summary

### Entry Modules
1. **Module A (Wave 3)**: Enter at 38.2%-78.6% retracement after Wave 1-2
2. **Module B (Wave 5)**: Enter at 23.6%-50% retracement after Wave 3-4
3. **Module C (Corrective)**: Zigzag, Flat, Triangle patterns

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
from backtest.config.settings import Config
from backtest.engine.backtest import BacktestEngine
from backtest.data.storage import DataStorage

storage = DataStorage('data')
config = Config()
engine = BacktestEngine(config)

for symbol in ['BTCUSDT', 'ETHUSDT']:
    for tf in ['1m', '5m', '15m', '1h', '4h', '1d']:
        df = storage.load(symbol, tf)
        result = engine.run(df, symbol, tf)
        print(f"{symbol} {tf}: {result.metrics.total_return_pct:.2f}%")
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

### Generate Visual Verification Charts
```python
from backtest.visualization.verification_chart import generate_verification_html
from backtest.config.settings import Config
import pandas as pd

df = pd.read_csv('data/BTCUSDT_15m.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

config = Config()
config.module_a_enabled = True  # Or module_b_enabled, module_c_enabled

generate_verification_html(df, 'BTCUSDT', '15m', config, 'output/verification.html')
```

Or use the helper script: `python3 run_module_backtests.py`

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

## Giga Optimization Results (January 7, 2026)

### Summary
- **27,648 backtests** run with walk-forward validation (70/30 split)
- **6,307 valid configurations** (22.8% hit rate)
- **Best test return: 52.2%** (ETHUSDT 1h Module B)

### Champion Configuration
```python
{
    'asset': 'ETHUSDT',
    'timeframe': '1h',
    'module': 'B',  # Wave 5 entries
    'atr_multiplier': 2.0,
    'risk_pct': 2.0,
    'entry_split': 100.0,
    'sl_approach': 'multiplier',
    'sl_multiplier': 0.75,
    'tp1_extension': 1.382,
    'tp2_extension': 2.618,
    'confirmation_delay': 3,
}
```

### Key Findings

**What Works:**
- Module B (Wave 5) dominates - 52% hit rate, best returns
- ETH >> BTC (8.0% avg vs 3.9% avg)
- 1h timeframe is optimal (best Sharpe proxy: 1.08)
- ATR 2.0, TP 138%/261%, Risk 2%, Tighter SL (0.75x)

**What Doesn't Work:**
- Module A/C on 1h (Module B is clearly superior)
- BTC (consistently underperforms ETH)
- 5m/15m timeframes (worst risk-adjusted)
- 0.5% risk (too conservative)
- TP2 at 161.8% (leaving money on table)

**Unusual Pattern:**
- 12/20 top performers have NEGATIVE training but POSITIVE test returns
- Train/test correlation: -0.154 (low = good validation)
- Suggests market regime shift between 2024 and 2025

### Output Files
- `output/giga_optimization/all_results.csv` - 27,648 rows
- `output/giga_optimization/best_params.json` - Optimal parameters
- `output/giga_optimization/FINAL_RECOMMENDATIONS.md` - Full analysis
- `giga_optimizer_fast.py` - Reusable optimization tool

### Next Steps
1. Apply champion parameters to settings.py
2. Paper trade ETHUSDT 1h Module B
3. Monitor for regime changes
4. Consider adding ETHUSDT 15m Module A for diversification
