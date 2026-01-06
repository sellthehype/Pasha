# Pasha - Quick Start Guide

## 1. Install Dependencies
```bash
cd /Users/jennieseline/Desktop/Pasha
pip3 install -r requirements.txt
```

## 2. Set Up API Keys
Edit `.env` file with your Binance credentials:
```
BINANCE_API_KEY=your_key
BINANCE_API_SECRET=your_secret
```

## 3. Download Data (requires VPN if Binance blocked)
```bash
python3 backtest/main.py --action download
```

## 4. Run a Backtest
```python
from backtest.config.settings import Config
from backtest.engine.backtest_optimized import OptimizedBacktestEngine
from backtest.data.storage import DataStorage

storage = DataStorage('data')
config = Config()
engine = OptimizedBacktestEngine(config)

# Load data and run
df = storage.load('BTCUSDT', '1h')
result = engine.run(df, 'BTCUSDT', '1h', show_progress=True)

# View results
print(f"Return: {result.metrics.total_return_pct:.2f}%")
print(f"Sharpe: {result.metrics.sharpe_ratio:.2f}")
print(f"Trades: {result.metrics.total_trades}")
print(f"Win Rate: {result.metrics.win_rate*100:.1f}%")
```

## 5. Run All Backtests
```bash
python3 << 'EOF'
from backtest.config.settings import Config
from backtest.engine.backtest_optimized import OptimizedBacktestEngine
from backtest.data.storage import DataStorage

storage = DataStorage('data')
config = Config()
engine = OptimizedBacktestEngine(config)

for symbol in ['BTCUSDT', 'ETHUSDT']:
    for tf in ['1m', '5m', '15m', '1h', '4h', '1d']:
        df = storage.load(symbol, tf)
        result = engine.run(df, symbol, tf, show_progress=False)
        print(f"{symbol} {tf}: {result.metrics.total_return_pct:.2f}% return, {result.metrics.sharpe_ratio:.2f} Sharpe")
EOF
```

## 6. View Results
- CSV: `output/backtest_results_summary.csv`
- HTML Report: `output/reports/full_comparison_report.html`

## Key Files
- `PROJECT.md` - Full documentation
- `Elliott_Wave_Trading_System.md` - Strategy details
- `backtest/engine/backtest_optimized.py` - Fast backtest engine

## Performance
- 1m data (1M candles): ~1.2 seconds
- All 12 backtests: ~3 seconds total
