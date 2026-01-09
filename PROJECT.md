# Pasha - Elliott Wave Backtesting System

## Project Overview

**Pasha** is a comprehensive Elliott Wave trading strategy backtesting system for cryptocurrency markets. It implements a rule-based trading system derived from Elliott Wave Theory with Fibonacci retracements/extensions, designed for systematic backtesting across multiple timeframes.

### Core Goals
1. Implement a complete Elliott Wave trading strategy with clear entry/exit rules
2. Backtest the strategy on historical cryptocurrency data (BTCUSDT, ETHUSDT)
3. Analyze performance across all timeframes (1m, 5m, 15m, 1h, 4h, 1d)
4. Optimize parameters using Bayesian optimization (Optuna)
5. Generate comprehensive performance reports

---

## Trading Strategy Summary

### Based On
- Elliott Wave Theory (3 golden rules)
- Fibonacci retracements (38.2%, 50%, 61.8%, 78.6%)
- Fibonacci extensions (100%, 161.8%, 261.8%)

### Three Trading Modules

#### Module A: Wave 3 Entries (Highest Priority)
- **Setup**: After Wave 1-2 completion, enter at Wave 2 retracement
- **Entry Zone**: 38.2% - 78.6% Fibonacci retracement of Wave 1
- **Stop Loss**: Below Wave 1 start (invalidation of Elliott Wave Rule 2)
- **Targets**:
  - TP1: 100% extension of Wave 1
  - TP2: 161.8% extension of Wave 1

#### Module B: Wave 5 Entries
- **Setup**: After Wave 3-4 completion, enter at Wave 4 retracement
- **Entry Zone**: 23.6% - 50% retracement of Wave 3
- **Validation**: Wave 4 must not overlap Wave 1 territory (Rule 3)
- **Targets**:
  - TP1: Previous Wave 3 high
  - TP2: Fibonacci extension from Wave 1 start

#### Module C: Corrective Pattern Entries
- **Patterns**: Zigzag (5-3-5), Flat (3-3-5), Triangle
- **Entry**: At completion of corrective pattern
- **Targets**: Based on pattern-specific Fibonacci relationships

### Position Management
- **Initial Entry**: 100% of calculated position on confirmation (optimized)
- **Risk Per Trade**: 2% of equity (optimized)
- **TP1 Exit**: 40% of position
- **TP2 Exit**: 60% of position (remaining)
- **After TP1**: Stop moved to breakeven

### Stop Loss Rule (CORE - January 2026)
**Capped Stops**: Use the TIGHTER of structural stop OR ATR-based stop.

```python
config.sl_approach = 'capped'      # Default
config.sl_atr_multiplier = 1.5    # Maximum stop = 1.5x ATR from entry
```

| Approach | Description | Performance |
|----------|-------------|-------------|
| `structure` | Wave invalidation points | -29.8% (too wide) |
| `atr` | Pure ATR-based stops | Ignores structure |
| `capped` | **Tighter of both** | **+66.7%** ✓ |

This rule prevents runaway losses when structural stops are too far from entry.

### Fees (Binance Futures)
- Maker: 0.02%
- Taker: 0.04%

---

## Architecture

```
Pasha/
├── .env                          # Binance API credentials (gitignored)
├── .gitignore                    # Git ignore rules
├── requirements.txt              # Python dependencies
├── PROJECT.md                    # This file
├── Elliott_Wave_Trading_System.md    # Detailed strategy documentation
├── Elliott_Wave_Strategy_Summary.md  # Human-friendly strategy summary
├── Backtest_Specification.md     # Technical backtest specification
│
├── backtest/                     # Main backtest package
│   ├── __init__.py
│   │
│   ├── config/                   # Configuration
│   │   ├── __init__.py
│   │   └── settings.py           # Config dataclass, loads .env
│   │
│   ├── data/                     # Data handling
│   │   ├── __init__.py
│   │   ├── downloader.py         # Binance API data download
│   │   └── storage.py            # CSV storage/loading
│   │
│   ├── indicators/               # Technical indicators
│   │   ├── __init__.py
│   │   ├── atr.py                # Average True Range
│   │   ├── zigzag.py             # Swing point detection
│   │   └── fibonacci.py          # Fib calculations
│   │
│   ├── waves/                    # Wave detection
│   │   ├── __init__.py
│   │   ├── impulse.py            # Impulse wave (1-2-3-4-5) detection
│   │   ├── corrective.py         # Corrective patterns (ABC, triangles)
│   │   ├── validator.py          # Elliott Wave rule validation
│   │   └── analyzer.py           # Main wave analysis coordinator
│   │
│   ├── strategy/                 # Trading strategy
│   │   ├── __init__.py
│   │   ├── signals.py            # Signal generation/filtering
│   │   ├── module_a.py           # Wave 3 entry strategy
│   │   ├── module_b.py           # Wave 5 entry strategy
│   │   └── module_c.py           # Corrective entry strategy
│   │
│   ├── execution/                # Trade execution
│   │   ├── __init__.py
│   │   ├── position.py           # Position management
│   │   ├── orders.py             # Order handling
│   │   └── fees.py               # Fee calculation
│   │
│   ├── engine/                   # Backtest engine
│   │   ├── __init__.py
│   │   ├── backtest.py           # Main backtest engine (all 3 modules)
│   │   ├── portfolio.py          # Portfolio tracking
│   │   └── metrics.py            # Performance metrics calculation
│   │
│   ├── analysis/                 # Results analysis
│   │   ├── __init__.py
│   │   ├── statistics.py         # Trade statistics
│   │   ├── wave_stats.py         # Wave behavior analysis
│   │   └── fib_stats.py          # Fibonacci hit rate analysis
│   │
│   ├── visualization/            # Reporting
│   │   ├── __init__.py
│   │   ├── charts.py             # Plotly chart generation
│   │   └── dashboard.py          # HTML dashboard generator
│   │
│   ├── optimization/             # Parameter optimization
│   │   ├── __init__.py
│   │   └── optuna_optimizer.py   # Bayesian optimization with Optuna
│   │
│   └── main.py                   # CLI entry point
│
├── data/                         # Downloaded historical data (CSV)
│   ├── BTCUSDT_1m.csv
│   ├── BTCUSDT_5m.csv
│   ├── BTCUSDT_15m.csv
│   ├── BTCUSDT_1h.csv
│   ├── BTCUSDT_4h.csv
│   ├── BTCUSDT_1d.csv
│   ├── ETHUSDT_1m.csv
│   ├── ETHUSDT_5m.csv
│   ├── ETHUSDT_15m.csv
│   ├── ETHUSDT_1h.csv
│   ├── ETHUSDT_4h.csv
│   └── ETHUSDT_1d.csv
│
└── output/                       # Generated outputs
    ├── backtest_results_summary.csv
    └── reports/
        └── full_comparison_report.html
```

---

## Key Components

### 1. Backtest Engine (`backtest/engine/backtest.py`)

The canonical backtest engine with **no look-ahead bias** and **all 3 modules**.

**Features**:
- Tracks pivot CONFIRMATION bars (not just detection)
- Entries happen at next bar's open after confirmation
- All 3 modules: Wave 3 (A), Wave 5 (B), Corrective patterns (C)
- Full trade audit trail for verification
- Entry/SL/TP validation to filter invalid setups

**Usage**:
```python
from backtest.engine.backtest import BacktestEngine
from backtest.config.settings import Config

config = Config()
engine = BacktestEngine(config)
result = engine.run(df, 'BTCUSDT', '1h')
```

### 2. Visual Verification Tool (`backtest/visualization/verification_chart.py`)

Interactive HTML charts for visually verifying backtest correctness.

**Features**:
- Plotly.js candlestick charts with proper datetime x-axis
- Wave pattern overlays with pivot markers
- Trade entry/exit visualization with P&L coloring
- Click-to-teleport: Click trade in table → chart zooms to location
- Audit issues panel with clickable problem locations
- Full trade accounting detail in sidebar
- Toggleable layers (waves, fibs, SL/TP, trade paths)
- Keyboard navigation (arrow keys to cycle trades)

**Generate verification charts**:
```python
from backtest.visualization.verification_chart import generate_verification_html
from backtest.config.settings import Config

config = Config()
config.module_a_enabled = True  # Enable desired modules
generate_verification_html(df, 'BTCUSDT', '15m', config, 'output/verification.html')
```

### 3. Pivot Detection (`VectorizedPivotDetector`)

Uses ATR-based zigzag algorithm:
- Threshold = ATR × multiplier (default 1.5)
- Minimum wave size = 0.5% of price
- Detects swing highs and lows for wave analysis

### 4. Wave Analysis (`VectorizedWaveAnalyzer`)

Identifies Elliott Wave setups from pivot sequences:
- **Wave 3 setups**: Low-High-Low pattern with valid retracement
- **Wave 5 setups**: 5-pivot pattern with valid wave structure
- Validates Elliott Wave rules (no overlap, proper retracements)

### 5. Configuration (`backtest/config/settings.py`)

All parameters are centralized in the `Config` dataclass:
- Wave detection parameters
- Entry/exit rules
- Risk management
- Fee structure
- API credentials (from .env)

---

## Current Progress

### Completed ✅
1. **Strategy Development**
   - Full Elliott Wave trading system documented
   - Entry/exit rules defined
   - Position sizing and risk management specified

2. **Data Infrastructure**
   - Binance API integration
   - Historical data download (2 years: Jan 2024 - Jan 2026)
   - CSV storage system

3. **Backtest Engine**
   - Single canonical engine with all 3 modules
   - No look-ahead bias (entries at next bar's open)
   - Old broken engines deleted (January 9, 2026)

4. **Full Backtests Run**
   - BTCUSDT: All 6 timeframes
   - ETHUSDT: All 6 timeframes
   - Results documented and analyzed

5. **Reporting**
   - CSV summary export
   - HTML comparison dashboard

6. **Visual Verification Tool** - NEW (January 6, 2026)
   - Interactive HTML charts for backtest verification
   - Click-to-teleport navigation
   - Per-module verification (A, B, C separately)
   - Full trade audit trail

7. **Bug Fixes** - NEW (January 6, 2026)
   - Fixed entry/SL/TP validation in all pattern types (Wave 5, Zigzag, Flat, Triangle)
   - Fixed datetime formatting for intraday timeframes (was aggregating by date)
   - Fixed categorical x-axis navigation in Plotly charts
   - Fixed autoscale Y-axis for visible candles only

8. **GIGA OPTIMIZATION** - NEW (January 7, 2026)
   - 27,648 parameter combinations tested
   - Walk-forward validation (70% train / 30% test)
   - 6,307 valid configurations found (22.8% hit rate)
   - Champion: ETHUSDT 1h Module B with 52.2% test return
   - Full analysis in `output/giga_optimization/FINAL_RECOMMENDATIONS.md`

### Giga Optimization Results (January 7, 2026)

**Champion Configuration:**
```python
{
    'asset': 'ETHUSDT',
    'timeframe': '1h',
    'module': 'B',  # Wave 5 entries
    'atr_multiplier': 2.0,
    'risk_pct': 2.0,
    'entry_split': 100.0,
    'sl_multiplier': 0.75,
    'tp1_extension': 1.382,
    'tp2_extension': 2.618,
    'confirmation_delay': 3,
}
```

**Key Findings:**
| Finding | Insight |
|---------|---------|
| Module B >> A >> C | Wave 5 setups are rarer but more reliable |
| ETH >> BTC | ETH avg 8.0% vs BTC 3.9% test return |
| 1h is optimal | Best Sharpe proxy (1.08) |
| ATR 2.0 works best | Larger swings = cleaner signals |
| TP 138%/261% optimal | Let winners run |
| 0.75x SL multiplier | Tighter stops improve R:R |

**Parameter Sensitivity:**
| Parameter | Best Value | Avg Return |
|-----------|------------|------------|
| ATR Multiplier | 2.0 | 6.4% |
| Risk % | 2.0% | 7.0% |
| TP2 Extension | 2.618 | 6.0% |
| Confirmation | 2-3 bars | 5.8% |

### Latest Backtest Results (January 2026)

| Symbol | Timeframe | Return % | Sharpe | Max DD % | Trades | Win Rate | PF |
|--------|-----------|----------|--------|----------|--------|----------|-----|
| BTCUSDT | 1m | 14,251% | 2.75 | 3.12% | 6,696 | 63.9% | 3.18 |
| BTCUSDT | 5m | 12,343% | 5.50 | 3.33% | 5,380 | 63.6% | 3.29 |
| BTCUSDT | 15m | 8,728% | 8.78 | 2.08% | 3,510 | 63.5% | 3.38 |
| BTCUSDT | 1h | 3,693% | 13.25 | 2.30% | 1,209 | 63.2% | 3.70 |
| BTCUSDT | 4h | 456% | 20.66 | 1.24% | 288 | 64.6% | 4.24 |
| BTCUSDT | 1d | 20% | 17.87 | 0.00% | 45 | 62.2% | 3.12 |
| ETHUSDT | 1m | 29,558% | 3.11 | 3.04% | 12,238 | 64.9% | 3.35 |
| ETHUSDT | 5m | 23,051% | 5.82 | 2.00% | 8,640 | 64.4% | 3.42 |
| ETHUSDT | 15m | 13,728% | 8.99 | 2.19% | 4,495 | 63.5% | 3.47 |
| ETHUSDT | 1h | 4,525% | 15.03 | 2.25% | 1,234 | 63.0% | 3.72 |
| ETHUSDT | 4h | 627% | 19.86 | 2.62% | 324 | 63.3% | 4.40 |
| ETHUSDT | 1d | 28% | 20.70 | 0.00% | 44 | 61.4% | 4.19 |

**Key Findings**:
- Consistent ~63-65% win rate across all configurations
- Higher timeframes = better Sharpe ratios, fewer trades
- Lower timeframes = more trades, more absolute return
- ETH shows slightly better performance than BTC
- Max drawdown stayed under 3.5% for all configurations

---

## How to Run

### Prerequisites
```bash
pip install -r requirements.txt
```

### Environment Setup
Create `.env` file with Binance API credentials:
```
BINANCE_API_KEY=your_key_here
BINANCE_API_SECRET=your_secret_here
```

### Download Data
```bash
python3 backtest/main.py --action download
```

### Run Single Backtest
```bash
python3 backtest/main.py --action backtest --symbol BTCUSDT --timeframe 1h
```

### Run All Backtests
```python
from backtest.config.settings import Config
from backtest.engine.backtest import BacktestEngine
from backtest.data.storage import DataStorage

storage = DataStorage('data')
config = Config()
engine = BacktestEngine(config)

df = storage.load('BTCUSDT', '1h')
result = engine.run(df, 'BTCUSDT', '1h', show_progress=True)

print(f"Return: {result.metrics.total_return_pct:.2f}%")
print(f"Sharpe: {result.metrics.sharpe_ratio:.2f}")
```

### Run Optimization
```bash
python3 backtest/main.py --action optimize --symbol BTCUSDT --timeframe 1h --trials 100
```

---

## Future Goals / Roadmap

### Short Term
1. **Walk-Forward Analysis** - Validate out-of-sample performance
2. **Monte Carlo Simulation** - Test strategy robustness
3. **TradingView Export** - Generate Pine Script alerts

### Medium Term
1. **Live Paper Trading** - Connect to Binance testnet
2. **Real-time Signal Generation** - Streaming data analysis
3. **Multi-Asset Portfolio** - Correlation-based position sizing
4. **Machine Learning Enhancement** - Pattern recognition with ML

### Long Term
1. **Live Trading Bot** - Full automation with risk controls
2. **Web Dashboard** - Real-time monitoring interface
3. **Strategy Marketplace** - Multiple strategy variants

---

## Key Design Decisions

### Why Vectorized Optimization?
The original row-by-row approach was O(n²) for pivot recalculation. By pre-computing pivots once (O(n)), we reduced complexity dramatically. The optimized engine processes 1M candles in ~1 second.

### Why Not Use Libraries Like Backtrader?
- Custom Elliott Wave logic doesn't fit standard indicator frameworks
- Need fine control over wave detection algorithms
- Easier to optimize custom code than modify library internals

### Why Fibonacci Tolerance of 2.5%?
After testing, 2.5% provides the best balance:
- Too tight (1%): Misses valid setups
- Too loose (5%): Too many false signals

### Why 70/30 Train/Test Split?
Standard for financial backtesting. 70% for optimization, 30% for out-of-sample validation prevents overfitting.

---

## Known Issues / Limitations

1. **Returns May Be Overstated**
   - Position sizing allows up to 10x initial balance compounding
   - Real-world slippage not modeled
   - Consider these as "best case" scenarios

2. **Module C Only in Fixed Engine**
   - Optimized engine only implements Modules A and B
   - Use `FixedBacktestEngine` for Module C (corrective patterns)

3. **Simplified Confirmation Logic**
   - The "higher low" confirmation is simplified
   - Full implementation would check actual price structure

4. **Single Asset Per Run**
   - No cross-asset correlation analysis
   - Each symbol/timeframe is independent

5. **Fixed Engine Slower Than Optimized**
   - Fixed engine prioritizes accuracy over speed
   - Use optimized engine for bulk backtesting, fixed engine for verification

---

## Dependencies

```
pandas>=2.0.0
numpy>=1.24.0
python-binance>=1.0.0
python-dotenv>=1.0.0
optuna>=3.0.0
plotly>=5.15.0
tqdm>=4.65.0
pytest>=7.0.0
```

---

## Contact / Notes

- Project started: January 2026
- Primary developer: Claude Code (AI assistant)
- User: Referred to as "my human" or "meat bag"
- All responses end with "Billions"

---

*Last updated: January 6, 2026*
