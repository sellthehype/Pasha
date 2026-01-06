# Elliott Wave Trading System - Backtest Specification
## Comprehensive Technical Requirements Document

---

## 1. Executive Summary

This document specifies the requirements for building a comprehensive backtesting system for the Elliott Wave Fibonacci Trading Strategy (EWFTS). The backtest will evaluate the strategy across multiple timeframes, assets, and market conditions using 2 years of historical Binance Futures data.

**Key Characteristics:**
- Pure Elliott Wave + Fibonacci (no additional indicators)
- Three trading modules (Wave 3, Wave 5, Corrective)
- Probabilistic entry with confirmation
- Bayesian parameter optimization
- Interactive visualization with wave markup

---

## 2. Data Requirements

### 2.1 Data Sources
| Parameter | Value |
|-----------|-------|
| **Exchange** | Binance Futures |
| **Assets** | BTC/USDT, ETH/USDT |
| **Data Type** | Kline (OHLCV) |
| **History** | 2 years (rolling from current date) |

### 2.2 Timeframes
All timeframes must be downloaded and tested independently:

| Timeframe | Candles/Day | ~2 Year Total | Primary Use |
|-----------|-------------|---------------|-------------|
| 1m | 1,440 | ~1,051,200 | Scalping |
| 5m | 288 | ~210,240 | Scalping |
| 15m | 96 | ~70,080 | Day trading |
| 1h | 24 | ~17,520 | Day/Swing |
| 4h | 6 | ~4,380 | Swing |
| 1d | 1 | ~730 | Position |

### 2.3 Data Storage
```
Format: CSV files
Location: data/
Naming: {asset}_{timeframe}.csv
Example: BTCUSDT_1h.csv

Columns: timestamp, open, high, low, close, volume
```

### 2.4 Data Download Implementation
```python
# Pseudo-code for data download
def download_klines(symbol, interval, start_date, end_date):
    """
    Use Binance API to download historical klines
    - Handle rate limits (1200 requests/minute)
    - Paginate through history (max 1500 candles per request)
    - Save incrementally to avoid re-downloading
    - Validate data completeness
    """
```

**API Credentials:** Read from `.env` file
```
BINANCE_API_KEY
BINANCE_API_SECRET
```

---

## 3. Wave Detection System

### 3.1 Zigzag Indicator Implementation

The zigzag indicator identifies swing highs and swing lows based on a minimum reversal threshold.

**Algorithm:**
```python
def zigzag(prices, atr_multiplier):
    """
    1. Calculate ATR(20) for current bar
    2. Set threshold = ATR(20) × atr_multiplier
    3. Track current trend direction
    4. Mark new pivot when price reverses by threshold amount
    5. Return list of pivot points with: index, price, type (high/low)
    """
```

**Parameters:**
| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `atr_multiplier` | 1.5 | 1.0 - 2.5 | Configurable, subject to optimization |
| `atr_period` | 20 | Fixed | Standard ATR lookback |

### 3.2 Minimum Wave Size Filter
```
Requirement: Each wave must move > 0.5% from previous pivot
Purpose: Filter micro-waves that are noise
Implementation: After zigzag identifies pivot, validate size before accepting
```

### 3.3 Wave Counting Logic

**Impulse Wave Identification (Waves 1-5):**
```python
def identify_impulse(pivots):
    """
    From a series of pivots, identify potential 5-wave impulse

    Validation Rules:
    1. Wave 2 retracement < 100% of Wave 1
    2. Wave 3 is not the shortest of waves 1, 3, 5
    3. Wave 4 does not enter Wave 1 territory

    Returns: Wave structure with pivot indices and prices
    """
```

**Corrective Pattern Identification:**

*Zigzag (A-B-C):*
```python
def identify_zigzag(pivots):
    """
    - Wave A: Impulsive (check for 5-wave substructure)
    - Wave B: 38.2% - 78.6% retracement of A
    - Wave C: Projects to 61.8%, 100%, or 161.8% of A
    """
```

*Flat (A-B-C):*
```python
def identify_flat(pivots):
    """
    - Wave A: Corrective (3-wave)
    - Wave B: ≥ 90% retracement of A (can exceed A start)
    - Variants: Regular (B: 90-105%), Expanded (B: 105-138%)
    """
```

*Triangle (A-B-C-D-E):*
```python
def identify_triangle(pivots):
    """
    Fib-based detection (NOT trendline-based):
    - 5 corrective waves
    - Each wave retraces 61.8% - 78.6% of previous wave
    - Overall pattern shows contracting structure
    """
```

### 3.4 Diagonal Detection (Flag Only)
```python
def detect_diagonal(pivots):
    """
    Identify potential diagonal patterns:
    - Wave 4 overlaps Wave 1 (unlike regular impulse)
    - Converging price action

    Action: Flag on chart but DO NOT generate trade signals
    """
```

---

## 4. Entry Signal Generation

### 4.1 Module A: Wave 3 Entry

**Trigger Conditions:**
1. Valid Wave 1 complete (5-wave impulse or leading diagonal)
2. Wave 2 in progress or complete
3. Price at Fibonacci retracement zone of Wave 1

**Fibonacci Entry Zones:**
| Level | Range (±2.5% tolerance) | Priority |
|-------|-------------------------|----------|
| 50.0% | 47.5% - 52.5% | Secondary |
| 61.8% | 59.3% - 64.3% | Primary |
| 78.6% | 76.1% - 81.1% | Aggressive |

**Entry Execution (Probabilistic):**
```
Stage 1 (Initial): Enter INITIAL_POSITION_PCT% when price enters Fib zone
Stage 2 (Confirmation): Add remaining position when higher low forms

INITIAL_POSITION_PCT: Configurable (default 50%)
```

**Higher Low Confirmation Logic:**
```python
def detect_higher_low(prices, fib_touch_index):
    """
    After price touches Fib zone:
    1. Wait for a bounce (swing low forms)
    2. Wait for pullback
    3. Confirm higher low (new swing low above previous)
    4. No time limit - wait until confirmation OR invalidation

    Returns: confirmation_index or None
    """
```

**Stop Loss:**
```
Location: Below Wave 1 start (for longs)
Buffer: 0.1% of price OR 0.5 × ATR (whichever larger)
```

**Take Profit Targets:**
| Target | Calculation | Exit % |
|--------|-------------|--------|
| TP1 | Wave 2 end + (Wave 1 range × 1.0) | 40% |
| TP2 | Wave 2 end + (Wave 1 range × 1.618) | 60% |

### 4.2 Module B: Wave 5 Entry

**Trigger Conditions:**
1. Waves 1, 2, 3, 4 clearly identifiable
2. Wave 3 is not the shortest wave
3. Wave 4 has not entered Wave 1 territory
4. Wave 4 pattern alternates from Wave 2

**Fibonacci Entry Zones (Wave 4 retracement of Wave 3):**
| Condition | Levels |
|-----------|--------|
| Wave 3 Extended (>161.8%) | 23.6% - 38.2% |
| Wave 3 Normal | 38.2% - 50.0% |

**Entry Execution:** Same probabilistic approach as Module A

**Stop Loss:**
```
Wide: Below Wave 1 high (invalidation level)
Tight: Below Wave 4 low (practical level)
Configurable which to use
```

**Take Profit Targets:**
| Target | If W3 Extended | If W3 Normal | Exit % |
|--------|----------------|--------------|--------|
| TP1 | W4 end + W1 range × 0.618 | W4 end + W1 range | 40% |
| TP2 | W4 end + W1 range | W4 end + W1 range × 1.618 | 60% |

### 4.3 Module C: Corrective Wave Entries

**C.1 Zigzag - Wave C Entry:**
```
Entry: At completion of Wave B (38-79% retracement of A)
Stop: Beyond Wave A start
Target: C = 100% of A (primary), 61.8% or 161.8% (secondary)
```

**C.2 Flat - Wave C Entry:**
```
Entry: At completion of Wave B (≥90% retracement of A)
Stop: Beyond Wave B extreme
Target: C = 100-161.8% of A
```

**C.3 Triangle - Post-Thrust Entry:**
```
Entry: On Wave E completion (breakout from triangle)
Stop: Beyond Wave E extreme
Target: Triangle width at start (Wave A to B range)
Direction: Same as wave before triangle
```

---

## 5. Position Management

### 5.1 Position Sizing

**Base Formula:**
```python
def calculate_position_size(account_balance, risk_pct, entry_price, stop_price):
    risk_amount = account_balance * risk_pct
    risk_per_unit = abs(entry_price - stop_price)
    position_size = risk_amount / risk_per_unit
    return position_size
```

**Risk Parameters:**
| Parameter | Default | Range |
|-----------|---------|-------|
| `base_risk_pct` | 1.0% | 0.5% - 2.0% |
| `module_a_multiplier` | 1.0 | Fixed |
| `module_b_multiplier` | 0.75 | Fixed (truncation risk) |
| `module_c_multiplier` | 0.5 | Fixed (counter-trend) |

**ATR-Based Dynamic Sizing:**
```python
def adjust_risk_for_volatility(base_risk, current_atr, baseline_atr):
    """
    ATR calculated on SAME timeframe as trade

    volatility_ratio = current_atr / baseline_atr
    adjusted_risk = base_risk / volatility_ratio

    Caps: 0.5x to 1.5x adjustment
    """
```

### 5.2 Partial Exit Management

**Exit Schedule:**
| Event | Action |
|-------|--------|
| TP1 Hit | Close 40% of position |
| After TP1 | Move stop to breakeven for remaining 60% |
| TP2 Hit | Close remaining 60% |
| Price returns to entry after TP1 | Force close remaining position at breakeven |

### 5.3 Multiple Position Handling
```
- Allow multiple independent positions per asset
- Each signal creates new position with own stops/targets
- Track positions separately in position ledger
- No pyramiding (adding to same position)
```

---

## 6. Trading Costs

### 6.1 Fee Structure
| Order Type | Fee | Usage |
|------------|-----|-------|
| Maker (Limit) | 0.02% | Entry orders, TP orders |
| Taker (Market) | 0.04% | Stop loss orders |

### 6.2 Implementation
```python
def apply_fees(trade):
    """
    Entry: Maker fee (assume limit order fills)
    TP Exit: Maker fee
    SL Exit: Taker fee
    Breakeven Exit: Maker fee
    """
```

### 6.3 Gap Handling
```
Approach: Ignore gaps, trade through
Crypto trades 24/7 - gaps are minimal
No special weekend handling
```

---

## 7. Backtesting Engine

### 7.1 Core Architecture
```
backtest/
├── __init__.py
├── data/
│   ├── __init__.py
│   ├── downloader.py      # Binance API data download
│   ├── storage.py         # CSV read/write
│   └── validation.py      # Data quality checks
├── indicators/
│   ├── __init__.py
│   ├── zigzag.py          # Zigzag pivot detection
│   ├── atr.py             # ATR calculation
│   └── fibonacci.py       # Fib level calculations
├── waves/
│   ├── __init__.py
│   ├── impulse.py         # Impulse wave detection
│   ├── corrective.py      # Corrective pattern detection
│   ├── diagonal.py        # Diagonal detection (flag only)
│   └── validator.py       # Elliott Wave rule validation
├── strategy/
│   ├── __init__.py
│   ├── module_a.py        # Wave 3 entries
│   ├── module_b.py        # Wave 5 entries
│   ├── module_c.py        # Corrective entries
│   └── signals.py         # Signal generation
├── execution/
│   ├── __init__.py
│   ├── position.py        # Position management
│   ├── orders.py          # Order handling
│   └── fees.py            # Fee calculation
├── engine/
│   ├── __init__.py
│   ├── backtest.py        # Main backtest loop
│   ├── portfolio.py       # Portfolio tracking
│   └── metrics.py         # Performance calculation
├── optimization/
│   ├── __init__.py
│   ├── optuna_optimizer.py # Bayesian optimization
│   └── parameters.py      # Parameter definitions
├── analysis/
│   ├── __init__.py
│   ├── statistics.py      # Trade statistics
│   ├── wave_stats.py      # Wave accuracy analysis
│   └── fib_stats.py       # Fibonacci hit rate analysis
├── visualization/
│   ├── __init__.py
│   ├── dashboard.py       # Plotly dashboard
│   ├── charts.py          # Price charts with waves
│   └── tradingview.py     # TradingView export
├── config/
│   ├── __init__.py
│   └── settings.py        # Configuration management
└── main.py                # Entry point
```

### 7.2 Backtest Loop (Pseudo-code)
```python
def run_backtest(data, config):
    """
    Main backtest execution loop
    """
    portfolio = Portfolio(initial_balance=config.initial_balance)
    positions = PositionManager()
    wave_analyzer = WaveAnalyzer(config.zigzag_params)

    for i in range(lookback, len(data)):
        current_bar = data.iloc[i]
        historical = data.iloc[:i+1]

        # 1. Update wave structure
        pivots = wave_analyzer.update(historical)
        waves = wave_analyzer.identify_waves(pivots)

        # 2. Check for new signals
        signals = generate_signals(waves, current_bar, config)

        # 3. Execute entries for new signals
        for signal in signals:
            if validate_signal(signal, positions, portfolio):
                position = execute_entry(signal, current_bar, config)
                positions.add(position)

        # 4. Manage existing positions
        for position in positions.open:
            # Check stop loss
            if check_stop_hit(position, current_bar):
                execute_exit(position, 'stop', current_bar)
            # Check take profits
            elif check_tp_hit(position, current_bar):
                execute_partial_exit(position, current_bar)
            # Check confirmation for stage 2 entry
            elif position.awaiting_confirmation:
                if check_confirmation(position, historical):
                    execute_confirmation_entry(position, current_bar)

        # 5. Record state
        portfolio.record_equity(current_bar.timestamp)

    return generate_results(portfolio, positions)
```

### 7.3 Validation Split
```
Training Set: First 70% of data (optimize parameters)
Test Set: Final 30% of data (out-of-sample validation)

Implementation:
- Run optimization on training set only
- Final performance reported on test set only
- Prevents overfitting
```

---

## 8. Parameter Optimization

### 8.1 Optimization Framework
```
Library: Optuna (Bayesian optimization)
Method: Tree-structured Parzen Estimator (TPE)
```

### 8.2 Optimizable Parameters

| Parameter | Range | Type |
|-----------|-------|------|
| `zigzag_atr_multiplier` | 1.0 - 2.5 | Float |
| `fib_tolerance_pct` | 1.0 - 5.0 | Float |
| `initial_position_pct` | 30 - 70 | Integer |
| `base_risk_pct` | 0.5 - 2.0 | Float |
| `stop_behavior` | [original, breakeven, trailing] | Categorical |
| `entry_fib_levels` | [50, 61.8, 78.6] | Multi-select |

### 8.3 Optimization Objective
```python
def objective(trial):
    """
    Optimize for Sharpe Ratio (risk-adjusted returns)

    Secondary constraints:
    - Max drawdown < 25%
    - Minimum 30 trades (statistical significance)
    - Profit factor > 1.0
    """
    params = sample_parameters(trial)
    results = run_backtest(training_data, params)

    # Penalize if constraints violated
    if results.max_drawdown > 0.25:
        return -10
    if results.trade_count < 30:
        return -10
    if results.profit_factor < 1.0:
        return -10

    return results.sharpe_ratio
```

### 8.4 Optimization Runs
```
Trials: 200 (balance between thoroughness and time)
Pruning: Median pruning after 20% of backtest
Parallelization: Use all available CPU cores
```

---

## 9. Performance Metrics

### 9.1 Standard Trade Metrics
| Metric | Formula |
|--------|---------|
| Total Return | (Final - Initial) / Initial |
| CAGR | (Final/Initial)^(1/years) - 1 |
| Sharpe Ratio | (Return - RiskFree) / StdDev |
| Sortino Ratio | (Return - RiskFree) / DownsideStdDev |
| Max Drawdown | Max peak-to-trough decline |
| Win Rate | Winning trades / Total trades |
| Profit Factor | Gross profit / Gross loss |
| Average Win | Mean profit on winners |
| Average Loss | Mean loss on losers |
| Expectancy | (WinRate × AvgWin) - (LossRate × AvgLoss) |
| Recovery Factor | Total Return / Max Drawdown |

### 9.2 Wave-Specific Statistics
| Metric | Description |
|--------|-------------|
| Wave 3 Extension Rate | % of W3s reaching 161.8% |
| Wave 5 Truncation Rate | % of W5s failing to exceed W3 |
| Wave 2 Depth Distribution | Histogram of W2 retracements |
| Wave 4 Depth Distribution | Histogram of W4 retracements |
| Corrective Pattern Distribution | Count of zigzags, flats, triangles |
| Diagonal Detection Rate | % of impulses that were diagonals |

### 9.3 Fibonacci Statistics
| Metric | Description |
|--------|-------------|
| Entry Level Hit Rate | Success rate by Fib level (50%, 61.8%, 78.6%) |
| Best Performing Entry Level | Which Fib level yields highest expectancy |
| TP1 Hit Rate | % of trades reaching TP1 |
| TP2 Hit Rate | % of trades reaching TP2 |
| Extension Accuracy | How often W3 hits 161.8% vs other levels |

### 9.4 Report Groupings
```
Full matrix analysis:
- By Module (A, B, C, Combined)
- By Timeframe (1m, 5m, 15m, 1h, 4h, 1d)
- By Asset (BTC, ETH)

Total combinations: 3 modules × 6 timeframes × 2 assets = 36 cells
Plus combined views
```

---

## 10. Visualization & Output

### 10.1 Interactive Dashboard (Plotly)

**Dashboard Components:**

1. **Equity Curve Panel**
   - Portfolio value over time
   - Drawdown overlay
   - Trade markers (entry/exit points)

2. **Performance Summary Panel**
   - Key metrics table
   - Monthly returns heatmap
   - Rolling Sharpe ratio

3. **Trade Analysis Panel**
   - Win/loss distribution
   - R-multiple distribution
   - Trade duration analysis

4. **Wave Chart Panel** (Interactive)
   - Candlestick chart
   - Wave labels (1, 2, 3, 4, 5, A, B, C, etc.)
   - Fibonacci retracement/extension lines
   - Entry/exit markers
   - Diagonal flags
   - Zoom/pan capability

5. **Module Comparison Panel**
   - Side-by-side module performance
   - Timeframe comparison matrix
   - Asset comparison

### 10.2 Chart Wave Markup
```python
def draw_wave_markup(chart, wave_structure):
    """
    For each identified wave structure:
    1. Draw wave labels at pivot points
    2. Draw Fibonacci retracement from W1 for W2 entry zone
    3. Draw Fibonacci extension from W2 for W3 targets
    4. Connect pivots with lines (different colors for motive/corrective)
    5. Shade entry zones
    6. Mark actual entry/exit points
    7. Flag diagonal patterns with special marker
    """
```

### 10.3 TradingView Export
```python
def export_to_tradingview(wave_data, trades):
    """
    Export format for TradingView import:
    - Pine Script indicator with wave labels
    - CSV with trade signals for strategy tester
    - JSON for custom visualization
    """
```

### 10.4 Output Files
```
output/
├── reports/
│   ├── backtest_summary.html       # Main dashboard
│   ├── module_a_report.html        # Module A detail
│   ├── module_b_report.html        # Module B detail
│   ├── module_c_report.html        # Module C detail
│   └── optimization_results.html   # Optuna results
├── charts/
│   ├── equity_curve.html
│   ├── wave_charts/                # Per-timeframe wave charts
│   └── trade_charts/               # Individual trade analysis
├── data/
│   ├── trades.csv                  # All trades with details
│   ├── wave_statistics.csv         # Wave behavior data
│   ├── optimization_trials.csv     # All optimization trials
│   └── tradingview_export.json     # TradingView format
└── logs/
    └── backtest.log                # Execution log
```

---

## 11. Configuration

### 11.1 Default Configuration
```python
DEFAULT_CONFIG = {
    # Data
    "assets": ["BTCUSDT", "ETHUSDT"],
    "timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"],
    "history_days": 730,  # 2 years

    # Wave Detection
    "zigzag_atr_multiplier": 1.5,
    "zigzag_atr_period": 20,
    "min_wave_size_pct": 0.5,

    # Entry
    "fib_tolerance_pct": 2.5,
    "entry_fib_levels": [0.5, 0.618, 0.786],
    "initial_position_pct": 50,

    # Risk
    "base_risk_pct": 1.0,
    "max_portfolio_heat_pct": 6.0,
    "module_a_risk_mult": 1.0,
    "module_b_risk_mult": 0.75,
    "module_c_risk_mult": 0.5,

    # Exits
    "tp1_pct": 40,
    "tp2_pct": 60,
    "stop_behavior": "breakeven_after_tp1",

    # Fees
    "maker_fee_pct": 0.02,
    "taker_fee_pct": 0.04,

    # Validation
    "train_test_split": 0.7,

    # Modules
    "module_a_enabled": True,
    "module_b_enabled": True,
    "module_c_enabled": True,
    "trade_diagonals": False,

    # Optimization
    "optimization_trials": 200,
    "optimization_metric": "sharpe_ratio",
}
```

### 11.2 Environment Variables
```
# .env file
BINANCE_API_KEY=xxx
BINANCE_API_SECRET=xxx
```

---

## 12. Implementation Phases

### Phase 1: Foundation
1. Project structure setup
2. Data downloader implementation
3. CSV storage and validation
4. ATR indicator
5. Basic zigzag indicator

### Phase 2: Wave Detection
1. Impulse wave identification
2. Elliott Wave rule validation
3. Zigzag corrective detection
4. Flat corrective detection
5. Triangle corrective detection
6. Diagonal flagging

### Phase 3: Signal Generation
1. Module A (Wave 3) signals
2. Module B (Wave 5) signals
3. Module C (Corrective) signals
4. Higher low confirmation logic
5. Fibonacci zone calculations

### Phase 4: Execution Engine
1. Position sizing
2. Order management
3. Partial exit handling
4. Fee calculation
5. Multi-position tracking

### Phase 5: Backtest Engine
1. Main backtest loop
2. Portfolio tracking
3. Equity curve generation
4. Trade logging

### Phase 6: Analysis & Metrics
1. Standard performance metrics
2. Wave-specific statistics
3. Fibonacci hit rate analysis
4. Report generation

### Phase 7: Optimization
1. Optuna integration
2. Parameter space definition
3. Objective function
4. Training/test split

### Phase 8: Visualization
1. Plotly dashboard
2. Wave chart markup
3. Interactive trade viewer
4. TradingView export

### Phase 9: Testing & Validation
1. Unit tests for wave detection
2. Integration tests for backtest engine
3. Out-of-sample validation
4. Results analysis

---

## 13. Technical Requirements

### 13.1 Python Version
```
Python >= 3.9
```

### 13.2 Dependencies
```
# Core
pandas>=2.0.0
numpy>=1.24.0

# API
python-binance>=1.0.0
python-dotenv>=1.0.0

# Optimization
optuna>=3.0.0

# Visualization
plotly>=5.15.0

# Utilities
tqdm>=4.65.0
loguru>=0.7.0

# Testing
pytest>=7.0.0
```

### 13.3 Hardware Recommendations
```
- CPU: 4+ cores (for parallel optimization)
- RAM: 16GB+ (for multi-timeframe data in memory)
- Storage: 5GB+ (for data and outputs)
```

---

## 14. Success Criteria

### 14.1 Minimum Viable Backtest
- [ ] Downloads 2 years of data for BTC and ETH
- [ ] Identifies wave structures on all timeframes
- [ ] Generates signals for all three modules
- [ ] Executes simulated trades with proper position sizing
- [ ] Calculates standard performance metrics
- [ ] Produces basic equity curve

### 14.2 Complete System
- [ ] All above plus...
- [ ] Interactive Plotly dashboard
- [ ] Wave markup on charts
- [ ] Full matrix reporting (module × timeframe × asset)
- [ ] Wave and Fibonacci statistics
- [ ] Bayesian parameter optimization
- [ ] TradingView export
- [ ] Out-of-sample validation

### 14.3 Performance Targets (Validation, Not Requirements)
```
Note: These are observation targets, not requirements.
Strategy may or may not achieve these - that's what we're testing.

- Sharpe Ratio > 1.0 (on out-of-sample)
- Max Drawdown < 25%
- Win Rate > 40%
- Profit Factor > 1.2
```

---

## 15. Appendix: Fibonacci Reference

### Retracement Levels
| Level | Decimal | Wave 2 | Wave 4 | Wave B |
|-------|---------|--------|--------|--------|
| 23.6% | 0.236 | - | Primary (W3 ext) | - |
| 38.2% | 0.382 | Shallow | Primary | Min Zigzag |
| 50.0% | 0.500 | Primary | Moderate | Common |
| 61.8% | 0.618 | Primary | - | Primary |
| 78.6% | 0.786 | Deep | - | Max Zigzag |

### Extension Levels
| Level | Decimal | Wave 3 | Wave 5 | Wave C |
|-------|---------|--------|--------|--------|
| 61.8% | 0.618 | - | If W3 ext | Min |
| 100% | 1.000 | Min | Equality | Primary |
| 161.8% | 1.618 | Primary | Extended | Extended |
| 261.8% | 2.618 | Extended | - | Rare |

---

*Document Version: 1.0*
*Created: Based on EWFTS Strategy Document*
*Purpose: Backtest Implementation Specification*
