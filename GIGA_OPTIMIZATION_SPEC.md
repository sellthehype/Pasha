# Giga Optimization Specification

## Overview

Comprehensive parameter optimization for the Pasha Elliott Wave trading system across all data, timeframes, and modules.

---

## Objectives & Constraints

| Metric | Target |
|--------|--------|
| **Primary Goal** | Maximize total profit |
| **Max Drawdown** | < 30% |
| **Win Rate** | Not a concern (asymmetric returns OK) |

**Philosophy**: It's acceptable to lose frequently with small losses if the wins are large enough to compensate.

---

## Data Configuration

### Assets
- **BTCUSDT** - Separate parameter optimization
- **ETHUSDT** - Separate parameter optimization
- Parameters MAY differ between BTC and ETH

### Timeframes to Optimize
| Timeframe | Status |
|-----------|--------|
| 1m | ✅ Include |
| 5m | ✅ Include |
| 15m | ✅ Include |
| 1h | ✅ Include |
| 4h | ✅ Include |
| 1d | ❌ Exclude (insufficient trades for valid optimization) |

### Data Split
- **Training**: First 70% of data
- **Testing**: Last 30% of data
- **Validation Method**: 3-fold walk-forward cross-validation

### Regime Filtering
- **Disabled** - Optimize on combined data without bull/bear separation

---

## Modules

All three modules optimized **independently** with separate parameters:

| Module | Description | Entry Zone (FIXED) |
|--------|-------------|-------------------|
| **Module A** | Wave 3 entries | 38.2% - 78.6% retracement |
| **Module B** | Wave 5 entries | 23.6% - 50% retracement |
| **Module C** | Corrective patterns | Pattern-specific |

**Note**: Entry zones are kept at classic Fibonacci levels (not optimized).

---

## Parameters to Optimize

### 1. Stop Loss Strategy

Test ALL of the following approaches:

| Approach | Description | Parameter Range |
|----------|-------------|-----------------|
| Rule-based | Elliott invalidation point | N/A (baseline) |
| ATR buffer | Add buffer below invalidation | 0.5x - 2.0x ATR |
| Tighter stops | Closer stops for better R:R | 0.25x - 0.75x of current |
| SL multiplier | Scale current SL distance | 0.5x - 2.0x |

### 2. Take Profit Strategy

Test ALL of the following approaches:

| Approach | Description | Parameters |
|----------|-------------|------------|
| Fixed Fib levels | TP1/TP2 at Fib extensions | TP1: 100%, 127.2%, 138.2%<br>TP2: 161.8%, 200%, 261.8% |
| Trailing stop | After TP1, trail remaining | Trail distance: 1x-3x ATR |
| Dynamic ATR | Scale targets with volatility | Multiplier: 2x-6x ATR |

### 3. Position Sizing

| Parameter | Values to Test |
|-----------|----------------|
| Risk per trade | 0.5%, 1.0%, 1.5%, 2.0% |
| Entry split ratio | 100/0, 70/30, 50/50, 30/70 |

### 4. Wave Detection

| Parameter | Range |
|-----------|-------|
| ATR multiplier for pivots | 1.0 - 2.0 (step: 0.1) |

### 5. Confirmation Logic

| Approach | Description |
|----------|-------------|
| Current | Existing higher-low confirmation |
| Disabled | Full position at signal (no confirmation) |
| Bar delay | Wait 1, 2, 3, or 5 bars for confirmation |

---

## Parameter Grid Summary

| Parameter | Values | Count |
|-----------|--------|-------|
| Asset | BTC, ETH | 2 |
| Timeframe | 1m, 5m, 15m, 1h, 4h | 5 |
| Module | A, B, C | 3 |
| SL approach | 4 variants | 4 |
| SL multiplier | 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0 | 7 |
| TP approach | 3 variants | 3 |
| TP1 level | 100%, 127.2%, 138.2% | 3 |
| TP2 level | 161.8%, 200%, 261.8% | 3 |
| Risk % | 0.5%, 1.0%, 1.5%, 2.0% | 4 |
| Split ratio | 100/0, 70/30, 50/50, 30/70 | 4 |
| ATR multiplier | 1.0-2.0 (0.1 step) | 11 |
| Confirmation | current, disabled, 1, 2, 3, 5 bars | 6 |

**Estimated total combinations**: 10,000+ (after applying constraints)

---

## Constraints (Hard)

Skip combinations where:
- TP1 >= TP2
- SL distance <= 0
- Risk % > 2%
- Any mathematically invalid configuration

---

## Validation Protocol

### 3-Fold Walk-Forward

```
Fold 1: Train [0%-46%]   → Test [47%-70%]
Fold 2: Train [23%-70%]  → Test [70%-93%]
Fold 3: Train [0%-70%]   → Test [70%-100%]
```

A parameter set is considered valid if:
1. Profitable on training data
2. Profitable on holdout test data
3. Drawdown < 30% on both train and test

### Final Validation
- **Champion vs Baseline**: Simple comparison of best found params vs current defaults
- Compare total return, Sharpe ratio, max drawdown, profit factor

---

## Output Specification

### Location
```
output/giga_optimization/
```

### Files Generated

| File | Contents |
|------|----------|
| `all_results.csv` | Every valid completed backtest run |
| `best_params_btc.json` | Optimal parameters for BTCUSDT |
| `best_params_eth.json` | Optimal parameters for ETHUSDT |
| `module_a_results.csv` | Module A specific results |
| `module_b_results.csv` | Module B specific results |
| `module_c_results.csv` | Module C specific results |
| `champion_vs_baseline.md` | Final comparison report |

### CSV Columns

```csv
run_id, asset, timeframe, module,
sl_approach, sl_multiplier,
tp_approach, tp1_level, tp2_level,
risk_pct, split_ratio,
atr_multiplier, confirmation_type,
train_return_pct, train_sharpe, train_max_dd, train_trades, train_win_rate, train_pf,
test_return_pct, test_sharpe, test_max_dd, test_trades, test_win_rate, test_pf,
fold_num, is_valid, notes
```

---

## Decision Framework

After optimization, provide recommendations for:

### Option 1: Universal Parameters
- Single parameter set that works across ALL timeframes
- Pros: Simple, robust
- Use if: Performance is consistent across timeframes

### Option 2: Per-Timeframe Parameters
- Different optimal parameters for each timeframe
- Include recommendation on which timeframes to DROP if unprofitable
- Use if: Significant performance variation by timeframe

### Option 3: Per-Asset Parameters
- Different parameters for BTC vs ETH
- Use if: Assets behave fundamentally differently

### Option 4: Per-Module Parameters
- Different parameters for Module A vs B vs C
- Include recommendation on which modules to DISABLE if unprofitable
- Use if: Module performance varies significantly

---

## Execution Process

1. **Phase 1**: Load all data, compute 70/30 splits for each fold
2. **Phase 2**: Run Module A optimization (all assets, timeframes)
3. **Phase 3**: Run Module B optimization (all assets, timeframes)
4. **Phase 4**: Run Module C optimization (all assets, timeframes)
5. **Phase 5**: Aggregate results, rank by test performance
6. **Phase 6**: Generate champion vs baseline comparison
7. **Phase 7**: Write recommendations with reasoning

### Iterative Refinement
After each batch of tests, analyze results to:
- Identify promising parameter regions
- Skip clearly unprofitable combinations
- Focus compute on high-potential areas

---

## Runtime Estimate

- **10,000+ backtests** expected
- **Optimized engine**: ~1 second per full backtest
- **Estimated total time**: 3-6 hours

---

## Success Criteria

The optimization is successful if:
1. ✅ At least one parameter combination beats baseline on holdout data
2. ✅ Drawdown stays under 30% threshold
3. ✅ Results are reproducible across folds
4. ✅ Clear reasoning provided for why optimal parameters work

---

*Specification created: January 6, 2026*
*Ready for execution upon user approval*
