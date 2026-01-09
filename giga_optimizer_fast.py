#!/usr/bin/env python3
"""
GIGA OPTIMIZER FAST - Optimized for speed

Key optimizations:
1. Pre-compute pivots once per ATR multiplier (the expensive operation)
2. Cache pivot computations
3. Test SL/TP/risk parameters without re-computing pivots
4. Start with faster timeframes (4h->1h->15m->5m->1m)
5. Use numpy vectorization throughout

This should be ~50-100x faster than the original.
"""

import os
import sys
import json
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest.config.settings import Config
from backtest.data.storage import DataStorage
from backtest.indicators.atr import calculate_atr


# =============================================================================
# FAST PIVOT COMPUTATION (cached per ATR multiplier)
# =============================================================================

class FastPivotCache:
    """Cache pivots by ATR multiplier to avoid recomputation"""

    def __init__(self):
        self.cache = {}  # (asset, tf, atr_mult) -> pivots

    def get_pivots(
        self,
        df: pd.DataFrame,
        asset: str,
        tf: str,
        atr_multiplier: float,
        min_wave_pct: float = 0.5
    ) -> List[Dict]:
        """Get pivots, using cache if available"""
        key = (asset, tf, round(atr_multiplier, 2))

        if key in self.cache:
            return self.cache[key]

        # Compute pivots
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        atr = calculate_atr(df, 20).values

        threshold = np.maximum(atr * atr_multiplier, close * min_wave_pct / 100)

        pivots = []
        direction = 0
        current_high = high[0]
        current_high_idx = 0
        current_low = low[0]
        current_low_idx = 0
        n = len(high)

        for i in range(1, n):
            th = threshold[i]

            if direction == 0:
                if high[i] > current_high:
                    current_high = high[i]
                    current_high_idx = i
                if low[i] < current_low:
                    current_low = low[i]
                    current_low_idx = i

                if current_high - current_low >= th:
                    if current_high_idx > current_low_idx:
                        direction = 1
                        pivots.append({
                            'pivot_idx': current_low_idx,
                            'confirmed_idx': i,
                            'price': current_low,
                            'type': -1
                        })
                        current_high = high[i]
                        current_high_idx = i
                    else:
                        direction = -1
                        pivots.append({
                            'pivot_idx': current_high_idx,
                            'confirmed_idx': i,
                            'price': current_high,
                            'type': 1
                        })
                        current_low = low[i]
                        current_low_idx = i

            elif direction == 1:
                if high[i] > current_high:
                    current_high = high[i]
                    current_high_idx = i
                elif current_high - low[i] >= th:
                    pivots.append({
                        'pivot_idx': current_high_idx,
                        'confirmed_idx': i,
                        'price': current_high,
                        'type': 1
                    })
                    direction = -1
                    current_low = low[i]
                    current_low_idx = i

            else:
                if low[i] < current_low:
                    current_low = low[i]
                    current_low_idx = i
                elif high[i] - current_low >= th:
                    pivots.append({
                        'pivot_idx': current_low_idx,
                        'confirmed_idx': i,
                        'price': current_low,
                        'type': -1
                    })
                    direction = 1
                    current_high = high[i]
                    current_high_idx = i

        self.cache[key] = pivots
        return pivots


# =============================================================================
# FAST SETUP FINDER (uses pre-computed pivots)
# =============================================================================

def find_setups_fast(
    pivots: List[Dict],
    open_prices: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    atr: np.ndarray,
    module: str,
    confirmation_delay: int,
    sl_approach: str,
    sl_multiplier: float,
    tp1_ext: float,
    tp2_ext: float,
    fib_tolerance: float = 0.025
) -> List[Dict]:
    """
    Find setups from pre-computed pivots with specific SL/TP params.
    This is fast because pivots are already computed.
    """
    setups = []
    n_pivots = len(pivots)
    n_bars = len(open_prices)

    if module == 'A':
        # Wave 3 setups
        for i in range(2, n_pivots):
            p0, p1, p2 = pivots[i-2], pivots[i-1], pivots[i]

            # Bullish
            if p0['type'] == -1 and p1['type'] == 1 and p2['type'] == -1:
                w1_start, w1_end, w2_end = p0['price'], p1['price'], p2['price']
                w1_range = w1_end - w1_start
                if w1_range <= 0:
                    continue

                retrace = (w1_end - w2_end) / w1_range
                if not (0.382 - fib_tolerance <= retrace <= 0.786 + fib_tolerance):
                    continue

                entry_bar = p2['confirmed_idx'] + confirmation_delay
                if entry_bar >= n_bars:
                    continue

                entry_price = open_prices[entry_bar]
                atr_at_entry = atr[min(entry_bar, len(atr)-1)]

                # Base SL
                base_sl = w1_start - (w1_range * 0.01)
                if sl_approach == 'multiplier':
                    sl = entry_price - (entry_price - base_sl) * sl_multiplier
                else:
                    sl = base_sl

                tp1 = entry_price + w1_range * tp1_ext
                tp2 = entry_price + w1_range * tp2_ext

                if entry_price <= sl or entry_price >= tp1 or tp1 >= tp2:
                    continue

                setups.append({
                    'direction': 'long', 'entry_bar': entry_bar,
                    'entry_price': entry_price, 'stop_loss': sl,
                    'tp1': tp1, 'tp2': tp2, 'wave_range': w1_range
                })

            # Bearish
            elif p0['type'] == 1 and p1['type'] == -1 and p2['type'] == 1:
                w1_start, w1_end, w2_end = p0['price'], p1['price'], p2['price']
                w1_range = w1_start - w1_end
                if w1_range <= 0:
                    continue

                retrace = (w2_end - w1_end) / w1_range
                if not (0.382 - fib_tolerance <= retrace <= 0.786 + fib_tolerance):
                    continue

                entry_bar = p2['confirmed_idx'] + confirmation_delay
                if entry_bar >= n_bars:
                    continue

                entry_price = open_prices[entry_bar]
                base_sl = w1_start + (w1_range * 0.01)
                if sl_approach == 'multiplier':
                    sl = entry_price + (base_sl - entry_price) * sl_multiplier
                else:
                    sl = base_sl

                tp1 = entry_price - w1_range * tp1_ext
                tp2 = entry_price - w1_range * tp2_ext

                if entry_price >= sl or entry_price <= tp1 or tp1 <= tp2:
                    continue

                setups.append({
                    'direction': 'short', 'entry_bar': entry_bar,
                    'entry_price': entry_price, 'stop_loss': sl,
                    'tp1': tp1, 'tp2': tp2, 'wave_range': w1_range
                })

    elif module == 'B':
        # Wave 5 setups
        for i in range(4, n_pivots):
            p0, p1, p2, p3, p4 = [pivots[i-4+j] for j in range(5)]

            # Bullish
            if (p0['type'] == -1 and p1['type'] == 1 and p2['type'] == -1 and
                p3['type'] == 1 and p4['type'] == -1):
                w1_s, w1_e, w2_e, w3_e, w4_e = p0['price'], p1['price'], p2['price'], p3['price'], p4['price']

                if w2_e < w1_s or w4_e < w1_e:
                    continue

                w3_range = w3_e - w2_e
                w1_range = w1_e - w1_s
                if w3_range < w1_range * 0.5:
                    continue

                retrace = (w3_e - w4_e) / w3_range if w3_range > 0 else 0
                if not (0.236 - fib_tolerance <= retrace <= 0.5 + fib_tolerance):
                    continue

                entry_bar = p4['confirmed_idx'] + confirmation_delay
                if entry_bar >= n_bars:
                    continue

                entry_price = open_prices[entry_bar]
                base_sl = w1_e - (w1_range * 0.01)
                if sl_approach == 'multiplier':
                    sl = entry_price - (entry_price - base_sl) * sl_multiplier
                else:
                    sl = base_sl

                tp1 = entry_price + w3_range * tp1_ext
                tp2 = entry_price + w3_range * tp2_ext

                if entry_price <= sl or entry_price >= tp1 or tp1 >= tp2:
                    continue

                setups.append({
                    'direction': 'long', 'entry_bar': entry_bar,
                    'entry_price': entry_price, 'stop_loss': sl,
                    'tp1': tp1, 'tp2': tp2, 'wave_range': w3_range
                })

            # Bearish
            elif (p0['type'] == 1 and p1['type'] == -1 and p2['type'] == 1 and
                  p3['type'] == -1 and p4['type'] == 1):
                w1_s, w1_e, w2_e, w3_e, w4_e = p0['price'], p1['price'], p2['price'], p3['price'], p4['price']

                if w2_e > w1_s or w4_e > w1_e:
                    continue

                w3_range = w2_e - w3_e
                w1_range = w1_s - w1_e
                if w3_range < w1_range * 0.5:
                    continue

                retrace = (w4_e - w3_e) / w3_range if w3_range > 0 else 0
                if not (0.236 - fib_tolerance <= retrace <= 0.5 + fib_tolerance):
                    continue

                entry_bar = p4['confirmed_idx'] + confirmation_delay
                if entry_bar >= n_bars:
                    continue

                entry_price = open_prices[entry_bar]
                base_sl = w1_e + (w1_range * 0.01)
                if sl_approach == 'multiplier':
                    sl = entry_price + (base_sl - entry_price) * sl_multiplier
                else:
                    sl = base_sl

                tp1 = entry_price - w3_range * tp1_ext
                tp2 = entry_price - w3_range * tp2_ext

                if entry_price >= sl or entry_price <= tp1 or tp1 <= tp2:
                    continue

                setups.append({
                    'direction': 'short', 'entry_bar': entry_bar,
                    'entry_price': entry_price, 'stop_loss': sl,
                    'tp1': tp1, 'tp2': tp2, 'wave_range': w3_range
                })

    elif module == 'C':
        # Corrective (zigzag) setups
        for i in range(2, n_pivots):
            p0, p1, p2 = pivots[i-2], pivots[i-1], pivots[i]

            # Bearish zigzag
            if p0['type'] == 1 and p1['type'] == -1 and p2['type'] == 1:
                a_start, a_end, b_end = p0['price'], p1['price'], p2['price']
                a_range = a_start - a_end
                if a_range <= 0:
                    continue

                b_retrace = (b_end - a_end) / a_range
                if not (0.382 - fib_tolerance <= b_retrace <= 0.786 + fib_tolerance):
                    continue
                if b_end >= a_start:
                    continue

                entry_bar = p2['confirmed_idx'] + confirmation_delay
                if entry_bar >= n_bars:
                    continue

                entry_price = open_prices[entry_bar]
                base_sl = a_start + (a_range * 0.02)
                if sl_approach == 'multiplier':
                    sl = entry_price + (base_sl - entry_price) * sl_multiplier
                else:
                    sl = base_sl

                tp1 = entry_price - a_range * 0.618 * tp1_ext
                tp2 = entry_price - a_range * tp2_ext

                if entry_price >= sl or entry_price <= tp1 or tp1 <= tp2:
                    continue

                setups.append({
                    'direction': 'short', 'entry_bar': entry_bar,
                    'entry_price': entry_price, 'stop_loss': sl,
                    'tp1': tp1, 'tp2': tp2, 'wave_range': a_range
                })

            # Bullish zigzag
            elif p0['type'] == -1 and p1['type'] == 1 and p2['type'] == -1:
                a_start, a_end, b_end = p0['price'], p1['price'], p2['price']
                a_range = a_end - a_start
                if a_range <= 0:
                    continue

                b_retrace = (a_end - b_end) / a_range
                if not (0.382 - fib_tolerance <= b_retrace <= 0.786 + fib_tolerance):
                    continue
                if b_end <= a_start:
                    continue

                entry_bar = p2['confirmed_idx'] + confirmation_delay
                if entry_bar >= n_bars:
                    continue

                entry_price = open_prices[entry_bar]
                base_sl = a_start - (a_range * 0.02)
                if sl_approach == 'multiplier':
                    sl = entry_price - (entry_price - base_sl) * sl_multiplier
                else:
                    sl = base_sl

                tp1 = entry_price + a_range * 0.618 * tp1_ext
                tp2 = entry_price + a_range * tp2_ext

                if entry_price <= sl or entry_price >= tp1 or tp1 >= tp2:
                    continue

                setups.append({
                    'direction': 'long', 'entry_bar': entry_bar,
                    'entry_price': entry_price, 'stop_loss': sl,
                    'tp1': tp1, 'tp2': tp2, 'wave_range': a_range
                })

    return sorted(setups, key=lambda x: x['entry_bar'])


# =============================================================================
# FAST TRADE SIMULATION (vectorized where possible)
# =============================================================================

def simulate_trades_fast(
    setups: List[Dict],
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    risk_pct: float,
    entry_split: float,
    initial_balance: float = 10000.0
) -> Dict[str, float]:
    """
    Fast trade simulation.
    Returns metrics dict.
    """
    if not setups:
        return {
            'total_return_pct': 0.0, 'sharpe_ratio': 0.0,
            'max_drawdown_pct': 0.0, 'total_trades': 0,
            'win_rate': 0.0, 'profit_factor': 0.0
        }

    n = len(close)
    balance = initial_balance
    equity = [balance]

    trades_pnl = []
    open_trades = []
    setup_idx = 0

    tp1_pct = 0.4
    tp2_pct = 0.6
    fee_rate = 0.0004

    for i in range(1, n):
        curr_high = high[i]
        curr_low = low[i]
        curr_close = close[i]

        # Check exits
        new_open = []
        for trade in open_trades:
            exit_price = None

            if trade['side'] == 'long':
                if curr_low <= trade['sl']:
                    exit_price = trade['sl']
                elif not trade['tp1_hit'] and curr_high >= trade['tp1']:
                    # Partial exit
                    pqty = trade['qty'] * tp1_pct
                    ppnl = pqty * (trade['tp1'] - trade['entry']) - pqty * trade['tp1'] * fee_rate
                    balance += ppnl
                    trade['partial_pnl'] = ppnl
                    trade['qty'] -= pqty
                    trade['tp1_hit'] = True
                    trade['sl'] = trade['entry']  # Move to BE
                elif trade['tp1_hit'] and curr_high >= trade['tp2']:
                    exit_price = trade['tp2']
                elif trade['tp1_hit'] and curr_low <= trade['entry']:
                    exit_price = trade['entry']
            else:  # short
                if curr_high >= trade['sl']:
                    exit_price = trade['sl']
                elif not trade['tp1_hit'] and curr_low <= trade['tp1']:
                    pqty = trade['qty'] * tp1_pct
                    ppnl = pqty * (trade['entry'] - trade['tp1']) - pqty * trade['tp1'] * fee_rate
                    balance += ppnl
                    trade['partial_pnl'] = ppnl
                    trade['qty'] -= pqty
                    trade['tp1_hit'] = True
                    trade['sl'] = trade['entry']
                elif trade['tp1_hit'] and curr_low <= trade['tp2']:
                    exit_price = trade['tp2']
                elif trade['tp1_hit'] and curr_high >= trade['entry']:
                    exit_price = trade['entry']

            if exit_price is not None:
                if trade['side'] == 'long':
                    pnl = trade['qty'] * (exit_price - trade['entry'])
                else:
                    pnl = trade['qty'] * (trade['entry'] - exit_price)
                pnl += trade.get('partial_pnl', 0)
                pnl -= trade['qty'] * exit_price * fee_rate
                balance += pnl - trade.get('partial_pnl', 0)
                trades_pnl.append(pnl)
            else:
                new_open.append(trade)

        open_trades = new_open

        # New entries
        while setup_idx < len(setups) and setups[setup_idx]['entry_bar'] <= i:
            setup = setups[setup_idx]
            setup_idx += 1

            if setup['entry_bar'] != i or len(open_trades) >= 10:
                continue

            sizing_bal = min(max(balance, 0), initial_balance * 10)
            risk_amt = sizing_bal * risk_pct / 100
            risk_per_unit = abs(setup['entry_price'] - setup['stop_loss'])

            if risk_per_unit <= 0 or risk_amt <= 0:
                continue

            qty = (risk_amt / risk_per_unit) * (entry_split / 100)
            qty = min(qty, sizing_bal * 0.5 / setup['entry_price'])

            if qty <= 0:
                continue

            open_trades.append({
                'side': setup['direction'],
                'entry': setup['entry_price'],
                'qty': qty,
                'sl': setup['stop_loss'],
                'tp1': setup['tp1'],
                'tp2': setup['tp2'],
                'tp1_hit': False,
                'partial_pnl': 0
            })

        # Record equity periodically
        if i % 100 == 0:
            unreal = sum(
                t['qty'] * (curr_close - t['entry']) if t['side'] == 'long'
                else t['qty'] * (t['entry'] - curr_close)
                for t in open_trades
            )
            equity.append(balance + unreal)

    # Close remaining
    for trade in open_trades:
        if trade['side'] == 'long':
            pnl = trade['qty'] * (close[-1] - trade['entry'])
        else:
            pnl = trade['qty'] * (trade['entry'] - close[-1])
        pnl += trade.get('partial_pnl', 0) - trade['qty'] * close[-1] * fee_rate
        trades_pnl.append(pnl)
        balance += pnl - trade.get('partial_pnl', 0)

    equity.append(balance)

    # Calculate metrics
    equity_arr = np.array(equity)
    total_return_pct = (equity_arr[-1] - initial_balance) / initial_balance * 100

    returns = np.diff(equity_arr) / equity_arr[:-1]
    returns = returns[~np.isnan(returns) & ~np.isinf(returns)]
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 1 and np.std(returns) > 0 else 0

    peak = np.maximum.accumulate(equity_arr)
    dd = (peak - equity_arr) / peak * 100
    max_dd = np.max(dd)

    wins = [p for p in trades_pnl if p > 0]
    losses = [p for p in trades_pnl if p <= 0]
    win_rate = len(wins) / len(trades_pnl) if trades_pnl else 0
    pf = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else sum(wins) if wins else 0

    return {
        'total_return_pct': total_return_pct,
        'sharpe_ratio': sharpe,
        'max_drawdown_pct': max_dd,
        'total_trades': len(trades_pnl),
        'win_rate': win_rate,
        'profit_factor': pf
    }


# =============================================================================
# GIGA OPTIMIZER FAST
# =============================================================================

class GigaOptimizerFast:
    """
    Fast giga optimizer with pivot caching.
    """

    def __init__(self, output_dir: str = 'output/giga_optimization'):
        self.output_dir = output_dir
        self.storage = DataStorage('data')
        self.pivot_cache = FastPivotCache()

        os.makedirs(output_dir, exist_ok=True)

        self.results = []
        self.results_file = os.path.join(output_dir, 'all_results.csv')
        self._init_csv()

    def _init_csv(self):
        if not os.path.exists(self.results_file):
            headers = [
                'run_id', 'timestamp', 'asset', 'timeframe', 'module',
                'atr_multiplier', 'risk_pct', 'entry_split',
                'sl_approach', 'sl_multiplier',
                'tp1_extension', 'tp2_extension',
                'confirmation_delay',
                'train_return', 'train_sharpe', 'train_dd', 'train_trades',
                'test_return', 'test_sharpe', 'test_dd', 'test_trades',
                'is_valid'
            ]
            pd.DataFrame(columns=headers).to_csv(self.results_file, index=False)

    def _save_result(self, result: Dict):
        pd.DataFrame([result]).to_csv(self.results_file, mode='a', header=False, index=False)

    def run(
        self,
        assets: List[str] = ['BTCUSDT', 'ETHUSDT'],
        timeframes: List[str] = ['4h', '1h', '15m', '5m', '1m'],  # Fastest first
        modules: List[str] = ['A', 'B', 'C'],
        max_drawdown: float = 30.0
    ):
        """Run optimization with fast caching."""

        # Simplified parameter grid
        atr_mults = [1.0, 1.5, 2.0]
        risk_pcts = [0.5, 1.0, 1.5, 2.0]
        entry_splits = [50.0, 100.0]
        sl_approaches = ['rule_based', 'multiplier']
        sl_multipliers = [0.75, 1.0, 1.5]
        tp1_exts = [1.0, 1.382]
        tp2_exts = [1.618, 2.618]
        conf_delays = [1, 2, 3]

        print("=" * 70)
        print("GIGA OPTIMIZER FAST")
        print("=" * 70)

        run_id = 0
        total_valid = 0

        for module in modules:
            print(f"\n{'='*50}")
            print(f"MODULE {module}")
            print(f"{'='*50}")

            for asset in assets:
                for tf in timeframes:
                    print(f"\n--- {asset} {tf} ---")

                    try:
                        df = self.storage.load(asset, tf)
                        # Minimum data: 500 for 1d, 1000 for 4h, 5000 for others
                        min_candles = 500 if tf == '1d' else (1000 if tf == '4h' else 5000)
                        if df is None or len(df) < min_candles:
                            print(f"  Skipping: insufficient data (need {min_candles}, have {len(df) if df is not None else 0})")
                            continue
                    except Exception as e:
                        print(f"  Skipping: {e}")
                        continue

                    # Prepare data
                    high = df['high'].values
                    low = df['low'].values
                    close = df['close'].values
                    open_prices = df['open'].values
                    atr = calculate_atr(df, 20).values
                    n = len(df)

                    # Split data for walk-forward
                    train_end = int(n * 0.7)

                    best_test_return = -np.inf
                    best_params = None
                    tested = 0

                    # Outer loop: ATR multiplier (pivot-dependent)
                    for atr_mult in atr_mults:
                        # Get pivots (cached)
                        pivots = self.pivot_cache.get_pivots(df, asset, tf, atr_mult)

                        # Inner loop: other params (fast, no pivot recomputation)
                        for risk in risk_pcts:
                            for split in entry_splits:
                                for sl_app in sl_approaches:
                                    sl_mult_list = sl_multipliers if sl_app == 'multiplier' else [1.0]

                                    for sl_mult in sl_mult_list:
                                        for tp1 in tp1_exts:
                                            for tp2 in tp2_exts:
                                                if tp1 >= tp2:
                                                    continue

                                                for conf in conf_delays:
                                                    run_id += 1
                                                    tested += 1

                                                    # Find ALL setups on full data (faster)
                                                    all_setups = find_setups_fast(
                                                        pivots, open_prices,
                                                        high, low, atr, module, conf,
                                                        sl_app, sl_mult, tp1, tp2
                                                    )

                                                    # Split into train/test based on entry_bar
                                                    train_setups = [s for s in all_setups if s['entry_bar'] < train_end]
                                                    test_setups_orig = [s for s in all_setups if s['entry_bar'] >= train_end]

                                                    # Adjust test entry bars relative to test data start
                                                    test_setups = []
                                                    for s in test_setups_orig:
                                                        s_copy = s.copy()
                                                        s_copy['entry_bar'] = s['entry_bar'] - train_end
                                                        test_setups.append(s_copy)

                                                    train_metrics = simulate_trades_fast(
                                                        train_setups, high[:train_end],
                                                        low[:train_end], close[:train_end],
                                                        risk, split
                                                    )

                                                    test_metrics = simulate_trades_fast(
                                                        test_setups, high[train_end:],
                                                        low[train_end:], close[train_end:],
                                                        risk, split
                                                    )

                                                    # Check validity
                                                    is_valid = (
                                                        train_metrics['max_drawdown_pct'] < max_drawdown and
                                                        test_metrics['max_drawdown_pct'] < max_drawdown and
                                                        test_metrics['total_return_pct'] > 0 and
                                                        test_metrics['total_trades'] >= 5
                                                    )

                                                    record = {
                                                        'run_id': run_id,
                                                        'timestamp': datetime.now().isoformat(),
                                                        'asset': asset,
                                                        'timeframe': tf,
                                                        'module': module,
                                                        'atr_multiplier': atr_mult,
                                                        'risk_pct': risk,
                                                        'entry_split': split,
                                                        'sl_approach': sl_app,
                                                        'sl_multiplier': sl_mult,
                                                        'tp1_extension': tp1,
                                                        'tp2_extension': tp2,
                                                        'confirmation_delay': conf,
                                                        'train_return': train_metrics['total_return_pct'],
                                                        'train_sharpe': train_metrics['sharpe_ratio'],
                                                        'train_dd': train_metrics['max_drawdown_pct'],
                                                        'train_trades': train_metrics['total_trades'],
                                                        'test_return': test_metrics['total_return_pct'],
                                                        'test_sharpe': test_metrics['sharpe_ratio'],
                                                        'test_dd': test_metrics['max_drawdown_pct'],
                                                        'test_trades': test_metrics['total_trades'],
                                                        'is_valid': is_valid
                                                    }

                                                    self._save_result(record)
                                                    self.results.append(record)

                                                    if is_valid:
                                                        total_valid += 1
                                                        if test_metrics['total_return_pct'] > best_test_return:
                                                            best_test_return = test_metrics['total_return_pct']
                                                            best_params = record

                    print(f"  Tested: {tested} | Valid: {sum(1 for r in self.results[-tested:] if r['is_valid'])}", end='')
                    if best_params:
                        print(f" | Best: {best_test_return:.1f}%")
                    else:
                        print()

        print("\n" + "=" * 70)
        print(f"COMPLETE: {run_id} runs, {total_valid} valid")
        print(f"Results: {self.results_file}")
        print("=" * 70)

        self._generate_summary()

    def _generate_summary(self):
        """Generate summary"""
        if not self.results:
            return

        df = pd.DataFrame(self.results)
        valid = df[df['is_valid'] == True]

        summary_path = os.path.join(self.output_dir, 'summary.md')

        with open(summary_path, 'w') as f:
            f.write("# Giga Optimization Summary\n\n")
            f.write(f"Total runs: {len(df)}\n")
            f.write(f"Valid runs: {len(valid)}\n\n")

            if len(valid) > 0:
                f.write("## Top 10 by Test Return\n\n")
                top = valid.nlargest(10, 'test_return')
                f.write("| Asset | TF | Module | Test Ret | Train Ret | Test DD | Trades |\n")
                f.write("|-------|-----|--------|----------|-----------|---------|--------|\n")
                for _, r in top.iterrows():
                    f.write(f"| {r['asset']} | {r['timeframe']} | {r['module']} | "
                           f"{r['test_return']:.1f}% | {r['train_return']:.1f}% | "
                           f"{r['test_dd']:.1f}% | {r['test_trades']} |\n")
                f.write("\n")

                f.write("## Best Parameters per Module\n\n")
                for mod in ['A', 'B', 'C']:
                    mod_df = valid[valid['module'] == mod]
                    if len(mod_df) > 0:
                        best = mod_df.loc[mod_df['test_return'].idxmax()]
                        f.write(f"### Module {mod}\n")
                        f.write(f"- Test Return: {best['test_return']:.2f}%\n")
                        f.write(f"- Asset/TF: {best['asset']} {best['timeframe']}\n")
                        f.write(f"- ATR: {best['atr_multiplier']}, Risk: {best['risk_pct']}%\n")
                        f.write(f"- SL: {best['sl_approach']} ({best['sl_multiplier']})\n")
                        f.write(f"- TP: {best['tp1_extension']}/{best['tp2_extension']}\n")
                        f.write(f"- Confirmation: {best['confirmation_delay']} bars\n\n")

        print(f"Summary saved: {summary_path}")


def main():
    optimizer = GigaOptimizerFast()
    optimizer.run(
        assets=['BTCUSDT', 'ETHUSDT'],
        timeframes=['4h', '1h', '15m', '5m', '1m'],
        modules=['A', 'B', 'C'],
        max_drawdown=30.0
    )


if __name__ == '__main__':
    main()
