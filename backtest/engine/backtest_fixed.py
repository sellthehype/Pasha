"""
FIXED Backtest Engine - No Look-Ahead Bias

Key fixes from original backtest_optimized.py:
1. Pivots track CONFIRMATION bar, not just pivot bar
2. Entries happen at confirmation bar (realistic timing)
3. Entry price is next bar's open (can't enter instantly)
4. Stop loss and take profits recalculated from actual entry price
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from tqdm import tqdm
import warnings

from ..config.settings import Config
from ..indicators.atr import calculate_atr
from .portfolio import Portfolio
from .metrics import MetricsCalculator, PerformanceMetrics


@dataclass
class Trade:
    """Trade record"""
    entry_idx: int
    entry_price: float
    entry_time: datetime
    side: str
    module: str
    quantity: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float

    exit_idx: int = None
    exit_price: float = None
    exit_time: datetime = None
    exit_reason: str = None
    pnl: float = 0.0
    fees: float = 0.0
    tp1_hit: bool = False
    partial_exit_pnl: float = 0.0


@dataclass
class FixedBacktestResult:
    """Results from fixed backtest"""
    symbol: str
    timeframe: str
    config: Config
    metrics: PerformanceMetrics
    trades: List[Trade]
    equity_curve: pd.DataFrame
    signals_generated: int
    signals_executed: int
    pivots_found: int


class RealisticPivotDetector:
    """
    Pivot detection that tracks WHEN pivots are confirmed.
    No look-ahead bias.
    """

    def __init__(self, atr_multiplier: float = 1.5, min_wave_pct: float = 0.5):
        self.atr_multiplier = atr_multiplier
        self.min_wave_pct = min_wave_pct

    def find_pivots_with_confirmation(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        atr: np.ndarray
    ) -> List[Dict]:
        """
        Find pivots and track when they were confirmed.

        Returns list of dicts with:
        - pivot_idx: where the pivot occurred
        - confirmed_idx: when we KNEW it was a pivot
        - price: pivot price
        - type: 1 for high, -1 for low
        """
        n = len(high)
        threshold = np.maximum(atr * self.atr_multiplier, close * self.min_wave_pct / 100)

        pivots = []

        direction = 0
        current_high = high[0]
        current_high_idx = 0
        current_low = low[0]
        current_low_idx = 0

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
                            'confirmed_idx': i,  # CONFIRMED at current bar
                            'price': current_low,
                            'type': -1  # low
                        })
                        current_high = high[i]
                        current_high_idx = i
                    else:
                        direction = -1
                        pivots.append({
                            'pivot_idx': current_high_idx,
                            'confirmed_idx': i,
                            'price': current_high,
                            'type': 1  # high
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

            else:  # direction == -1
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

        return pivots


class RealisticWaveAnalyzer:
    """Wave analysis with realistic entry timing"""

    def __init__(self, fib_tolerance: float = 2.5):
        self.fib_tolerance = fib_tolerance / 100

    def find_wave3_setups(
        self,
        pivots: List[Dict],
        open_prices: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray
    ) -> List[Dict]:
        """
        Find Wave 3 setups with REALISTIC entry timing.
        Entry is at next bar's open after the setup is confirmed.
        """
        setups = []
        n_pivots = len(pivots)

        for i in range(2, n_pivots):
            p0 = pivots[i-2]
            p1 = pivots[i-1]
            p2 = pivots[i]

            # Check for low-high-low (bullish) or high-low-high (bearish)
            if p0['type'] == -1 and p1['type'] == 1 and p2['type'] == -1:
                # Bullish: Wave 1 up, Wave 2 retracement down
                w1_start = p0['price']
                w1_end = p1['price']
                w2_end = p2['price']  # This is the pivot price

                w1_range = w1_end - w1_start
                if w1_range <= 0:
                    continue

                retrace = (w1_end - w2_end) / w1_range

                if 0.382 - self.fib_tolerance <= retrace <= 0.786 + self.fib_tolerance:
                    # Setup is CONFIRMED when p2 is confirmed
                    confirmed_bar = p2['confirmed_idx']

                    # Entry is at NEXT bar's open (realistic - can't enter instantly)
                    entry_bar = confirmed_bar + 1
                    if entry_bar >= len(open_prices):
                        continue

                    entry_price = open_prices[entry_bar]

                    # Stop loss below Wave 1 start
                    stop_loss = w1_start - (w1_range * 0.01)

                    # Take profits based on ACTUAL entry price
                    # TP1 = entry + same distance as original W1 range
                    # TP2 = entry + 1.618 * W1 range
                    tp1 = entry_price + w1_range * 1.0
                    tp2 = entry_price + w1_range * 1.618

                    setups.append({
                        'type': 'wave3',
                        'direction': 'long',
                        'confirmed_bar': confirmed_bar,
                        'entry_bar': entry_bar,
                        'entry_price': entry_price,
                        'pivot_price': w2_end,  # For reference
                        'stop_loss': stop_loss,
                        'tp1': tp1,
                        'tp2': tp2,
                        'w1_range': w1_range,
                        'retrace': retrace
                    })

            elif p0['type'] == 1 and p1['type'] == -1 and p2['type'] == 1:
                # Bearish
                w1_start = p0['price']
                w1_end = p1['price']
                w2_end = p2['price']

                w1_range = w1_start - w1_end
                if w1_range <= 0:
                    continue

                retrace = (w2_end - w1_end) / w1_range

                if 0.382 - self.fib_tolerance <= retrace <= 0.786 + self.fib_tolerance:
                    confirmed_bar = p2['confirmed_idx']
                    entry_bar = confirmed_bar + 1
                    if entry_bar >= len(open_prices):
                        continue

                    entry_price = open_prices[entry_bar]
                    stop_loss = w1_start + (w1_range * 0.01)
                    tp1 = entry_price - w1_range * 1.0
                    tp2 = entry_price - w1_range * 1.618

                    setups.append({
                        'type': 'wave3',
                        'direction': 'short',
                        'confirmed_bar': confirmed_bar,
                        'entry_bar': entry_bar,
                        'entry_price': entry_price,
                        'pivot_price': w2_end,
                        'stop_loss': stop_loss,
                        'tp1': tp1,
                        'tp2': tp2,
                        'w1_range': w1_range,
                        'retrace': retrace
                    })

        return setups

    def find_wave5_setups(
        self,
        pivots: List[Dict],
        open_prices: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray
    ) -> List[Dict]:
        """Find Wave 5 setups with realistic entry timing"""
        setups = []
        n_pivots = len(pivots)

        for i in range(4, n_pivots):
            p0 = pivots[i-4]
            p1 = pivots[i-3]
            p2 = pivots[i-2]
            p3 = pivots[i-1]
            p4 = pivots[i]

            # Check for alternating pivots
            types_bull = (p0['type'] == -1 and p1['type'] == 1 and
                         p2['type'] == -1 and p3['type'] == 1 and p4['type'] == -1)
            types_bear = (p0['type'] == 1 and p1['type'] == -1 and
                         p2['type'] == 1 and p3['type'] == -1 and p4['type'] == 1)

            if types_bull:
                w1_start = p0['price']
                w1_end = p1['price']
                w2_end = p2['price']
                w3_end = p3['price']
                w4_end = p4['price']

                # Validate Elliott Wave rules
                if w2_end < w1_start:
                    continue
                if w4_end < w1_end:
                    continue

                w3_range = w3_end - w2_end
                w1_range = w1_end - w1_start
                if w3_range < w1_range * 0.5:
                    continue

                retrace = (w3_end - w4_end) / w3_range if w3_range > 0 else 0

                if 0.236 - self.fib_tolerance <= retrace <= 0.5 + self.fib_tolerance:
                    confirmed_bar = p4['confirmed_idx']
                    entry_bar = confirmed_bar + 1
                    if entry_bar >= len(open_prices):
                        continue

                    entry_price = open_prices[entry_bar]
                    stop_loss = w1_end - (w1_range * 0.01)
                    tp1 = w3_end  # Previous high
                    tp2 = w3_end + (w3_end - w1_start) * 0.618

                    setups.append({
                        'type': 'wave5',
                        'direction': 'long',
                        'confirmed_bar': confirmed_bar,
                        'entry_bar': entry_bar,
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'tp1': tp1,
                        'tp2': tp2,
                        'retrace': retrace
                    })

            elif types_bear:
                w1_start = p0['price']
                w1_end = p1['price']
                w2_end = p2['price']
                w3_end = p3['price']
                w4_end = p4['price']

                if w2_end > w1_start:
                    continue
                if w4_end > w1_end:
                    continue

                w3_range = w2_end - w3_end
                w1_range = w1_start - w1_end
                if w3_range < w1_range * 0.5:
                    continue

                retrace = (w4_end - w3_end) / w3_range if w3_range > 0 else 0

                if 0.236 - self.fib_tolerance <= retrace <= 0.5 + self.fib_tolerance:
                    confirmed_bar = p4['confirmed_idx']
                    entry_bar = confirmed_bar + 1
                    if entry_bar >= len(open_prices):
                        continue

                    entry_price = open_prices[entry_bar]
                    stop_loss = w1_end + (w1_range * 0.01)
                    tp1 = w3_end
                    tp2 = w3_end - (w1_start - w3_end) * 0.618

                    setups.append({
                        'type': 'wave5',
                        'direction': 'short',
                        'confirmed_bar': confirmed_bar,
                        'entry_bar': entry_bar,
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'tp1': tp1,
                        'tp2': tp2,
                        'retrace': retrace
                    })

        return setups


class FixedBacktestEngine:
    """
    FIXED Backtest Engine - No Look-Ahead Bias

    Key differences from OptimizedBacktestEngine:
    1. Tracks pivot confirmation timing
    2. Entries at next bar open after confirmation
    3. Stop/TP calculated from actual entry price
    """

    def __init__(self, config: Config):
        self.config = config
        self.pivot_detector = RealisticPivotDetector(
            atr_multiplier=config.zigzag_atr_multiplier,
            min_wave_pct=config.min_wave_size_pct
        )
        self.wave_analyzer = RealisticWaveAnalyzer(
            fib_tolerance=config.fib_tolerance_pct
        )
        self.metrics_calc = MetricsCalculator()

    def run(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        show_progress: bool = True
    ) -> FixedBacktestResult:
        """Run realistic backtest without look-ahead bias"""

        if show_progress:
            print(f"Pre-computing indicators and pivots (with confirmation tracking)...")

        # Convert to numpy
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        open_prices = df['open'].values
        timestamps = df['timestamp'].values

        n = len(df)

        # Pre-compute ATR
        atr = calculate_atr(df, self.config.zigzag_atr_period).values

        # Find pivots WITH confirmation timing
        pivots = self.pivot_detector.find_pivots_with_confirmation(high, low, close, atr)

        if show_progress:
            print(f"Found {len(pivots)} pivots")

        # Find setups with realistic entry timing
        wave3_setups = self.wave_analyzer.find_wave3_setups(
            pivots, open_prices, high, low, close
        )
        wave5_setups = self.wave_analyzer.find_wave5_setups(
            pivots, open_prices, high, low, close
        )

        all_setups = []
        for s in wave3_setups:
            s['module'] = 'A'
            all_setups.append(s)
        for s in wave5_setups:
            s['module'] = 'B'
            all_setups.append(s)

        # Sort by ENTRY bar (not confirmation bar)
        all_setups.sort(key=lambda x: x['entry_bar'])

        if show_progress:
            print(f"Found {len(all_setups)} potential setups")

        # Simulate trades
        trades = []
        equity = [self.config.initial_balance]
        equity_timestamps = [timestamps[0]]

        open_trades: List[Trade] = []
        balance = self.config.initial_balance

        setup_idx = 0

        iterator = range(1, n)
        if show_progress:
            iterator = tqdm(iterator, desc=f"Simulating {symbol} {timeframe}")

        for i in iterator:
            current_high = high[i]
            current_low = low[i]
            current_close = close[i]
            current_time = timestamps[i]

            # Check exits for open trades
            closed_trades = []
            for trade in open_trades:
                exit_price = None
                exit_reason = None

                if trade.side == 'long':
                    if current_low <= trade.stop_loss:
                        exit_price = trade.stop_loss
                        exit_reason = 'Stop Loss'
                    elif not trade.tp1_hit and current_high >= trade.take_profit_1:
                        partial_qty = trade.quantity * (self.config.tp1_pct / 100)
                        partial_pnl = partial_qty * (trade.take_profit_1 - trade.entry_price)
                        partial_fee = partial_qty * trade.take_profit_1 * (self.config.taker_fee_pct / 100)
                        trade.partial_exit_pnl = partial_pnl - partial_fee
                        trade.quantity -= partial_qty
                        trade.tp1_hit = True
                        trade.stop_loss = trade.entry_price
                        balance += trade.partial_exit_pnl
                    elif trade.tp1_hit and current_high >= trade.take_profit_2:
                        exit_price = trade.take_profit_2
                        exit_reason = 'TP2'
                    elif trade.tp1_hit and current_low <= trade.entry_price:
                        exit_price = trade.entry_price
                        exit_reason = 'Breakeven'

                else:  # short
                    if current_high >= trade.stop_loss:
                        exit_price = trade.stop_loss
                        exit_reason = 'Stop Loss'
                    elif not trade.tp1_hit and current_low <= trade.take_profit_1:
                        partial_qty = trade.quantity * (self.config.tp1_pct / 100)
                        partial_pnl = partial_qty * (trade.entry_price - trade.take_profit_1)
                        partial_fee = partial_qty * trade.take_profit_1 * (self.config.taker_fee_pct / 100)
                        trade.partial_exit_pnl = partial_pnl - partial_fee
                        trade.quantity -= partial_qty
                        trade.tp1_hit = True
                        trade.stop_loss = trade.entry_price
                        balance += trade.partial_exit_pnl
                    elif trade.tp1_hit and current_low <= trade.take_profit_2:
                        exit_price = trade.take_profit_2
                        exit_reason = 'TP2'
                    elif trade.tp1_hit and current_high >= trade.entry_price:
                        exit_price = trade.entry_price
                        exit_reason = 'Breakeven'

                if exit_price is not None:
                    trade.exit_idx = i
                    trade.exit_price = exit_price
                    trade.exit_time = current_time
                    trade.exit_reason = exit_reason

                    if trade.side == 'long':
                        trade.pnl = trade.quantity * (exit_price - trade.entry_price)
                    else:
                        trade.pnl = trade.quantity * (trade.entry_price - exit_price)

                    trade.pnl += trade.partial_exit_pnl
                    exit_fee = trade.quantity * exit_price * (self.config.taker_fee_pct / 100)
                    trade.fees += exit_fee
                    trade.pnl -= exit_fee

                    balance += trade.pnl - trade.partial_exit_pnl
                    trades.append(trade)
                    closed_trades.append(trade)

            for t in closed_trades:
                open_trades.remove(t)

            # Check for new entries (at entry_bar, not confirmation bar)
            while setup_idx < len(all_setups) and all_setups[setup_idx]['entry_bar'] <= i:
                setup = all_setups[setup_idx]
                setup_idx += 1

                # Only enter on the exact entry bar
                if setup['entry_bar'] != i:
                    continue

                max_pos = getattr(self.config, 'max_positions', 10)
                if len(open_trades) >= max_pos:
                    continue

                # Position sizing
                sizing_balance = min(balance, self.config.initial_balance * 10)
                sizing_balance = max(sizing_balance, 0)

                risk_amount = sizing_balance * (self.config.base_risk_pct / 100)
                risk_per_unit = abs(setup['entry_price'] - setup['stop_loss'])

                if risk_per_unit <= 0 or risk_amount <= 0:
                    continue

                quantity = risk_amount / risk_per_unit
                quantity *= (self.config.initial_position_pct / 100)

                max_qty = (sizing_balance * 0.5) / setup['entry_price']
                quantity = min(quantity, max_qty)

                if quantity <= 0:
                    continue

                entry_fee = quantity * setup['entry_price'] * (self.config.maker_fee_pct / 100)

                trade = Trade(
                    entry_idx=setup['entry_bar'],
                    entry_price=setup['entry_price'],
                    entry_time=timestamps[setup['entry_bar']],
                    side=setup['direction'],
                    module=setup['module'],
                    quantity=quantity,
                    stop_loss=setup['stop_loss'],
                    take_profit_1=setup['tp1'],
                    take_profit_2=setup['tp2'],
                    fees=entry_fee
                )

                open_trades.append(trade)

            # Record equity periodically
            if i % 100 == 0 or i == n - 1:
                unrealized = 0
                for trade in open_trades:
                    if trade.side == 'long':
                        unrealized += trade.quantity * (current_close - trade.entry_price)
                    else:
                        unrealized += trade.quantity * (trade.entry_price - current_close)

                equity.append(balance + unrealized)
                equity_timestamps.append(current_time)

        # Close remaining trades
        for trade in open_trades:
            trade.exit_idx = n - 1
            trade.exit_price = close[-1]
            trade.exit_time = timestamps[-1]
            trade.exit_reason = 'End of backtest'

            if trade.side == 'long':
                trade.pnl = trade.quantity * (trade.exit_price - trade.entry_price)
            else:
                trade.pnl = trade.quantity * (trade.entry_price - trade.exit_price)

            trade.pnl += trade.partial_exit_pnl
            exit_fee = trade.quantity * trade.exit_price * (self.config.taker_fee_pct / 100)
            trade.fees += exit_fee
            trade.pnl -= exit_fee

            balance += trade.pnl - trade.partial_exit_pnl
            trades.append(trade)

        # Build equity curve
        equity_df = pd.DataFrame({
            'timestamp': pd.to_datetime(equity_timestamps),
            'equity': equity
        })

        # Calculate metrics
        metrics = self._calculate_metrics(trades, equity_df, self.config.initial_balance)

        return FixedBacktestResult(
            symbol=symbol,
            timeframe=timeframe,
            config=self.config,
            metrics=metrics,
            trades=trades,
            equity_curve=equity_df,
            signals_generated=len(all_setups),
            signals_executed=len(trades),
            pivots_found=len(pivots)
        )

    def _calculate_metrics(
        self,
        trades: List[Trade],
        equity_df: pd.DataFrame,
        initial_balance: float
    ) -> PerformanceMetrics:
        """Calculate performance metrics"""

        metrics = PerformanceMetrics()

        if not trades:
            return metrics

        final_equity = equity_df['equity'].iloc[-1]
        metrics.total_return = final_equity - initial_balance
        metrics.total_return_pct = (final_equity - initial_balance) / initial_balance * 100

        equity_values = equity_df['equity'].values
        returns = np.diff(equity_values) / equity_values[:-1]
        returns = returns[~np.isnan(returns) & ~np.isinf(returns)]

        if len(returns) > 1 and np.std(returns) > 0:
            metrics.sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)

        negative_returns = returns[returns < 0]
        if len(negative_returns) > 1 and np.std(negative_returns) > 0:
            metrics.sortino_ratio = np.mean(returns) / np.std(negative_returns) * np.sqrt(252)
        else:
            metrics.sortino_ratio = metrics.sharpe_ratio

        peak = np.maximum.accumulate(equity_values)
        drawdown = (peak - equity_values) / peak * 100
        metrics.max_drawdown_pct = np.max(drawdown)
        metrics.max_drawdown = np.max(peak - equity_values)

        pnls = [t.pnl for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        metrics.total_trades = len(trades)
        metrics.winning_trades = len(wins)
        metrics.losing_trades = len(losses)
        metrics.win_rate = len(wins) / len(trades) if trades else 0

        if wins:
            metrics.average_win = sum(wins) / len(wins)
            metrics.largest_win = max(wins)

        if losses:
            metrics.average_loss = sum(losses) / len(losses)
            metrics.largest_loss = min(losses)

        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 1
        metrics.profit_factor = gross_profit / gross_loss if gross_loss > 0 else gross_profit

        metrics.expectancy = np.mean(pnls) if pnls else 0

        return metrics
