"""
Visual Verification Chart Generator

Generates a self-contained HTML file for visually verifying backtest correctness.
Uses Plotly.js for interactive candlestick charts with overlaid wave patterns,
trade entries/exits, and timeline scrubbing.
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field, asdict
import html

from ..config.settings import Config
from ..indicators.atr import calculate_atr
from ..engine.backtest_fixed import (
    RealisticPivotDetector,
    RealisticWaveAnalyzer,
    Trade,
    FixedBacktestResult
)


@dataclass
class DetailedTrade:
    """Trade with full detail for visualization"""
    id: int
    module: str  # 'A' or 'B'
    pattern_id: int
    direction: str  # 'long' or 'short'

    # Entry
    entry_bar: int
    entry_price: float
    entry_date: str
    quantity: float
    initial_quantity: float

    # Levels
    stop_loss: float
    tp1: float
    tp2: float

    # Exits
    tp1_bar: Optional[int] = None
    tp1_price: Optional[float] = None
    tp1_qty: Optional[float] = None

    final_exit_bar: Optional[int] = None
    final_exit_price: Optional[float] = None
    final_exit_reason: Optional[str] = None

    # P&L
    fees_entry: float = 0.0
    fees_tp1: float = 0.0
    fees_exit: float = 0.0
    pnl_gross: float = 0.0
    pnl_net: float = 0.0

    # Balance tracking
    balance_before: float = 0.0
    balance_after: float = 0.0

    # For audit
    confirmed_bar: int = 0
    pivot_bar: int = 0
    retrace_pct: float = 0.0

    # Wave structure
    wave_pivots: List[int] = field(default_factory=list)
    wave_prices: List[float] = field(default_factory=list)


@dataclass
class DetailedPattern:
    """Wave pattern with full detail"""
    id: int
    type: str  # 'wave3' or 'wave5'
    module: str
    direction: str

    # Pivots that form this pattern
    pivot_indices: List[int]
    pivot_prices: List[float]
    pivot_types: List[int]  # 1 for high, -1 for low

    # Entry zone
    entry_zone_low: float
    entry_zone_high: float

    # Levels
    tp1: float
    tp2: float
    stop_loss: float

    # Timing
    confirmed_bar: int
    entry_bar: int

    # Status
    traded: bool
    trade_id: Optional[int] = None
    entry_price: Optional[float] = None
    retrace_pct: float = 0.0


@dataclass
class DetailedPivot:
    """Pivot with confirmation timing"""
    idx: int
    pivot_bar: int
    confirmed_bar: int
    price: float
    type: int  # 1 for high, -1 for low


@dataclass
class EquityPoint:
    """Equity curve point"""
    bar: int
    date: str
    equity: float
    drawdown_pct: float


@dataclass
class AuditIssue:
    """Audit finding"""
    type: str  # 'timing', 'price', 'wave', 'data'
    severity: str  # 'warning', 'error'
    trade_id: Optional[int]
    bar: int
    message: str


class VerificationDataCollector:
    """
    Collects all data needed for visual verification.
    Re-runs the backtest logic while tracking additional details.
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

    def collect(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str
    ) -> Dict:
        """
        Run backtest and collect all detailed data for visualization.
        """
        # Convert to numpy
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        open_prices = df['open'].values
        timestamps = pd.to_datetime(df['timestamp']).values

        n = len(df)

        # Pre-compute ATR
        atr = calculate_atr(df, self.config.zigzag_atr_period).values

        # Find pivots WITH confirmation timing
        raw_pivots = self.pivot_detector.find_pivots_with_confirmation(high, low, close, atr)

        # Convert to DetailedPivot
        pivots = []
        for i, p in enumerate(raw_pivots):
            pivots.append(DetailedPivot(
                idx=i,
                pivot_bar=p['pivot_idx'],
                confirmed_bar=p['confirmed_idx'],
                price=p['price'],
                type=p['type']
            ))

        # Find wave setups based on config
        all_patterns = []

        # Module A: Wave 3 setups
        if getattr(self.config, 'module_a_enabled', True):
            wave3_setups = self._find_wave3_setups_detailed(
                raw_pivots, open_prices, high, low, close
            )
            all_patterns.extend(wave3_setups)

        # Module B: Wave 5 setups
        if getattr(self.config, 'module_b_enabled', True):
            wave5_setups = self._find_wave5_setups_detailed(
                raw_pivots, open_prices, high, low, close
            )
            all_patterns.extend(wave5_setups)

        # Module C: Corrective patterns
        if getattr(self.config, 'module_c_enabled', True):
            if getattr(self.config, 'module_c_zigzag_enabled', True):
                zigzag_setups = self._find_zigzag_setups_detailed(
                    raw_pivots, open_prices, high, low, close
                )
                all_patterns.extend(zigzag_setups)

            if getattr(self.config, 'module_c_flat_enabled', True):
                flat_setups = self._find_flat_setups_detailed(
                    raw_pivots, open_prices, high, low, close
                )
                all_patterns.extend(flat_setups)

            if getattr(self.config, 'module_c_triangle_enabled', True):
                triangle_setups = self._find_triangle_setups_detailed(
                    raw_pivots, open_prices, high, low, close
                )
                all_patterns.extend(triangle_setups)

        for i, p in enumerate(all_patterns):
            p.id = i

        # Sort by entry bar
        all_patterns.sort(key=lambda x: x.entry_bar)

        # Run simulation
        trades, equity_curve, audit_issues = self._simulate_trades(
            all_patterns, open_prices, high, low, close, timestamps
        )

        # Build candle data
        candles = []
        for i in range(n):
            candles.append({
                'bar': i,
                'date': str(pd.Timestamp(timestamps[i]).strftime('%Y-%m-%d %H:%M')),
                'open': float(open_prices[i]),
                'high': float(high[i]),
                'low': float(low[i]),
                'close': float(close[i])
            })

        # Calculate metrics
        metrics = self._calculate_metrics(trades, equity_curve)

        # Additional data validation
        data_issues = self._validate_data(df, candles)
        audit_issues.extend(data_issues)

        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'candles': candles,
            'pivots': [asdict(p) for p in pivots],
            'patterns': [self._pattern_to_dict(p) for p in all_patterns],
            'trades': [self._trade_to_dict(t) for t in trades],
            'equity': [asdict(e) for e in equity_curve],
            'audit': [asdict(a) for a in audit_issues],
            'metrics': metrics
        }

    def _find_wave3_setups_detailed(
        self,
        pivots: List[Dict],
        open_prices: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray
    ) -> List[DetailedPattern]:
        """Find Wave 3 setups with full detail for visualization"""
        patterns = []
        n_pivots = len(pivots)

        for i in range(2, n_pivots):
            p0 = pivots[i-2]
            p1 = pivots[i-1]
            p2 = pivots[i]

            # Check for low-high-low (bullish)
            if p0['type'] == -1 and p1['type'] == 1 and p2['type'] == -1:
                w1_start = p0['price']
                w1_end = p1['price']
                w2_end = p2['price']

                w1_range = w1_end - w1_start
                if w1_range <= 0:
                    continue

                retrace = (w1_end - w2_end) / w1_range
                fib_tol = self.wave_analyzer.fib_tolerance

                if 0.382 - fib_tol <= retrace <= 0.786 + fib_tol:
                    confirmed_bar = p2['confirmed_idx']
                    entry_bar = confirmed_bar + 1
                    if entry_bar >= len(open_prices):
                        continue

                    entry_price = open_prices[entry_bar]
                    stop_loss = w1_start - (w1_range * 0.01)
                    tp1 = entry_price + w1_range * 1.0
                    tp2 = entry_price + w1_range * 1.618

                    # Entry zone based on Fib levels
                    zone_low = w1_end - w1_range * 0.786
                    zone_high = w1_end - w1_range * 0.382

                    patterns.append(DetailedPattern(
                        id=0,
                        type='wave3',
                        module='A',
                        direction='long',
                        pivot_indices=[p0['pivot_idx'], p1['pivot_idx'], p2['pivot_idx']],
                        pivot_prices=[p0['price'], p1['price'], p2['price']],
                        pivot_types=[p0['type'], p1['type'], p2['type']],
                        entry_zone_low=zone_low,
                        entry_zone_high=zone_high,
                        tp1=tp1,
                        tp2=tp2,
                        stop_loss=stop_loss,
                        confirmed_bar=confirmed_bar,
                        entry_bar=entry_bar,
                        traded=False,
                        entry_price=entry_price,
                        retrace_pct=retrace * 100
                    ))

            # Check for high-low-high (bearish)
            elif p0['type'] == 1 and p1['type'] == -1 and p2['type'] == 1:
                w1_start = p0['price']
                w1_end = p1['price']
                w2_end = p2['price']

                w1_range = w1_start - w1_end
                if w1_range <= 0:
                    continue

                retrace = (w2_end - w1_end) / w1_range
                fib_tol = self.wave_analyzer.fib_tolerance

                if 0.382 - fib_tol <= retrace <= 0.786 + fib_tol:
                    confirmed_bar = p2['confirmed_idx']
                    entry_bar = confirmed_bar + 1
                    if entry_bar >= len(open_prices):
                        continue

                    entry_price = open_prices[entry_bar]
                    stop_loss = w1_start + (w1_range * 0.01)
                    tp1 = entry_price - w1_range * 1.0
                    tp2 = entry_price - w1_range * 1.618

                    zone_low = w1_end + w1_range * 0.382
                    zone_high = w1_end + w1_range * 0.786

                    patterns.append(DetailedPattern(
                        id=0,
                        type='wave3',
                        module='A',
                        direction='short',
                        pivot_indices=[p0['pivot_idx'], p1['pivot_idx'], p2['pivot_idx']],
                        pivot_prices=[p0['price'], p1['price'], p2['price']],
                        pivot_types=[p0['type'], p1['type'], p2['type']],
                        entry_zone_low=zone_low,
                        entry_zone_high=zone_high,
                        tp1=tp1,
                        tp2=tp2,
                        stop_loss=stop_loss,
                        confirmed_bar=confirmed_bar,
                        entry_bar=entry_bar,
                        traded=False,
                        entry_price=entry_price,
                        retrace_pct=retrace * 100
                    ))

        return patterns

    def _find_wave5_setups_detailed(
        self,
        pivots: List[Dict],
        open_prices: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray
    ) -> List[DetailedPattern]:
        """Find Wave 5 setups with full detail"""
        patterns = []
        n_pivots = len(pivots)

        for i in range(4, n_pivots):
            p0 = pivots[i-4]
            p1 = pivots[i-3]
            p2 = pivots[i-2]
            p3 = pivots[i-1]
            p4 = pivots[i]

            # Bullish pattern
            types_bull = (p0['type'] == -1 and p1['type'] == 1 and
                         p2['type'] == -1 and p3['type'] == 1 and p4['type'] == -1)
            # Bearish pattern
            types_bear = (p0['type'] == 1 and p1['type'] == -1 and
                         p2['type'] == 1 and p3['type'] == -1 and p4['type'] == 1)

            if types_bull:
                w1_start = p0['price']
                w1_end = p1['price']
                w2_end = p2['price']
                w3_end = p3['price']
                w4_end = p4['price']

                # Validate Elliott rules
                if w2_end < w1_start:
                    continue
                if w4_end < w1_end:
                    continue

                w3_range = w3_end - w2_end
                w1_range = w1_end - w1_start
                if w3_range < w1_range * 0.5:
                    continue

                retrace = (w3_end - w4_end) / w3_range if w3_range > 0 else 0
                fib_tol = self.wave_analyzer.fib_tolerance

                if 0.236 - fib_tol <= retrace <= 0.5 + fib_tol:
                    confirmed_bar = p4['confirmed_idx']
                    entry_bar = confirmed_bar + 1
                    if entry_bar >= len(open_prices):
                        continue

                    entry_price = open_prices[entry_bar]
                    stop_loss = w1_end - (w1_range * 0.01)
                    tp1 = w3_end
                    tp2 = w3_end + (w3_end - w1_start) * 0.618

                    # Validate: for long, entry must be below TP1 and above SL
                    if entry_price >= tp1 or entry_price <= stop_loss:
                        continue

                    zone_low = w3_end - w3_range * 0.5
                    zone_high = w3_end - w3_range * 0.236

                    patterns.append(DetailedPattern(
                        id=0,
                        type='wave5',
                        module='B',
                        direction='long',
                        pivot_indices=[p0['pivot_idx'], p1['pivot_idx'], p2['pivot_idx'],
                                      p3['pivot_idx'], p4['pivot_idx']],
                        pivot_prices=[p0['price'], p1['price'], p2['price'],
                                     p3['price'], p4['price']],
                        pivot_types=[p0['type'], p1['type'], p2['type'],
                                    p3['type'], p4['type']],
                        entry_zone_low=zone_low,
                        entry_zone_high=zone_high,
                        tp1=tp1,
                        tp2=tp2,
                        stop_loss=stop_loss,
                        confirmed_bar=confirmed_bar,
                        entry_bar=entry_bar,
                        traded=False,
                        entry_price=entry_price,
                        retrace_pct=retrace * 100
                    ))

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
                fib_tol = self.wave_analyzer.fib_tolerance

                if 0.236 - fib_tol <= retrace <= 0.5 + fib_tol:
                    confirmed_bar = p4['confirmed_idx']
                    entry_bar = confirmed_bar + 1
                    if entry_bar >= len(open_prices):
                        continue

                    entry_price = open_prices[entry_bar]
                    stop_loss = w1_end + (w1_range * 0.01)
                    tp1 = w3_end
                    tp2 = w3_end - (w1_start - w3_end) * 0.618

                    # Validate: for short, entry must be above TP1 and below SL
                    if entry_price <= tp1 or entry_price >= stop_loss:
                        continue

                    zone_low = w3_end + w3_range * 0.236
                    zone_high = w3_end + w3_range * 0.5

                    patterns.append(DetailedPattern(
                        id=0,
                        type='wave5',
                        module='B',
                        direction='short',
                        pivot_indices=[p0['pivot_idx'], p1['pivot_idx'], p2['pivot_idx'],
                                      p3['pivot_idx'], p4['pivot_idx']],
                        pivot_prices=[p0['price'], p1['price'], p2['price'],
                                     p3['price'], p4['price']],
                        pivot_types=[p0['type'], p1['type'], p2['type'],
                                    p3['type'], p4['type']],
                        entry_zone_low=zone_low,
                        entry_zone_high=zone_high,
                        tp1=tp1,
                        tp2=tp2,
                        stop_loss=stop_loss,
                        confirmed_bar=confirmed_bar,
                        entry_bar=entry_bar,
                        traded=False,
                        entry_price=entry_price,
                        retrace_pct=retrace * 100
                    ))

        return patterns

    def _find_zigzag_setups_detailed(
        self,
        pivots: List[Dict],
        open_prices: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray
    ) -> List[DetailedPattern]:
        """Find Zigzag (A-B-C) setups with full detail for visualization"""
        patterns = []
        n_pivots = len(pivots)
        fib_tol = self.wave_analyzer.fib_tolerance
        b_min = getattr(self.config, 'zigzag_b_min_retrace', 0.382)
        b_max = getattr(self.config, 'zigzag_b_max_retrace', 0.786)

        for i in range(2, n_pivots):
            p0 = pivots[i-2]
            p1 = pivots[i-1]
            p2 = pivots[i]

            # Bullish zigzag (down-up for trade): high-low-high pattern
            if p0['type'] == 1 and p1['type'] == -1 and p2['type'] == 1:
                a_start = p0['price']
                a_end = p1['price']
                b_end = p2['price']
                a_range = a_start - a_end
                if a_range <= 0:
                    continue

                b_retrace = (b_end - a_end) / a_range
                if b_min - fib_tol <= b_retrace <= b_max + fib_tol and b_end < a_start:
                    confirmed_bar = p2['confirmed_idx']
                    entry_bar = confirmed_bar + 1
                    if entry_bar >= len(open_prices):
                        continue

                    entry_price = open_prices[entry_bar]
                    stop_loss = a_start + (a_range * 0.02)
                    tp1 = entry_price - a_range * 0.618
                    tp2 = entry_price - a_range * 1.0

                    # Validate: for short, entry must be below SL and above TP1
                    if entry_price >= stop_loss or entry_price <= tp1:
                        continue

                    patterns.append(DetailedPattern(
                        id=0,
                        type='zigzag',
                        module='C',
                        direction='short',
                        pivot_indices=[p0['pivot_idx'], p1['pivot_idx'], p2['pivot_idx']],
                        pivot_prices=[p0['price'], p1['price'], p2['price']],
                        pivot_types=[p0['type'], p1['type'], p2['type']],
                        entry_zone_low=min(entry_price, tp1),
                        entry_zone_high=max(entry_price, stop_loss),
                        tp1=tp1,
                        tp2=tp2,
                        stop_loss=stop_loss,
                        confirmed_bar=confirmed_bar,
                        entry_bar=entry_bar,
                        traded=False,
                        entry_price=entry_price,
                        retrace_pct=b_retrace * 100
                    ))

            # Bearish zigzag (up for trade): low-high-low pattern
            elif p0['type'] == -1 and p1['type'] == 1 and p2['type'] == -1:
                a_start = p0['price']
                a_end = p1['price']
                b_end = p2['price']
                a_range = a_end - a_start
                if a_range <= 0:
                    continue

                b_retrace = (a_end - b_end) / a_range
                if b_min - fib_tol <= b_retrace <= b_max + fib_tol and b_end > a_start:
                    confirmed_bar = p2['confirmed_idx']
                    entry_bar = confirmed_bar + 1
                    if entry_bar >= len(open_prices):
                        continue

                    entry_price = open_prices[entry_bar]
                    stop_loss = a_start - (a_range * 0.02)
                    tp1 = entry_price + a_range * 0.618
                    tp2 = entry_price + a_range * 1.0

                    # Validate: for long, entry must be above SL and below TP1
                    if entry_price <= stop_loss or entry_price >= tp1:
                        continue

                    patterns.append(DetailedPattern(
                        id=0,
                        type='zigzag',
                        module='C',
                        direction='long',
                        pivot_indices=[p0['pivot_idx'], p1['pivot_idx'], p2['pivot_idx']],
                        pivot_prices=[p0['price'], p1['price'], p2['price']],
                        pivot_types=[p0['type'], p1['type'], p2['type']],
                        entry_zone_low=min(entry_price, stop_loss),
                        entry_zone_high=max(entry_price, tp1),
                        tp1=tp1,
                        tp2=tp2,
                        stop_loss=stop_loss,
                        confirmed_bar=confirmed_bar,
                        entry_bar=entry_bar,
                        traded=False,
                        entry_price=entry_price,
                        retrace_pct=b_retrace * 100
                    ))

        return patterns

    def _find_flat_setups_detailed(
        self,
        pivots: List[Dict],
        open_prices: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray
    ) -> List[DetailedPattern]:
        """Find Flat (A-B-C) setups with full detail for visualization"""
        patterns = []
        n_pivots = len(pivots)
        fib_tol = self.wave_analyzer.fib_tolerance
        b_min = getattr(self.config, 'flat_b_min_retrace', 0.90)
        b_max = getattr(self.config, 'flat_b_max_retrace', 1.38)

        for i in range(2, n_pivots):
            p0 = pivots[i-2]
            p1 = pivots[i-1]
            p2 = pivots[i]

            # Bullish flat context: high-low-high
            if p0['type'] == 1 and p1['type'] == -1 and p2['type'] == 1:
                a_start = p0['price']
                a_end = p1['price']
                b_end = p2['price']
                a_range = a_start - a_end
                if a_range <= 0:
                    continue

                b_retrace = (b_end - a_end) / a_range
                if b_min - fib_tol <= b_retrace <= b_max + fib_tol:
                    confirmed_bar = p2['confirmed_idx']
                    entry_bar = confirmed_bar + 1
                    if entry_bar >= len(open_prices):
                        continue

                    entry_price = open_prices[entry_bar]
                    stop_loss = b_end + (a_range * 0.05)
                    tp1 = entry_price - a_range * 0.618
                    tp2 = entry_price - a_range * 1.0

                    # Validate: for short, entry must be below SL and above TP1
                    if entry_price >= stop_loss or entry_price <= tp1:
                        continue

                    patterns.append(DetailedPattern(
                        id=0,
                        type='flat',
                        module='C',
                        direction='short',
                        pivot_indices=[p0['pivot_idx'], p1['pivot_idx'], p2['pivot_idx']],
                        pivot_prices=[p0['price'], p1['price'], p2['price']],
                        pivot_types=[p0['type'], p1['type'], p2['type']],
                        entry_zone_low=min(entry_price, tp1),
                        entry_zone_high=max(entry_price, stop_loss),
                        tp1=tp1,
                        tp2=tp2,
                        stop_loss=stop_loss,
                        confirmed_bar=confirmed_bar,
                        entry_bar=entry_bar,
                        traded=False,
                        entry_price=entry_price,
                        retrace_pct=b_retrace * 100
                    ))

            # Bearish flat context: low-high-low
            elif p0['type'] == -1 and p1['type'] == 1 and p2['type'] == -1:
                a_start = p0['price']
                a_end = p1['price']
                b_end = p2['price']
                a_range = a_end - a_start
                if a_range <= 0:
                    continue

                b_retrace = (a_end - b_end) / a_range
                if b_min - fib_tol <= b_retrace <= b_max + fib_tol:
                    confirmed_bar = p2['confirmed_idx']
                    entry_bar = confirmed_bar + 1
                    if entry_bar >= len(open_prices):
                        continue

                    entry_price = open_prices[entry_bar]
                    stop_loss = b_end - (a_range * 0.05)
                    tp1 = entry_price + a_range * 0.618
                    tp2 = entry_price + a_range * 1.0

                    # Validate: for long, entry must be above SL and below TP1
                    if entry_price <= stop_loss or entry_price >= tp1:
                        continue

                    patterns.append(DetailedPattern(
                        id=0,
                        type='flat',
                        module='C',
                        direction='long',
                        pivot_indices=[p0['pivot_idx'], p1['pivot_idx'], p2['pivot_idx']],
                        pivot_prices=[p0['price'], p1['price'], p2['price']],
                        pivot_types=[p0['type'], p1['type'], p2['type']],
                        entry_zone_low=min(entry_price, stop_loss),
                        entry_zone_high=max(entry_price, tp1),
                        tp1=tp1,
                        tp2=tp2,
                        stop_loss=stop_loss,
                        confirmed_bar=confirmed_bar,
                        entry_bar=entry_bar,
                        traded=False,
                        entry_price=entry_price,
                        retrace_pct=b_retrace * 100
                    ))

        return patterns

    def _find_triangle_setups_detailed(
        self,
        pivots: List[Dict],
        open_prices: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray
    ) -> List[DetailedPattern]:
        """Find Triangle (A-B-C-D-E) setups with full detail for visualization"""
        patterns = []
        n_pivots = len(pivots)

        for i in range(4, n_pivots):
            p0 = pivots[i-4]
            p1 = pivots[i-3]
            p2 = pivots[i-2]
            p3 = pivots[i-1]
            p4 = pivots[i]

            # Contracting triangle bullish: high-low-high-low-high
            if (p0['type'] == 1 and p1['type'] == -1 and p2['type'] == 1 and
                p3['type'] == -1 and p4['type'] == 1):
                a_high = p0['price']
                b_low = p1['price']
                c_high = p2['price']
                d_low = p3['price']
                e_high = p4['price']

                # Validate contracting pattern
                if c_high >= a_high or d_low <= b_low or e_high >= c_high:
                    continue

                triangle_width = a_high - b_low
                if triangle_width <= 0:
                    continue

                confirmed_bar = p4['confirmed_idx']
                entry_bar = confirmed_bar + 1
                if entry_bar >= len(open_prices):
                    continue

                entry_price = open_prices[entry_bar]
                stop_loss = d_low - (triangle_width * 0.05)
                tp1 = entry_price + triangle_width * 0.5
                tp2 = entry_price + triangle_width

                # Validate: for long, entry must be above SL and below TP1
                if entry_price <= stop_loss or entry_price >= tp1:
                    continue

                patterns.append(DetailedPattern(
                    id=0,
                    type='triangle',
                    module='C',
                    direction='long',
                    pivot_indices=[p0['pivot_idx'], p1['pivot_idx'], p2['pivot_idx'],
                                  p3['pivot_idx'], p4['pivot_idx']],
                    pivot_prices=[p0['price'], p1['price'], p2['price'],
                                 p3['price'], p4['price']],
                    pivot_types=[p0['type'], p1['type'], p2['type'],
                                p3['type'], p4['type']],
                    entry_zone_low=stop_loss,
                    entry_zone_high=tp1,
                    tp1=tp1,
                    tp2=tp2,
                    stop_loss=stop_loss,
                    confirmed_bar=confirmed_bar,
                    entry_bar=entry_bar,
                    traded=False,
                    entry_price=entry_price,
                    retrace_pct=0
                ))

            # Contracting triangle bearish: low-high-low-high-low
            elif (p0['type'] == -1 and p1['type'] == 1 and p2['type'] == -1 and
                  p3['type'] == 1 and p4['type'] == -1):
                a_low = p0['price']
                b_high = p1['price']
                c_low = p2['price']
                d_high = p3['price']
                e_low = p4['price']

                if c_low <= a_low or d_high >= b_high or e_low <= c_low:
                    continue

                triangle_width = b_high - a_low
                if triangle_width <= 0:
                    continue

                confirmed_bar = p4['confirmed_idx']
                entry_bar = confirmed_bar + 1
                if entry_bar >= len(open_prices):
                    continue

                entry_price = open_prices[entry_bar]
                stop_loss = d_high + (triangle_width * 0.05)
                tp1 = entry_price - triangle_width * 0.5
                tp2 = entry_price - triangle_width

                # Validate: for short, entry must be below SL and above TP1
                if entry_price >= stop_loss or entry_price <= tp1:
                    continue

                patterns.append(DetailedPattern(
                    id=0,
                    type='triangle',
                    module='C',
                    direction='short',
                    pivot_indices=[p0['pivot_idx'], p1['pivot_idx'], p2['pivot_idx'],
                                  p3['pivot_idx'], p4['pivot_idx']],
                    pivot_prices=[p0['price'], p1['price'], p2['price'],
                                 p3['price'], p4['price']],
                    pivot_types=[p0['type'], p1['type'], p2['type'],
                                p3['type'], p4['type']],
                    entry_zone_low=tp1,
                    entry_zone_high=stop_loss,
                    tp1=tp1,
                    tp2=tp2,
                    stop_loss=stop_loss,
                    confirmed_bar=confirmed_bar,
                    entry_bar=entry_bar,
                    traded=False,
                    entry_price=entry_price,
                    retrace_pct=0
                ))

        return patterns

    def _simulate_trades(
        self,
        patterns: List[DetailedPattern],
        open_prices: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        timestamps: np.ndarray
    ) -> Tuple[List[DetailedTrade], List[EquityPoint], List[AuditIssue]]:
        """
        Simulate trades and collect detailed information.
        """
        n = len(open_prices)
        trades = []
        equity_curve = []
        audit_issues = []

        # Trading state
        open_trades = []  # List of active DetailedTrade
        balance = self.config.initial_balance
        trade_id = 0
        pattern_idx = 0

        # Track partial exit state
        partial_exits = {}  # trade_id -> partial exit info

        for i in range(n):
            current_high = high[i]
            current_low = low[i]
            current_close = close[i]
            current_date = str(pd.Timestamp(timestamps[i]).strftime('%Y-%m-%d %H:%M'))

            # Check exits for open trades
            closed_trades = []
            for trade in open_trades:
                exit_price = None
                exit_reason = None

                if trade.direction == 'long':
                    # Check stop loss
                    if current_low <= trade.stop_loss:
                        exit_price = trade.stop_loss
                        exit_reason = 'Stop Loss'
                    # Check TP1 (partial exit)
                    elif trade.tp1_bar is None and current_high >= trade.tp1:
                        # Partial exit at TP1
                        partial_qty = trade.initial_quantity * (self.config.tp1_pct / 100)
                        partial_pnl = partial_qty * (trade.tp1 - trade.entry_price)
                        partial_fee = partial_qty * trade.tp1 * (self.config.taker_fee_pct / 100)

                        trade.tp1_bar = i
                        trade.tp1_price = trade.tp1
                        trade.tp1_qty = partial_qty
                        trade.fees_tp1 = partial_fee
                        trade.quantity -= partial_qty

                        # Move stop to breakeven
                        trade.stop_loss = trade.entry_price

                        balance += partial_pnl - partial_fee
                    # Check TP2
                    elif trade.tp1_bar is not None and current_high >= trade.tp2:
                        exit_price = trade.tp2
                        exit_reason = 'TP2'
                    # Check breakeven stop
                    elif trade.tp1_bar is not None and current_low <= trade.entry_price:
                        exit_price = trade.entry_price
                        exit_reason = 'Breakeven'

                else:  # short
                    if current_high >= trade.stop_loss:
                        exit_price = trade.stop_loss
                        exit_reason = 'Stop Loss'
                    elif trade.tp1_bar is None and current_low <= trade.tp1:
                        partial_qty = trade.initial_quantity * (self.config.tp1_pct / 100)
                        partial_pnl = partial_qty * (trade.entry_price - trade.tp1)
                        partial_fee = partial_qty * trade.tp1 * (self.config.taker_fee_pct / 100)

                        trade.tp1_bar = i
                        trade.tp1_price = trade.tp1
                        trade.tp1_qty = partial_qty
                        trade.fees_tp1 = partial_fee
                        trade.quantity -= partial_qty
                        trade.stop_loss = trade.entry_price

                        balance += partial_pnl - partial_fee
                    elif trade.tp1_bar is not None and current_low <= trade.tp2:
                        exit_price = trade.tp2
                        exit_reason = 'TP2'
                    elif trade.tp1_bar is not None and current_high >= trade.entry_price:
                        exit_price = trade.entry_price
                        exit_reason = 'Breakeven'

                if exit_price is not None:
                    trade.final_exit_bar = i
                    trade.final_exit_price = exit_price
                    trade.final_exit_reason = exit_reason

                    # Calculate final P&L
                    if trade.direction == 'long':
                        remaining_pnl = trade.quantity * (exit_price - trade.entry_price)
                    else:
                        remaining_pnl = trade.quantity * (trade.entry_price - exit_price)

                    exit_fee = trade.quantity * exit_price * (self.config.taker_fee_pct / 100)
                    trade.fees_exit = exit_fee

                    # Calculate gross and net
                    if trade.tp1_bar is not None:
                        tp1_pnl = trade.tp1_qty * (
                            trade.tp1_price - trade.entry_price if trade.direction == 'long'
                            else trade.entry_price - trade.tp1_price
                        )
                        trade.pnl_gross = tp1_pnl + remaining_pnl
                    else:
                        trade.pnl_gross = remaining_pnl

                    total_fees = trade.fees_entry + trade.fees_tp1 + trade.fees_exit
                    trade.pnl_net = trade.pnl_gross - total_fees

                    trade.balance_after = balance + remaining_pnl - exit_fee
                    balance = trade.balance_after

                    trades.append(trade)
                    closed_trades.append(trade)

                    # Audit: verify exit price matches OHLC
                    if exit_reason == 'Stop Loss':
                        if trade.direction == 'long' and exit_price > current_low:
                            audit_issues.append(AuditIssue(
                                type='price',
                                severity='warning',
                                trade_id=trade.id,
                                bar=i,
                                message=f"Stop loss price {exit_price:.2f} above bar low {current_low:.2f}"
                            ))
                        elif trade.direction == 'short' and exit_price < current_high:
                            audit_issues.append(AuditIssue(
                                type='price',
                                severity='warning',
                                trade_id=trade.id,
                                bar=i,
                                message=f"Stop loss price {exit_price:.2f} below bar high {current_high:.2f}"
                            ))

            for t in closed_trades:
                open_trades.remove(t)

            # Check for new entries
            while pattern_idx < len(patterns) and patterns[pattern_idx].entry_bar <= i:
                pattern = patterns[pattern_idx]
                pattern_idx += 1

                if pattern.entry_bar != i:
                    continue

                max_pos = getattr(self.config, 'max_positions', 10)
                if len(open_trades) >= max_pos:
                    continue

                # Position sizing
                sizing_balance = min(balance, self.config.initial_balance * 10)
                sizing_balance = max(sizing_balance, 0)

                risk_amount = sizing_balance * (self.config.base_risk_pct / 100)
                risk_per_unit = abs(pattern.entry_price - pattern.stop_loss)

                if risk_per_unit <= 0 or risk_amount <= 0:
                    continue

                quantity = risk_amount / risk_per_unit
                quantity *= (self.config.initial_position_pct / 100)

                max_qty = (sizing_balance * 0.5) / pattern.entry_price
                quantity = min(quantity, max_qty)

                if quantity <= 0:
                    continue

                entry_fee = quantity * pattern.entry_price * (self.config.maker_fee_pct / 100)

                # Create detailed trade
                trade = DetailedTrade(
                    id=trade_id,
                    module=pattern.module,
                    pattern_id=pattern.id,
                    direction=pattern.direction,
                    entry_bar=pattern.entry_bar,
                    entry_price=pattern.entry_price,
                    entry_date=current_date,
                    quantity=quantity,
                    initial_quantity=quantity,
                    stop_loss=pattern.stop_loss,
                    tp1=pattern.tp1,
                    tp2=pattern.tp2,
                    fees_entry=entry_fee,
                    balance_before=balance,
                    confirmed_bar=pattern.confirmed_bar,
                    pivot_bar=pattern.pivot_indices[-1],  # Last pivot
                    retrace_pct=pattern.retrace_pct,
                    wave_pivots=pattern.pivot_indices.copy(),
                    wave_prices=pattern.pivot_prices.copy()
                )

                # Mark pattern as traded
                pattern.traded = True
                pattern.trade_id = trade_id

                open_trades.append(trade)
                trade_id += 1

                # Audit: verify entry timing
                if pattern.entry_bar <= pattern.confirmed_bar:
                    audit_issues.append(AuditIssue(
                        type='timing',
                        severity='error',
                        trade_id=trade.id,
                        bar=i,
                        message=f"Entry bar {pattern.entry_bar} not after confirmed bar {pattern.confirmed_bar}"
                    ))

                # Audit: verify entry price
                expected_open = open_prices[pattern.entry_bar]
                if abs(pattern.entry_price - expected_open) > 0.01:
                    audit_issues.append(AuditIssue(
                        type='price',
                        severity='warning',
                        trade_id=trade.id,
                        bar=i,
                        message=f"Entry price {pattern.entry_price:.2f} doesn't match bar open {expected_open:.2f}"
                    ))

            # Record equity
            unrealized = 0
            for trade in open_trades:
                if trade.direction == 'long':
                    unrealized += trade.quantity * (current_close - trade.entry_price)
                else:
                    unrealized += trade.quantity * (trade.entry_price - current_close)

            total_equity = balance + unrealized

            # Calculate drawdown
            if i == 0:
                peak = total_equity
            else:
                peak = max(peak, total_equity)

            dd_pct = (peak - total_equity) / peak * 100 if peak > 0 else 0

            equity_curve.append(EquityPoint(
                bar=i,
                date=current_date,
                equity=total_equity,
                drawdown_pct=dd_pct
            ))

        # Close remaining trades at end
        for trade in open_trades:
            trade.final_exit_bar = n - 1
            trade.final_exit_price = close[-1]
            trade.final_exit_reason = 'End of backtest'

            if trade.direction == 'long':
                remaining_pnl = trade.quantity * (close[-1] - trade.entry_price)
            else:
                remaining_pnl = trade.quantity * (trade.entry_price - close[-1])

            exit_fee = trade.quantity * close[-1] * (self.config.taker_fee_pct / 100)
            trade.fees_exit = exit_fee

            if trade.tp1_bar is not None:
                tp1_pnl = trade.tp1_qty * (
                    trade.tp1_price - trade.entry_price if trade.direction == 'long'
                    else trade.entry_price - trade.tp1_price
                )
                trade.pnl_gross = tp1_pnl + remaining_pnl
            else:
                trade.pnl_gross = remaining_pnl

            total_fees = trade.fees_entry + trade.fees_tp1 + trade.fees_exit
            trade.pnl_net = trade.pnl_gross - total_fees
            trade.balance_after = balance + remaining_pnl - exit_fee

            trades.append(trade)

        return trades, equity_curve, audit_issues

    def _calculate_metrics(
        self,
        trades: List[DetailedTrade],
        equity_curve: List[EquityPoint]
    ) -> Dict:
        """Calculate summary metrics"""
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_return_pct': 0,
                'sharpe': 0,
                'max_drawdown_pct': 0,
                'profit_factor': 0
            }

        pnls = [t.pnl_net for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        final_equity = equity_curve[-1].equity if equity_curve else self.config.initial_balance
        total_return_pct = (final_equity - self.config.initial_balance) / self.config.initial_balance * 100

        max_dd = max([e.drawdown_pct for e in equity_curve]) if equity_curve else 0

        # Sharpe
        equity_values = [e.equity for e in equity_curve]
        returns = np.diff(equity_values) / np.array(equity_values[:-1])
        returns = returns[~np.isnan(returns) & ~np.isinf(returns)]
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 1 and np.std(returns) > 0 else 0

        # Profit factor
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 1
        pf = gross_profit / gross_loss if gross_loss > 0 else gross_profit

        return {
            'total_trades': len(trades),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': len(wins) / len(trades) * 100 if trades else 0,
            'total_return_pct': total_return_pct,
            'sharpe': sharpe,
            'max_drawdown_pct': max_dd,
            'profit_factor': pf,
            'avg_win': np.mean(wins) if wins else 0,
            'avg_loss': np.mean(losses) if losses else 0,
            'largest_win': max(wins) if wins else 0,
            'largest_loss': min(losses) if losses else 0
        }

    def _validate_data(self, df: pd.DataFrame, candles: List[Dict]) -> List[AuditIssue]:
        """Validate data integrity"""
        issues = []

        for c in candles:
            # High >= Low
            if c['high'] < c['low']:
                issues.append(AuditIssue(
                    type='data',
                    severity='error',
                    trade_id=None,
                    bar=c['bar'],
                    message=f"High {c['high']:.2f} < Low {c['low']:.2f}"
                ))

            # Open and Close within High-Low range
            if c['open'] > c['high'] or c['open'] < c['low']:
                issues.append(AuditIssue(
                    type='data',
                    severity='warning',
                    trade_id=None,
                    bar=c['bar'],
                    message=f"Open {c['open']:.2f} outside High-Low range"
                ))

            if c['close'] > c['high'] or c['close'] < c['low']:
                issues.append(AuditIssue(
                    type='data',
                    severity='warning',
                    trade_id=None,
                    bar=c['bar'],
                    message=f"Close {c['close']:.2f} outside High-Low range"
                ))

        return issues

    def _pattern_to_dict(self, p: DetailedPattern) -> Dict:
        """Convert pattern to dict for JSON"""
        return {
            'id': p.id,
            'type': p.type,
            'module': p.module,
            'direction': p.direction,
            'pivot_indices': p.pivot_indices,
            'pivot_prices': p.pivot_prices,
            'pivot_types': p.pivot_types,
            'entry_zone_low': p.entry_zone_low,
            'entry_zone_high': p.entry_zone_high,
            'tp1': p.tp1,
            'tp2': p.tp2,
            'stop_loss': p.stop_loss,
            'confirmed_bar': p.confirmed_bar,
            'entry_bar': p.entry_bar,
            'traded': p.traded,
            'trade_id': p.trade_id,
            'entry_price': p.entry_price,
            'retrace_pct': p.retrace_pct
        }

    def _trade_to_dict(self, t: DetailedTrade) -> Dict:
        """Convert trade to dict for JSON"""
        return {
            'id': t.id,
            'module': t.module,
            'pattern_id': t.pattern_id,
            'direction': t.direction,
            'entry_bar': t.entry_bar,
            'entry_price': t.entry_price,
            'entry_date': t.entry_date,
            'quantity': t.quantity,
            'initial_quantity': t.initial_quantity,
            'stop_loss': t.stop_loss,
            'tp1': t.tp1,
            'tp2': t.tp2,
            'tp1_bar': t.tp1_bar,
            'tp1_price': t.tp1_price,
            'tp1_qty': t.tp1_qty,
            'final_exit_bar': t.final_exit_bar,
            'final_exit_price': t.final_exit_price,
            'final_exit_reason': t.final_exit_reason,
            'fees_entry': t.fees_entry,
            'fees_tp1': t.fees_tp1,
            'fees_exit': t.fees_exit,
            'pnl_gross': t.pnl_gross,
            'pnl_net': t.pnl_net,
            'balance_before': t.balance_before,
            'balance_after': t.balance_after,
            'confirmed_bar': t.confirmed_bar,
            'pivot_bar': t.pivot_bar,
            'retrace_pct': t.retrace_pct,
            'wave_pivots': t.wave_pivots,
            'wave_prices': t.wave_prices
        }


def generate_verification_html(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    config: Config,
    output_path: str
) -> str:
    """
    Generate self-contained HTML verification tool.

    Args:
        df: OHLC DataFrame
        symbol: Trading symbol (e.g., 'ETHUSDT')
        timeframe: Timeframe (e.g., '1d')
        config: Backtest configuration
        output_path: Where to save the HTML file

    Returns:
        Path to generated HTML file
    """
    # Collect all data
    collector = VerificationDataCollector(config)
    data = collector.collect(df, symbol, timeframe)

    # Generate HTML
    html_content = _generate_html(data)

    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    return output_path


def _build_audit_html(audit_list: list) -> str:
    """Build HTML for audit issues list"""
    if not audit_list:
        return '<div class="no-selection">No issues found</div>'

    items = []
    for a in audit_list[:15]:
        trade_id = a['trade_id'] if a['trade_id'] is not None else 'null'
        msg = a['message'][:35] + '...' if len(a['message']) > 35 else a['message']
        item = f'<div class="audit-item" onclick="goToAuditIssue({a["bar"]}, {trade_id})"><span class="audit-badge {a["severity"]}">{a["type"]}</span><span class="toggle-label">Bar {a["bar"]}: {msg}</span></div>'
        items.append(item)

    return ''.join(items)


def _generate_html(data: Dict) -> str:
    """Generate the complete HTML file with embedded data and Plotly.js"""

    # Convert data to JSON
    data_json = json.dumps(data, indent=None, default=str)

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{data['symbol']} {data['timeframe']} - Backtest Verification</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #0d1117;
            color: #c9d1d9;
            overflow-x: hidden;
        }}

        .header {{
            background: linear-gradient(135deg, #161b22 0%, #21262d 100%);
            padding: 15px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #30363d;
        }}

        .header h1 {{
            font-size: 1.4em;
            color: #58a6ff;
        }}

        .metrics-bar {{
            display: flex;
            gap: 25px;
        }}

        .metric {{
            text-align: center;
        }}

        .metric-value {{
            font-size: 1.3em;
            font-weight: bold;
        }}

        .metric-value.positive {{ color: #3fb950; }}
        .metric-value.negative {{ color: #f85149; }}
        .metric-value.neutral {{ color: #8b949e; }}

        .metric-label {{
            font-size: 0.75em;
            color: #8b949e;
            text-transform: uppercase;
        }}

        .main-container {{
            display: flex;
            height: calc(100vh - 60px);
        }}

        .chart-area {{
            flex: 1;
            display: flex;
            flex-direction: column;
            padding: 10px;
        }}

        .sidebar {{
            width: 320px;
            background: #161b22;
            border-left: 1px solid #30363d;
            overflow-y: auto;
            padding: 15px;
        }}

        .chart-container {{
            flex: 1;
            min-height: 400px;
        }}

        .equity-container {{
            height: 150px;
            margin-top: 10px;
        }}

        .timeline-container {{
            height: 50px;
            background: #161b22;
            border-radius: 8px;
            margin-top: 10px;
            padding: 10px 20px;
            display: flex;
            align-items: center;
            gap: 15px;
        }}

        .timeline-slider {{
            flex: 1;
            -webkit-appearance: none;
            height: 6px;
            border-radius: 3px;
            background: #30363d;
            outline: none;
        }}

        .timeline-slider::-webkit-slider-thumb {{
            -webkit-appearance: none;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: #58a6ff;
            cursor: pointer;
        }}

        .timeline-date {{
            font-size: 0.9em;
            min-width: 100px;
            text-align: center;
        }}

        .trade-table-container {{
            height: 200px;
            margin-top: 10px;
            overflow: auto;
            border: 1px solid #30363d;
            border-radius: 8px;
        }}

        .trade-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85em;
        }}

        .trade-table th {{
            background: #21262d;
            padding: 8px;
            text-align: left;
            position: sticky;
            top: 0;
            border-bottom: 1px solid #30363d;
            cursor: pointer;
        }}

        .trade-table th:hover {{
            background: #30363d;
        }}

        .trade-table td {{
            padding: 6px 8px;
            border-bottom: 1px solid #21262d;
        }}

        .trade-table tr:hover {{
            background: #21262d;
            cursor: pointer;
        }}

        .trade-table tr.selected {{
            background: #1f3a5f;
        }}

        .pnl-positive {{ color: #3fb950; }}
        .pnl-negative {{ color: #f85149; }}

        .sidebar-section {{
            margin-bottom: 20px;
        }}

        .sidebar-section h3 {{
            font-size: 0.9em;
            color: #8b949e;
            margin-bottom: 10px;
            text-transform: uppercase;
            border-bottom: 1px solid #30363d;
            padding-bottom: 5px;
        }}

        .toggle-group {{
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}

        .toggle-item {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .toggle-switch {{
            position: relative;
            width: 40px;
            height: 20px;
            background: #30363d;
            border-radius: 10px;
            cursor: pointer;
            transition: background 0.3s;
        }}

        .toggle-switch.active {{
            background: #238636;
        }}

        .toggle-switch::after {{
            content: '';
            position: absolute;
            top: 2px;
            left: 2px;
            width: 16px;
            height: 16px;
            background: #c9d1d9;
            border-radius: 50%;
            transition: left 0.3s;
        }}

        .toggle-switch.active::after {{
            left: 22px;
        }}

        .toggle-label {{
            font-size: 0.9em;
        }}

        .trade-detail {{
            background: #21262d;
            border-radius: 8px;
            padding: 12px;
            font-size: 0.85em;
        }}

        .trade-detail-row {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 6px;
        }}

        .trade-detail-label {{
            color: #8b949e;
        }}

        .trade-detail-section {{
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid #30363d;
        }}

        .trade-detail h4 {{
            font-size: 0.85em;
            color: #58a6ff;
            margin-bottom: 8px;
        }}

        .audit-badge {{
            display: inline-block;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.75em;
            margin-left: 5px;
        }}

        .audit-badge.warning {{
            background: #9e6a03;
            color: #fff;
        }}

        .audit-badge.error {{
            background: #f85149;
            color: #fff;
        }}

        .no-selection {{
            text-align: center;
            color: #8b949e;
            padding: 30px;
        }}

        .keyboard-hint {{
            font-size: 0.8em;
            color: #6e7681;
            margin-top: 10px;
            text-align: center;
        }}

        .module-a {{ color: #58a6ff; }}
        .module-b {{ color: #d29922; }}

        .audit-item {{
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 6px 8px;
            margin: 4px 0;
            border-radius: 4px;
            cursor: pointer;
            transition: background 0.2s;
        }}

        .audit-item:hover {{
            background: #30363d;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{data['symbol']} {data['timeframe']} Backtest Verification</h1>
        <div class="metrics-bar">
            <div class="metric">
                <div class="metric-value neutral">{data['metrics']['total_trades']}</div>
                <div class="metric-label">Trades</div>
            </div>
            <div class="metric">
                <div class="metric-value {'positive' if data['metrics']['win_rate'] >= 50 else 'negative'}">{data['metrics']['win_rate']:.1f}%</div>
                <div class="metric-label">Win Rate</div>
            </div>
            <div class="metric">
                <div class="metric-value {'positive' if data['metrics']['total_return_pct'] >= 0 else 'negative'}">{data['metrics']['total_return_pct']:.2f}%</div>
                <div class="metric-label">Return</div>
            </div>
            <div class="metric">
                <div class="metric-value {'positive' if data['metrics']['sharpe'] >= 0 else 'negative'}">{data['metrics']['sharpe']:.2f}</div>
                <div class="metric-label">Sharpe</div>
            </div>
            <div class="metric">
                <div class="metric-value negative">{data['metrics']['max_drawdown_pct']:.2f}%</div>
                <div class="metric-label">Max DD</div>
            </div>
            <div class="metric">
                <div class="metric-value {'positive' if data['metrics']['profit_factor'] >= 1 else 'negative'}">{data['metrics']['profit_factor']:.2f}</div>
                <div class="metric-label">Profit Factor</div>
            </div>
        </div>
    </div>

    <div class="main-container">
        <div class="chart-area">
            <div id="main-chart" class="chart-container"></div>
            <div id="equity-chart" class="equity-container"></div>
            <div class="timeline-container">
                <span class="timeline-date" id="timeline-date">-</span>
                <input type="range" class="timeline-slider" id="timeline-slider" min="0" max="{len(data['candles'])-1}" value="{len(data['candles'])-1}">
                <span class="timeline-date" id="timeline-bar">Bar: -</span>
            </div>
            <div class="trade-table-container">
                <table class="trade-table" id="trade-table">
                    <thead>
                        <tr>
                            <th data-sort="id">#</th>
                            <th data-sort="entry_date">Date</th>
                            <th data-sort="direction">Dir</th>
                            <th data-sort="module">Mod</th>
                            <th data-sort="entry_price">Entry</th>
                            <th data-sort="final_exit_price">Exit</th>
                            <th data-sort="pnl_net">P&L</th>
                            <th data-sort="final_exit_reason">Reason</th>
                        </tr>
                    </thead>
                    <tbody id="trade-table-body"></tbody>
                </table>
            </div>
        </div>

        <div class="sidebar">
            <div class="sidebar-section">
                <h3>Display Options</h3>
                <div class="toggle-group">
                    <div class="toggle-item">
                        <div class="toggle-switch active" id="toggle-waves" onclick="toggleLayer('waves')"></div>
                        <span class="toggle-label">Wave Patterns</span>
                    </div>
                    <div class="toggle-item">
                        <div class="toggle-switch" id="toggle-nontraded" onclick="toggleLayer('nontraded')"></div>
                        <span class="toggle-label">Non-traded Patterns</span>
                    </div>
                    <div class="toggle-item">
                        <div class="toggle-switch active" id="toggle-entries" onclick="toggleLayer('entries')"></div>
                        <span class="toggle-label">Entry Zones</span>
                    </div>
                    <div class="toggle-item">
                        <div class="toggle-switch active" id="toggle-tp" onclick="toggleLayer('tp')"></div>
                        <span class="toggle-label">TP Levels</span>
                    </div>
                    <div class="toggle-item">
                        <div class="toggle-switch active" id="toggle-sl" onclick="toggleLayer('sl')"></div>
                        <span class="toggle-label">SL Levels</span>
                    </div>
                    <div class="toggle-item">
                        <div class="toggle-switch active" id="toggle-paths" onclick="toggleLayer('paths')"></div>
                        <span class="toggle-label">Trade Paths</span>
                    </div>
                    <div class="toggle-item">
                        <div class="toggle-switch active" id="toggle-pivots" onclick="toggleLayer('pivots')"></div>
                        <span class="toggle-label">Pivot Arrows</span>
                    </div>
                </div>
            </div>

            <div class="sidebar-section">
                <h3>Selected Trade</h3>
                <div id="trade-detail">
                    <div class="no-selection">Click a trade to view details</div>
                </div>
            </div>

            <div class="sidebar-section">
                <h3>Audit Issues ({len(data['audit'])})</h3>
                <div id="audit-list">
                    {_build_audit_html(data['audit'])}
                </div>
            </div>

            <div class="keyboard-hint">
                Use ← → arrows to navigate trades
            </div>
        </div>
    </div>

    <script>
        // Embedded data
        const DATA = {data_json};

        // State
        let selectedTradeId = null;
        let currentBar = DATA.candles.length - 1;
        let layers = {{
            waves: true,
            nontraded: false,
            entries: true,
            tp: true,
            sl: true,
            paths: true,
            pivots: true
        }};

        // Initialize
        document.addEventListener('DOMContentLoaded', function() {{
            renderTradeTable();
            renderCharts();
            setupEventListeners();
        }});

        function renderTradeTable() {{
            const tbody = document.getElementById('trade-table-body');
            tbody.innerHTML = '';

            DATA.trades.forEach(trade => {{
                const row = document.createElement('tr');
                row.dataset.tradeId = trade.id;
                row.innerHTML = `
                    <td>${{trade.id + 1}}</td>
                    <td>${{trade.entry_date}}</td>
                    <td>${{trade.direction === 'long' ? '▲' : '▼'}}</td>
                    <td class="${{trade.module === 'A' ? 'module-a' : 'module-b'}}">${{trade.module}}</td>
                    <td>${{trade.entry_price.toFixed(2)}}</td>
                    <td>${{trade.final_exit_price ? trade.final_exit_price.toFixed(2) : '-'}}</td>
                    <td class="${{trade.pnl_net >= 0 ? 'pnl-positive' : 'pnl-negative'}}">${{trade.pnl_net >= 0 ? '+' : ''}}${{trade.pnl_net.toFixed(2)}}</td>
                    <td>${{trade.final_exit_reason || '-'}}</td>
                `;
                row.onclick = () => selectTrade(trade.id);
                tbody.appendChild(row);
            }});
        }}

        function renderCharts() {{
            renderMainChart();
            renderEquityChart();
        }}

        function renderMainChart() {{
            const candles = DATA.candles;
            const dates = candles.map(c => c.date);

            // Candlestick trace
            const candlestick = {{
                x: dates,
                open: candles.map(c => c.open),
                high: candles.map(c => c.high),
                low: candles.map(c => c.low),
                close: candles.map(c => c.close),
                type: 'candlestick',
                name: 'Price',
                increasing: {{line: {{color: '#3fb950'}}, fillcolor: '#238636'}},
                decreasing: {{line: {{color: '#f85149'}}, fillcolor: '#da3633'}},
            }};

            const traces = [candlestick];
            const shapes = [];
            const annotations = [];

            // Add trade markers and paths
            DATA.trades.forEach(trade => {{
                const entryDate = candles[trade.entry_bar].date;
                const exitDate = trade.final_exit_bar !== null ? candles[trade.final_exit_bar].date : dates[dates.length - 1];

                // Entry marker
                traces.push({{
                    x: [entryDate],
                    y: [trade.entry_price],
                    mode: 'markers+text',
                    type: 'scatter',
                    marker: {{
                        symbol: trade.direction === 'long' ? 'triangle-up' : 'triangle-down',
                        size: 14,
                        color: trade.module === 'A' ? '#58a6ff' : '#d29922',
                        line: {{color: '#fff', width: 1}}
                    }},
                    text: [trade.module],
                    textposition: trade.direction === 'long' ? 'top center' : 'bottom center',
                    textfont: {{color: '#fff', size: 10}},
                    name: `Trade ${{trade.id + 1}} Entry`,
                    showlegend: false,
                    hoverinfo: 'text',
                    hovertext: `Entry: ${{trade.entry_price.toFixed(2)}}<br>Module ${{trade.module}} ${{trade.direction}}`
                }});

                // Exit marker
                if (trade.final_exit_bar !== null) {{
                    const exitColor = trade.pnl_net >= 0 ? '#3fb950' : '#f85149';
                    traces.push({{
                        x: [exitDate],
                        y: [trade.final_exit_price],
                        mode: 'markers',
                        type: 'scatter',
                        marker: {{
                            symbol: trade.final_exit_reason === 'Stop Loss' ? 'x' : 'circle',
                            size: 10,
                            color: exitColor,
                            line: {{color: '#fff', width: 1}}
                        }},
                        name: `Trade ${{trade.id + 1}} Exit`,
                        showlegend: false,
                        hoverinfo: 'text',
                        hovertext: `Exit: ${{trade.final_exit_price.toFixed(2)}}<br>${{trade.final_exit_reason}}<br>P&L: ${{trade.pnl_net.toFixed(2)}}`
                    }});

                    // Trade path
                    if (layers.paths) {{
                        const pathX = [entryDate];
                        const pathY = [trade.entry_price];

                        if (trade.tp1_bar !== null) {{
                            pathX.push(candles[trade.tp1_bar].date);
                            pathY.push(trade.tp1_price);
                        }}

                        pathX.push(exitDate);
                        pathY.push(trade.final_exit_price);

                        traces.push({{
                            x: pathX,
                            y: pathY,
                            mode: 'lines',
                            type: 'scatter',
                            line: {{color: exitColor, width: 1, dash: 'dot'}},
                            name: `Trade ${{trade.id + 1}} Path`,
                            showlegend: false,
                            hoverinfo: 'skip'
                        }});
                    }}
                }}

                // SL/TP levels
                if (layers.sl) {{
                    shapes.push({{
                        type: 'line',
                        x0: entryDate,
                        x1: exitDate,
                        y0: trade.stop_loss,
                        y1: trade.stop_loss,
                        line: {{color: '#f85149', width: 1, dash: 'dash'}},
                        opacity: 0.5
                    }});
                }}

                if (layers.tp) {{
                    shapes.push({{
                        type: 'line',
                        x0: entryDate,
                        x1: exitDate,
                        y0: trade.tp1,
                        y1: trade.tp1,
                        line: {{color: '#3fb950', width: 1, dash: 'dash'}},
                        opacity: 0.5
                    }});
                    shapes.push({{
                        type: 'line',
                        x0: entryDate,
                        x1: exitDate,
                        y0: trade.tp2,
                        y1: trade.tp2,
                        line: {{color: '#238636', width: 1, dash: 'dash'}},
                        opacity: 0.3
                    }});
                }}
            }});

            // Add pivot arrows
            if (layers.pivots) {{
                DATA.pivots.forEach(pivot => {{
                    if (pivot.confirmed_bar < candles.length && pivot.pivot_bar < candles.length) {{
                        const pivotDate = candles[pivot.pivot_bar].date;
                        const confirmDate = candles[pivot.confirmed_bar].date;

                        annotations.push({{
                            x: pivotDate,
                            y: pivot.price,
                            ax: confirmDate,
                            ay: pivot.price,
                            xref: 'x',
                            yref: 'y',
                            axref: 'x',
                            ayref: 'y',
                            showarrow: true,
                            arrowhead: 2,
                            arrowsize: 1,
                            arrowwidth: 1,
                            arrowcolor: '#d29922',
                            opacity: 0.4
                        }});
                    }}
                }});
            }}

            // Add wave patterns
            if (layers.waves) {{
                DATA.patterns.forEach(pattern => {{
                    if (!pattern.traded && !layers.nontraded) return;

                    const opacity = pattern.traded ? 0.8 : 0.3;
                    const color = pattern.module === 'A' ? '#58a6ff' : '#d29922';

                    const waveX = pattern.pivot_indices.map(idx => candles[idx].date);
                    const waveY = pattern.pivot_prices;

                    traces.push({{
                        x: waveX,
                        y: waveY,
                        mode: 'lines+markers+text',
                        type: 'scatter',
                        line: {{color: color, width: pattern.traded ? 2 : 1}},
                        marker: {{size: 6, color: color}},
                        text: pattern.pivot_indices.map((_, i) => (i + 1).toString()),
                        textposition: pattern.pivot_types[0] === -1 ?
                            (waveY.map((_, i) => i % 2 === 0 ? 'bottom center' : 'top center')) :
                            (waveY.map((_, i) => i % 2 === 0 ? 'top center' : 'bottom center')),
                        textfont: {{color: color, size: 10}},
                        opacity: opacity,
                        name: `Pattern ${{pattern.id}}`,
                        showlegend: false,
                        hoverinfo: 'text',
                        hovertext: `${{pattern.type}} (${{pattern.module}})<br>Retrace: ${{pattern.retrace_pct.toFixed(1)}}%<br>${{pattern.traded ? 'TRADED' : 'Not traded'}}`
                    }});
                }});
            }}

            const layout = {{
                paper_bgcolor: '#0d1117',
                plot_bgcolor: '#0d1117',
                font: {{color: '#c9d1d9'}},
                xaxis: {{
                    type: 'category',
                    rangeslider: {{visible: false}},
                    gridcolor: '#21262d',
                    linecolor: '#30363d',
                    tickangle: -45,
                    nticks: 20
                }},
                yaxis: {{
                    gridcolor: '#21262d',
                    linecolor: '#30363d',
                    side: 'right'
                }},
                margin: {{l: 10, r: 60, t: 10, b: 60}},
                showlegend: false,
                shapes: shapes,
                annotations: annotations,
                dragmode: 'pan'
            }};

            const config = {{
                scrollZoom: true,
                displayModeBar: true,
                modeBarButtonsToRemove: ['lasso2d', 'select2d']
            }};

            Plotly.newPlot('main-chart', traces, layout, config);

            // Set initial zoom to last 180 bars
            const startIdx = Math.max(0, candles.length - 180);
            Plotly.relayout('main-chart', {{
                'xaxis.range': [startIdx, candles.length - 1]
            }});

            // Add autoscale on zoom/pan
            document.getElementById('main-chart').on('plotly_relayout', function(eventData) {{
                if (eventData['xaxis.range[0]'] || eventData['xaxis.range']) {{
                    autoscaleYAxis();
                }}
            }});

            // Initial autoscale
            setTimeout(autoscaleYAxis, 100);
        }}

        function autoscaleYAxis() {{
            const chartDiv = document.getElementById('main-chart');
            const xRange = chartDiv.layout.xaxis.range;

            if (!xRange) return;

            const candles = DATA.candles;

            // xRange contains numeric bar indices for categorical axis
            let startIdx = Math.max(0, Math.floor(xRange[0]));
            let endIdx = Math.min(candles.length - 1, Math.ceil(xRange[1]));

            // Ensure valid range
            if (startIdx > endIdx || isNaN(startIdx) || isNaN(endIdx)) {{
                startIdx = 0;
                endIdx = candles.length - 1;
            }}

            // Calculate min/max from visible candles ONLY
            let minPrice = Infinity;
            let maxPrice = -Infinity;

            for (let i = startIdx; i <= endIdx; i++) {{
                if (candles[i]) {{
                    minPrice = Math.min(minPrice, candles[i].low);
                    maxPrice = Math.max(maxPrice, candles[i].high);
                }}
            }}

            if (minPrice === Infinity || maxPrice === -Infinity) return;

            // Add 2% padding for tighter fit
            const padding = (maxPrice - minPrice) * 0.02;
            minPrice -= padding;
            maxPrice += padding;

            // Prevent infinite loop by checking if range actually changed
            const currentRange = chartDiv.layout.yaxis.range;
            if (currentRange &&
                Math.abs(currentRange[0] - minPrice) < 1 &&
                Math.abs(currentRange[1] - maxPrice) < 1) {{
                return;
            }}

            Plotly.relayout('main-chart', {{
                'yaxis.range': [minPrice, maxPrice],
                'yaxis.autorange': false
            }});
        }}

        function renderEquityChart() {{
            const equity = DATA.equity;
            const dates = equity.map(e => DATA.candles[e.bar].date);

            // Calculate peak for drawdown shading
            let peak = equity[0].equity;
            const peaks = [];
            equity.forEach(e => {{
                peak = Math.max(peak, e.equity);
                peaks.push(peak);
            }});

            const traces = [
                {{
                    x: dates,
                    y: peaks,
                    type: 'scatter',
                    mode: 'lines',
                    fill: 'tozeroy',
                    fillcolor: 'rgba(35, 134, 54, 0.1)',
                    line: {{color: 'transparent'}},
                    name: 'Peak',
                    showlegend: false,
                    hoverinfo: 'skip'
                }},
                {{
                    x: dates,
                    y: equity.map(e => e.equity),
                    type: 'scatter',
                    mode: 'lines',
                    fill: 'tonexty',
                    fillcolor: 'rgba(248, 81, 73, 0.3)',
                    line: {{color: '#58a6ff', width: 2}},
                    name: 'Equity',
                    hovertemplate: '%{{y:,.0f}}<extra></extra>'
                }}
            ];

            const layout = {{
                paper_bgcolor: '#0d1117',
                plot_bgcolor: '#0d1117',
                font: {{color: '#c9d1d9'}},
                xaxis: {{
                    type: 'category',
                    showgrid: false,
                    showticklabels: false,
                    linecolor: '#30363d'
                }},
                yaxis: {{
                    gridcolor: '#21262d',
                    linecolor: '#30363d',
                    side: 'right'
                }},
                margin: {{l: 10, r: 60, t: 5, b: 5}},
                showlegend: false
            }};

            Plotly.newPlot('equity-chart', traces, layout, {{displayModeBar: false}});
        }}

        function selectTrade(tradeId) {{
            selectedTradeId = tradeId;

            // Update table selection
            document.querySelectorAll('#trade-table-body tr').forEach(row => {{
                row.classList.toggle('selected', parseInt(row.dataset.tradeId) === tradeId);
            }});

            // Update detail panel
            const trade = DATA.trades.find(t => t.id === tradeId);
            if (!trade) return;

            const pattern = DATA.patterns.find(p => p.id === trade.pattern_id);
            const duration = trade.final_exit_bar !== null ?
                trade.final_exit_bar - trade.entry_bar :
                DATA.candles.length - trade.entry_bar;

            const detailHtml = `
                <div class="trade-detail">
                    <h4>Trade #${{trade.id + 1}} - Module ${{trade.module}} (${{trade.type || pattern?.type || 'Wave'}})</h4>

                    <div class="trade-detail-row">
                        <span class="trade-detail-label">Direction</span>
                        <span>${{trade.direction.toUpperCase()}}</span>
                    </div>
                    <div class="trade-detail-row">
                        <span class="trade-detail-label">Entry</span>
                        <span>${{trade.entry_date}} @ $${{trade.entry_price.toFixed(2)}}</span>
                    </div>
                    <div class="trade-detail-row">
                        <span class="trade-detail-label">Exit</span>
                        <span>${{trade.final_exit_bar !== null ? DATA.candles[trade.final_exit_bar].date : '-'}} @ $${{trade.final_exit_price?.toFixed(2) || '-'}}</span>
                    </div>
                    <div class="trade-detail-row">
                        <span class="trade-detail-label">Duration</span>
                        <span>${{duration}} bars</span>
                    </div>

                    <div class="trade-detail-section">
                        <h4>Position Sizing</h4>
                        <div class="trade-detail-row">
                            <span class="trade-detail-label">Balance Before</span>
                            <span>$${{trade.balance_before.toFixed(2)}}</span>
                        </div>
                        <div class="trade-detail-row">
                            <span class="trade-detail-label">Quantity</span>
                            <span>${{trade.initial_quantity.toFixed(4)}}</span>
                        </div>
                        <div class="trade-detail-row">
                            <span class="trade-detail-label">Stop Loss</span>
                            <span>$${{trade.stop_loss.toFixed(2)}}</span>
                        </div>
                        <div class="trade-detail-row">
                            <span class="trade-detail-label">TP1 / TP2</span>
                            <span>$${{trade.tp1.toFixed(2)}} / $${{trade.tp2.toFixed(2)}}</span>
                        </div>
                    </div>

                    <div class="trade-detail-section">
                        <h4>P&L Breakdown</h4>
                        ${{trade.tp1_bar !== null ? `
                        <div class="trade-detail-row">
                            <span class="trade-detail-label">TP1 (40%)</span>
                            <span>$${{(trade.tp1_qty * (trade.direction === 'long' ? trade.tp1_price - trade.entry_price : trade.entry_price - trade.tp1_price)).toFixed(2)}}</span>
                        </div>` : ''}}
                        <div class="trade-detail-row">
                            <span class="trade-detail-label">Gross P&L</span>
                            <span class="${{trade.pnl_gross >= 0 ? 'pnl-positive' : 'pnl-negative'}}">$${{trade.pnl_gross.toFixed(2)}}</span>
                        </div>
                        <div class="trade-detail-row">
                            <span class="trade-detail-label">Entry Fee</span>
                            <span>-$${{trade.fees_entry.toFixed(2)}}</span>
                        </div>
                        ${{trade.fees_tp1 > 0 ? `
                        <div class="trade-detail-row">
                            <span class="trade-detail-label">TP1 Fee</span>
                            <span>-$${{trade.fees_tp1.toFixed(2)}}</span>
                        </div>` : ''}}
                        <div class="trade-detail-row">
                            <span class="trade-detail-label">Exit Fee</span>
                            <span>-$${{trade.fees_exit.toFixed(2)}}</span>
                        </div>
                        <div class="trade-detail-row" style="font-weight: bold;">
                            <span class="trade-detail-label">Net P&L</span>
                            <span class="${{trade.pnl_net >= 0 ? 'pnl-positive' : 'pnl-negative'}}">$${{trade.pnl_net.toFixed(2)}} (${{(trade.pnl_net / trade.balance_before * 100).toFixed(2)}}%)</span>
                        </div>
                    </div>

                    <div class="trade-detail-section">
                        <h4>Timing Verification</h4>
                        <div class="trade-detail-row">
                            <span class="trade-detail-label">Pivot Bar</span>
                            <span>${{trade.pivot_bar}}</span>
                        </div>
                        <div class="trade-detail-row">
                            <span class="trade-detail-label">Confirmed Bar</span>
                            <span>${{trade.confirmed_bar}}</span>
                        </div>
                        <div class="trade-detail-row">
                            <span class="trade-detail-label">Entry Bar</span>
                            <span>${{trade.entry_bar}} ${{trade.entry_bar > trade.confirmed_bar ? '✓' : '⚠️'}}</span>
                        </div>
                        <div class="trade-detail-row">
                            <span class="trade-detail-label">Retrace</span>
                            <span>${{trade.retrace_pct.toFixed(1)}}%</span>
                        </div>
                    </div>
                </div>
            `;

            document.getElementById('trade-detail').innerHTML = detailHtml;

            // Zoom chart to trade (use bar indices for categorical x-axis)
            const startBar = Math.max(0, trade.entry_bar - 30);
            const endBar = Math.min(DATA.candles.length - 1, (trade.final_exit_bar || trade.entry_bar) + 30);

            Plotly.relayout('main-chart', {{
                'xaxis.range': [startBar, endBar]
            }}).then(() => {{
                setTimeout(autoscaleYAxis, 50);
            }});
        }}

        function toggleLayer(layer) {{
            layers[layer] = !layers[layer];
            document.getElementById('toggle-' + layer).classList.toggle('active');
            renderCharts();
            setTimeout(autoscaleYAxis, 100);
        }}

        function goToAuditIssue(bar, tradeId) {{
            // Zoom to the bar with context (use bar indices for categorical x-axis)
            const startBar = Math.max(0, bar - 20);
            const endBar = Math.min(DATA.candles.length - 1, bar + 20);

            Plotly.relayout('main-chart', {{
                'xaxis.range': [startBar, endBar]
            }}).then(() => {{
                setTimeout(autoscaleYAxis, 50);
            }});

            // If there's a related trade, select it
            if (tradeId !== null) {{
                selectTrade(tradeId);
            }}

            // Add a temporary marker at the issue bar (use bar index for categorical x-axis)
            const candle = DATA.candles[bar];

            // Flash highlight using a shape
            Plotly.relayout('main-chart', {{
                shapes: [{{
                    type: 'rect',
                    xref: 'x',
                    yref: 'paper',
                    x0: bar - 0.5,
                    x1: bar + 0.5,
                    y0: 0,
                    y1: 1,
                    line: {{color: '#f85149', width: 3}},
                    fillcolor: 'rgba(248, 81, 73, 0.2)'
                }}]
            }});

            // Remove highlight after 2 seconds
            setTimeout(() => {{
                Plotly.relayout('main-chart', {{shapes: []}});
            }}, 2000);
        }}

        function setupEventListeners() {{
            // Timeline slider
            const slider = document.getElementById('timeline-slider');
            slider.addEventListener('input', function() {{
                currentBar = parseInt(this.value);
                document.getElementById('timeline-date').textContent = DATA.candles[currentBar].date;
                document.getElementById('timeline-bar').textContent = 'Bar: ' + currentBar;
            }});

            // Keyboard navigation
            document.addEventListener('keydown', function(e) {{
                if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {{
                    const trades = DATA.trades;
                    if (trades.length === 0) return;

                    let newIdx;
                    if (selectedTradeId === null) {{
                        newIdx = e.key === 'ArrowRight' ? 0 : trades.length - 1;
                    }} else {{
                        const currentIdx = trades.findIndex(t => t.id === selectedTradeId);
                        newIdx = e.key === 'ArrowRight' ?
                            Math.min(currentIdx + 1, trades.length - 1) :
                            Math.max(currentIdx - 1, 0);
                    }}

                    selectTrade(trades[newIdx].id);
                }}
            }});

            // Table sorting
            document.querySelectorAll('#trade-table th').forEach(th => {{
                th.addEventListener('click', function() {{
                    const sortKey = this.dataset.sort;
                    // Simple sort implementation would go here
                }});
            }});

            // Initialize timeline
            document.getElementById('timeline-date').textContent = DATA.candles[currentBar].date;
            document.getElementById('timeline-bar').textContent = 'Bar: ' + currentBar;
        }}
    </script>
</body>
</html>
'''

    return html
