"""
Module C: Corrective Wave Entry Strategy
"""

import pandas as pd
from typing import List, Optional
from dataclasses import dataclass

from .signals import Signal, SignalType
from ..waves.analyzer import WaveStructure
from ..waves.corrective import ZigzagPattern, FlatPattern, TrianglePattern, CorrectionType
from ..waves.impulse import WaveDirection
from ..indicators.fibonacci import FibonacciCalculator
from ..indicators.atr import calculate_atr


class ModuleCStrategy:
    """
    Corrective Wave Entry Strategy

    Enters on corrective patterns (Zigzag, Flat, Triangle)
    """

    def __init__(
        self,
        fib_tolerance_pct: float = 2.5,
        buffer_atr_mult: float = 0.5,
        trade_zigzags: bool = True,
        trade_flats: bool = True,
        trade_triangles: bool = True
    ):
        """
        Initialize Module C

        Args:
            fib_tolerance_pct: Tolerance around Fib levels
            buffer_atr_mult: ATR multiplier for stop buffer
            trade_zigzags: Whether to trade zigzag patterns
            trade_flats: Whether to trade flat patterns
            trade_triangles: Whether to trade triangle patterns
        """
        self.fib_calc = FibonacciCalculator(tolerance_pct=fib_tolerance_pct)
        self.buffer_atr_mult = buffer_atr_mult
        self.trade_zigzags = trade_zigzags
        self.trade_flats = trade_flats
        self.trade_triangles = trade_triangles

    def generate_signals(
        self,
        df: pd.DataFrame,
        structure: WaveStructure,
        bar_index: int
    ) -> List[Signal]:
        """
        Generate corrective wave entry signals

        Args:
            df: Price data
            structure: Current wave structure
            bar_index: Current bar index

        Returns:
            List of signals
        """
        signals = []
        current_bar = df.iloc[bar_index]
        current_price = current_bar['close']

        atr = calculate_atr(df.iloc[:bar_index + 1])
        current_atr = atr.iloc[-1] if len(atr) > 0 else current_price * 0.01

        # Zigzag signals
        if self.trade_zigzags:
            for zigzag in structure.partial_zigzags:
                if zigzag.is_valid:
                    signal = self._evaluate_zigzag(
                        zigzag, current_bar, bar_index, current_price, current_atr
                    )
                    if signal:
                        signals.append(signal)

        # Flat signals
        if self.trade_flats:
            for flat in structure.partial_flats:
                if flat.is_valid:
                    signal = self._evaluate_flat(
                        flat, current_bar, bar_index, current_price, current_atr
                    )
                    if signal:
                        signals.append(signal)

        # Triangle signals
        if self.trade_triangles:
            for triangle in structure.partial_triangles:
                if triangle.is_valid:
                    signal = self._evaluate_triangle(
                        triangle, current_bar, bar_index, current_price, current_atr
                    )
                    if signal:
                        signals.append(signal)

        return signals

    def _evaluate_zigzag(
        self,
        pattern: ZigzagPattern,
        current_bar: pd.Series,
        bar_index: int,
        current_price: float,
        current_atr: float
    ) -> Optional[Signal]:
        """
        Evaluate zigzag pattern for Wave C entry
        """
        # Direction: opposite of Wave A
        if pattern.wave_a_end.price < pattern.wave_a_start.price:
            direction = WaveDirection.DOWN  # A was down, C will continue down
            signal_type = SignalType.ZIGZAG_SHORT
        else:
            direction = WaveDirection.UP
            signal_type = SignalType.ZIGZAG_LONG

        # Check if we're near Wave B end (entry point)
        entry_tolerance = current_price * 0.01  # 1% tolerance
        if abs(current_price - pattern.wave_b_end.price) > entry_tolerance:
            # Not at entry point yet
            return None

        # Calculate stop loss
        buffer = max(current_price * 0.001, current_atr * self.buffer_atr_mult)
        if direction == WaveDirection.UP:
            stop_loss = pattern.wave_a_start.price - buffer
        else:
            stop_loss = pattern.wave_a_start.price + buffer

        # Calculate targets for Wave C
        wave_a_range = pattern.wave_a_range
        if direction == WaveDirection.UP:
            tp1 = pattern.wave_b_end.price + wave_a_range * 0.618
            tp2 = pattern.wave_b_end.price + wave_a_range * 1.0
        else:
            tp1 = pattern.wave_b_end.price - wave_a_range * 0.618
            tp2 = pattern.wave_b_end.price - wave_a_range * 1.0

        signal = Signal(
            signal_type=signal_type,
            timestamp=current_bar['timestamp'],
            bar_index=bar_index,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit_1=tp1,
            take_profit_2=tp2,
            module='C',
            is_confirmation_entry=False,
            wave_context={
                'pattern_type': 'zigzag',
                'wave_a_start': pattern.wave_a_start.price,
                'wave_a_end': pattern.wave_a_end.price,
                'wave_b_end': pattern.wave_b_end.price,
                'b_retracement': pattern.b_retracement
            }
        )

        return signal

    def _evaluate_flat(
        self,
        pattern: FlatPattern,
        current_bar: pd.Series,
        bar_index: int,
        current_price: float,
        current_atr: float
    ) -> Optional[Signal]:
        """
        Evaluate flat pattern for Wave C entry
        """
        if pattern.wave_a_end.price < pattern.wave_a_start.price:
            direction = WaveDirection.DOWN
            signal_type = SignalType.FLAT_SHORT
        else:
            direction = WaveDirection.UP
            signal_type = SignalType.FLAT_LONG

        # Check if at entry point
        entry_tolerance = current_price * 0.01
        if abs(current_price - pattern.wave_b_end.price) > entry_tolerance:
            return None

        # Stop loss beyond Wave B extreme
        buffer = max(current_price * 0.001, current_atr * self.buffer_atr_mult)
        if direction == WaveDirection.UP:
            stop_loss = pattern.wave_b_end.price - buffer
        else:
            stop_loss = pattern.wave_b_end.price + buffer

        # Targets for Wave C (flats often have strong C waves)
        wave_a_range = pattern.wave_a_range
        if direction == WaveDirection.UP:
            tp1 = pattern.wave_b_end.price + wave_a_range * 1.0
            tp2 = pattern.wave_b_end.price + wave_a_range * 1.618
        else:
            tp1 = pattern.wave_b_end.price - wave_a_range * 1.0
            tp2 = pattern.wave_b_end.price - wave_a_range * 1.618

        signal = Signal(
            signal_type=signal_type,
            timestamp=current_bar['timestamp'],
            bar_index=bar_index,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit_1=tp1,
            take_profit_2=tp2,
            module='C',
            is_confirmation_entry=False,
            wave_context={
                'pattern_type': 'flat',
                'flat_type': pattern.flat_type.value,
                'wave_a_start': pattern.wave_a_start.price,
                'wave_a_end': pattern.wave_a_end.price,
                'wave_b_end': pattern.wave_b_end.price,
                'b_retracement': pattern.b_retracement
            }
        )

        return signal

    def _evaluate_triangle(
        self,
        pattern: TrianglePattern,
        current_bar: pd.Series,
        bar_index: int,
        current_price: float,
        current_atr: float
    ) -> Optional[Signal]:
        """
        Evaluate triangle pattern for thrust entry
        """
        if pattern.thrust_direction == 1:
            direction = WaveDirection.UP
            signal_type = SignalType.TRIANGLE_LONG
        else:
            direction = WaveDirection.DOWN
            signal_type = SignalType.TRIANGLE_SHORT

        # Check if near Wave D/E area (waiting for breakout)
        entry_tolerance = current_price * 0.015
        expected_e_area = pattern.wave_d_end.price
        if abs(current_price - expected_e_area) > entry_tolerance:
            return None

        # Stop loss beyond Wave E extreme (estimated)
        buffer = max(current_price * 0.001, current_atr * self.buffer_atr_mult)
        if direction == WaveDirection.UP:
            stop_loss = pattern.wave_d_end.price - pattern.width_at_start * 0.2 - buffer
        else:
            stop_loss = pattern.wave_d_end.price + pattern.width_at_start * 0.2 + buffer

        # Thrust targets
        thrust = pattern.width_at_start
        if direction == WaveDirection.UP:
            tp1 = current_price + thrust * 0.5
            tp2 = current_price + thrust
        else:
            tp1 = current_price - thrust * 0.5
            tp2 = current_price - thrust

        signal = Signal(
            signal_type=signal_type,
            timestamp=current_bar['timestamp'],
            bar_index=bar_index,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit_1=tp1,
            take_profit_2=tp2,
            module='C',
            is_confirmation_entry=False,
            wave_context={
                'pattern_type': 'triangle',
                'triangle_type': pattern.triangle_type.value,
                'width_at_start': pattern.width_at_start,
                'thrust_direction': pattern.thrust_direction
            }
        )

        return signal
