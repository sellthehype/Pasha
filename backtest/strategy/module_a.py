"""
Module A: Wave 3 Entry Strategy
"""

import pandas as pd
from typing import List, Optional, Dict
from dataclasses import dataclass

from .signals import Signal, SignalType
from ..waves.analyzer import WaveAnalyzer, WaveStructure
from ..waves.impulse import WaveDirection, PartialImpulse
from ..indicators.fibonacci import FibonacciCalculator
from ..indicators.atr import calculate_atr


@dataclass
class Wave3Setup:
    """Setup for Wave 3 entry"""
    partial: PartialImpulse
    entry_zone_top: float
    entry_zone_bottom: float
    fib_levels: List
    targets: List
    invalidation: float
    direction: WaveDirection


class ModuleAStrategy:
    """
    Wave 3 Entry Strategy

    Enters at the end of Wave 2 to catch Wave 3
    """

    def __init__(
        self,
        fib_tolerance_pct: float = 2.5,
        entry_fib_levels: List[float] = None,
        initial_position_pct: float = 50.0,
        buffer_atr_mult: float = 0.5
    ):
        """
        Initialize Module A

        Args:
            fib_tolerance_pct: Tolerance around Fib levels
            entry_fib_levels: Which Fib levels to enter at
            initial_position_pct: Percentage for initial entry
            buffer_atr_mult: ATR multiplier for stop buffer
        """
        self.fib_calc = FibonacciCalculator(tolerance_pct=fib_tolerance_pct)
        self.entry_fib_levels = entry_fib_levels or [0.5, 0.618, 0.786]
        self.initial_position_pct = initial_position_pct
        self.buffer_atr_mult = buffer_atr_mult

    def generate_signals(
        self,
        df: pd.DataFrame,
        structure: WaveStructure,
        bar_index: int
    ) -> List[Signal]:
        """
        Generate Wave 3 entry signals

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

        # Get ATR for buffer calculation
        atr = calculate_atr(df.iloc[:bar_index + 1])
        current_atr = atr.iloc[-1] if len(atr) > 0 else current_price * 0.01

        for partial in structure.partial_impulses_wave3:
            if not partial.is_valid:
                continue

            signal = self._evaluate_setup(
                partial, current_bar, bar_index, current_price, current_atr
            )
            if signal:
                signals.append(signal)

        return signals

    def _evaluate_setup(
        self,
        partial: PartialImpulse,
        current_bar: pd.Series,
        bar_index: int,
        current_price: float,
        current_atr: float
    ) -> Optional[Signal]:
        """
        Evaluate a single Wave 3 setup

        Args:
            partial: Partial impulse pattern
            current_bar: Current price bar
            bar_index: Bar index
            current_price: Current price
            current_atr: Current ATR

        Returns:
            Signal if valid setup found
        """
        # Calculate entry zone
        zone_top, zone_bottom, fib_levels = self.fib_calc.get_wave2_entry_zone(
            partial.wave1_start.price,
            partial.wave1_end.price,
            self.entry_fib_levels
        )

        # Check if price is in entry zone
        if partial.direction == WaveDirection.UP:
            in_zone = zone_bottom <= current_price <= zone_top
            # Also check low touched the zone
            in_zone = in_zone or (zone_bottom <= current_bar['low'] <= zone_top)
        else:
            in_zone = zone_top <= current_price <= zone_bottom
            in_zone = in_zone or (zone_top <= current_bar['high'] <= zone_bottom)

        if not in_zone:
            return None

        # Determine which Fib level was hit
        fib_hit = None
        for fib in fib_levels:
            tolerance = abs(fib.price * self.fib_calc.tolerance_pct / 100)
            if abs(current_price - fib.price) <= tolerance:
                fib_hit = fib.ratio
                break

        # Calculate stop loss (invalidation + buffer)
        invalidation = partial.wave1_start.price
        buffer = max(current_price * 0.001, current_atr * self.buffer_atr_mult)

        if partial.direction == WaveDirection.UP:
            stop_loss = invalidation - buffer
        else:
            stop_loss = invalidation + buffer

        # Calculate targets
        wave2_end = partial.wave2_end.price if partial.wave2_end else current_price
        targets = self.fib_calc.get_wave3_targets(
            partial.wave1_start.price,
            partial.wave1_end.price,
            wave2_end
        )

        if len(targets) < 2:
            return None

        # Create signal
        signal_type = SignalType.WAVE3_LONG if partial.direction == WaveDirection.UP else SignalType.WAVE3_SHORT

        signal = Signal(
            signal_type=signal_type,
            timestamp=current_bar['timestamp'],
            bar_index=bar_index,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit_1=targets[0].price,  # 100% extension
            take_profit_2=targets[1].price,  # 161.8% extension
            fib_level_hit=fib_hit,
            module='A',
            is_confirmation_entry=False,
            wave_context={
                'wave1_start': partial.wave1_start.price,
                'wave1_end': partial.wave1_end.price,
                'wave2_end': wave2_end,
                'invalidation': invalidation
            }
        )

        return signal

    def check_confirmation(
        self,
        df: pd.DataFrame,
        signal: Signal,
        current_index: int
    ) -> bool:
        """
        Check if higher low confirmation has formed

        Args:
            df: Price data
            signal: Original signal
            current_index: Current bar index

        Returns:
            True if confirmation found
        """
        if current_index <= signal.bar_index + 2:
            return False  # Need at least a few bars

        # Look for higher low pattern
        signal_bar = signal.bar_index
        subset = df.iloc[signal_bar:current_index + 1]

        if len(subset) < 3:
            return False

        # Find lowest low after signal
        low_idx = subset['low'].idxmin()
        lowest_low = subset.loc[low_idx, 'low']

        # Check if we've made a higher low after that
        if signal.is_long:
            # For longs, look for price to bounce, pullback, and make higher low
            post_low = subset.loc[low_idx:]
            if len(post_low) < 2:
                return False

            # Check if current low is higher than the lowest
            recent_low = post_low['low'].iloc[-1]
            if recent_low > lowest_low and recent_low > signal.stop_loss:
                return True
        else:
            # For shorts, look for lower high
            high_idx = subset['high'].idxmax()
            highest_high = subset.loc[high_idx, 'high']

            post_high = subset.loc[high_idx:]
            if len(post_high) < 2:
                return False

            recent_high = post_high['high'].iloc[-1]
            if recent_high < highest_high and recent_high < signal.stop_loss:
                return True

        return False
