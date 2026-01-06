"""
Module B: Wave 5 Entry Strategy
"""

import pandas as pd
from typing import List, Optional
from dataclasses import dataclass

from .signals import Signal, SignalType
from ..waves.analyzer import WaveStructure
from ..waves.impulse import WaveDirection, PartialImpulse, ImpulseWaveDetector
from ..indicators.fibonacci import FibonacciCalculator
from ..indicators.atr import calculate_atr


class ModuleBStrategy:
    """
    Wave 5 Entry Strategy

    Enters at the end of Wave 4 to catch Wave 5
    """

    def __init__(
        self,
        fib_tolerance_pct: float = 2.5,
        entry_fib_levels_normal: List[float] = None,
        entry_fib_levels_extended: List[float] = None,
        initial_position_pct: float = 50.0,
        buffer_atr_mult: float = 0.5,
        use_tight_stop: bool = True
    ):
        """
        Initialize Module B

        Args:
            fib_tolerance_pct: Tolerance around Fib levels
            entry_fib_levels_normal: Fib levels when W3 is normal
            entry_fib_levels_extended: Fib levels when W3 is extended
            initial_position_pct: Percentage for initial entry
            buffer_atr_mult: ATR multiplier for stop buffer
            use_tight_stop: Use Wave 4 low instead of Wave 1 high for stop
        """
        self.fib_calc = FibonacciCalculator(tolerance_pct=fib_tolerance_pct)
        self.entry_fib_levels_normal = entry_fib_levels_normal or [0.382, 0.5]
        self.entry_fib_levels_extended = entry_fib_levels_extended or [0.236, 0.382]
        self.initial_position_pct = initial_position_pct
        self.buffer_atr_mult = buffer_atr_mult
        self.use_tight_stop = use_tight_stop
        self.impulse_detector = ImpulseWaveDetector()

    def generate_signals(
        self,
        df: pd.DataFrame,
        structure: WaveStructure,
        bar_index: int
    ) -> List[Signal]:
        """
        Generate Wave 5 entry signals

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

        for partial in structure.partial_impulses_wave5:
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
        Evaluate a single Wave 5 setup
        """
        # Check if Wave 3 was extended
        is_extended = self.impulse_detector.is_wave3_extended(partial)

        entry_levels = self.entry_fib_levels_extended if is_extended else self.entry_fib_levels_normal

        # Calculate entry zone (Wave 4 retracement of Wave 3)
        zone_top, zone_bottom, fib_levels = self.fib_calc.get_wave4_entry_zone(
            partial.wave2_end.price,  # Wave 3 start
            partial.wave3_end.price,  # Wave 3 end
            partial.wave1_end.price,  # Wave 1 high (invalidation)
            is_extended
        )

        # Check if price is in entry zone
        if partial.direction == WaveDirection.UP:
            in_zone = zone_bottom <= current_price <= zone_top
            in_zone = in_zone or (zone_bottom <= current_bar['low'] <= zone_top)
        else:
            in_zone = zone_top <= current_price <= zone_bottom
            in_zone = in_zone or (zone_top <= current_bar['high'] <= zone_bottom)

        if not in_zone:
            return None

        # Determine Fib level hit
        fib_hit = None
        for fib in fib_levels:
            tolerance = abs(fib.price * self.fib_calc.tolerance_pct / 100)
            if abs(current_price - fib.price) <= tolerance:
                fib_hit = fib.ratio
                break

        # Calculate stop loss
        buffer = max(current_price * 0.001, current_atr * self.buffer_atr_mult)

        if self.use_tight_stop and partial.wave4_end:
            # Tight stop below Wave 4
            if partial.direction == WaveDirection.UP:
                stop_loss = partial.wave4_end.price - buffer
            else:
                stop_loss = partial.wave4_end.price + buffer
        else:
            # Wide stop at Wave 1 end (invalidation)
            if partial.direction == WaveDirection.UP:
                stop_loss = partial.wave1_end.price - buffer
            else:
                stop_loss = partial.wave1_end.price + buffer

        # Calculate targets
        wave4_end = partial.wave4_end.price if partial.wave4_end else current_price
        targets = self.fib_calc.get_wave5_targets(
            partial.wave1_start.price,
            partial.wave1_end.price,
            wave4_end,
            is_extended
        )

        if len(targets) < 2:
            return None

        signal_type = SignalType.WAVE5_LONG if partial.direction == WaveDirection.UP else SignalType.WAVE5_SHORT

        signal = Signal(
            signal_type=signal_type,
            timestamp=current_bar['timestamp'],
            bar_index=bar_index,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit_1=targets[0].price,
            take_profit_2=targets[1].price,
            fib_level_hit=fib_hit,
            module='B',
            is_confirmation_entry=False,
            wave_context={
                'wave1_start': partial.wave1_start.price,
                'wave1_end': partial.wave1_end.price,
                'wave3_end': partial.wave3_end.price,
                'wave4_end': wave4_end,
                'wave3_extended': is_extended,
                'invalidation': partial.wave1_end.price
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
        Check if higher low confirmation has formed for Wave 5 entry
        """
        if current_index <= signal.bar_index + 2:
            return False

        signal_bar = signal.bar_index
        subset = df.iloc[signal_bar:current_index + 1]

        if len(subset) < 3:
            return False

        if signal.is_long:
            low_idx = subset['low'].idxmin()
            lowest_low = subset.loc[low_idx, 'low']

            post_low = subset.loc[low_idx:]
            if len(post_low) < 2:
                return False

            recent_low = post_low['low'].iloc[-1]
            if recent_low > lowest_low and recent_low > signal.stop_loss:
                return True
        else:
            high_idx = subset['high'].idxmax()
            highest_high = subset.loc[high_idx, 'high']

            post_high = subset.loc[high_idx:]
            if len(post_high) < 2:
                return False

            recent_high = post_high['high'].iloc[-1]
            if recent_high < highest_high and recent_high < signal.stop_loss:
                return True

        return False
