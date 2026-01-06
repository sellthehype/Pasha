"""
Fibonacci level calculations for Elliott Wave analysis
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


# Standard Fibonacci retracement levels
FIB_RETRACEMENT_LEVELS = [0.236, 0.382, 0.5, 0.618, 0.786, 1.0]

# Standard Fibonacci extension levels
FIB_EXTENSION_LEVELS = [0.618, 1.0, 1.272, 1.382, 1.618, 2.0, 2.618, 3.618]


@dataclass
class FibLevel:
    """Represents a Fibonacci level"""
    ratio: float
    price: float
    label: str


class FibonacciCalculator:
    """Calculator for Fibonacci retracement and extension levels"""

    def __init__(self, tolerance_pct: float = 2.5):
        """
        Initialize calculator

        Args:
            tolerance_pct: Tolerance around Fib levels (e.g., 2.5 = ±2.5%)
        """
        self.tolerance_pct = tolerance_pct

    def calculate_retracements(
        self,
        start_price: float,
        end_price: float,
        levels: List[float] = None
    ) -> List[FibLevel]:
        """
        Calculate Fibonacci retracement levels

        Args:
            start_price: Wave start price
            end_price: Wave end price
            levels: Custom levels (defaults to standard)

        Returns:
            List of FibLevel objects
        """
        if levels is None:
            levels = FIB_RETRACEMENT_LEVELS

        wave_range = end_price - start_price
        fib_levels = []

        for ratio in levels:
            # Retracement from end back toward start
            price = end_price - (wave_range * ratio)
            fib_levels.append(FibLevel(
                ratio=ratio,
                price=price,
                label=f"{ratio * 100:.1f}%"
            ))

        return fib_levels

    def calculate_extensions(
        self,
        wave_start: float,
        wave_end: float,
        extension_start: float,
        levels: List[float] = None
    ) -> List[FibLevel]:
        """
        Calculate Fibonacci extension levels

        Args:
            wave_start: Start of the measured wave
            wave_end: End of the measured wave
            extension_start: Point to project extensions from
            levels: Custom levels (defaults to standard)

        Returns:
            List of FibLevel objects
        """
        if levels is None:
            levels = FIB_EXTENSION_LEVELS

        wave_range = abs(wave_end - wave_start)
        direction = 1 if wave_end > wave_start else -1

        fib_levels = []
        for ratio in levels:
            # Extension from the extension start point
            price = extension_start + (wave_range * ratio * direction)
            fib_levels.append(FibLevel(
                ratio=ratio,
                price=price,
                label=f"{ratio * 100:.1f}%"
            ))

        return fib_levels

    def is_at_fib_level(
        self,
        price: float,
        fib_levels: List[FibLevel],
        check_levels: List[float] = None
    ) -> Tuple[bool, Optional[FibLevel]]:
        """
        Check if price is at a Fibonacci level within tolerance

        Args:
            price: Current price to check
            fib_levels: List of calculated Fib levels
            check_levels: Specific ratios to check (None = check all)

        Returns:
            Tuple of (is_at_level, matched_level)
        """
        for fib in fib_levels:
            if check_levels and fib.ratio not in check_levels:
                continue

            tolerance = abs(fib.price * self.tolerance_pct / 100)
            if abs(price - fib.price) <= tolerance:
                return True, fib

        return False, None

    def get_wave2_entry_zone(
        self,
        wave1_start: float,
        wave1_end: float,
        entry_levels: List[float] = None
    ) -> Tuple[float, float, List[FibLevel]]:
        """
        Get the entry zone for Wave 3 entry (Wave 2 retracement area)

        Args:
            wave1_start: Wave 1 starting price
            wave1_end: Wave 1 ending price
            entry_levels: Fib levels to use for entry zone

        Returns:
            Tuple of (zone_top, zone_bottom, fib_levels)
        """
        if entry_levels is None:
            entry_levels = [0.5, 0.618, 0.786]

        retracements = self.calculate_retracements(wave1_start, wave1_end, entry_levels)

        prices = [fib.price for fib in retracements]
        zone_top = max(prices) * (1 + self.tolerance_pct / 100)
        zone_bottom = min(prices) * (1 - self.tolerance_pct / 100)

        # Ensure zone doesn't exceed 100% retracement
        invalidation = wave1_start
        if wave1_end > wave1_start:  # Uptrend
            zone_bottom = max(zone_bottom, invalidation)
        else:  # Downtrend
            zone_top = min(zone_top, invalidation)

        return zone_top, zone_bottom, retracements

    def get_wave3_targets(
        self,
        wave1_start: float,
        wave1_end: float,
        wave2_end: float
    ) -> List[FibLevel]:
        """
        Calculate Wave 3 profit targets

        Args:
            wave1_start: Wave 1 start
            wave1_end: Wave 1 end
            wave2_end: Wave 2 end (entry point)

        Returns:
            List of target levels
        """
        # Standard Wave 3 targets based on Wave 1 projected from Wave 2 end
        target_levels = [1.0, 1.618, 2.618]
        return self.calculate_extensions(wave1_start, wave1_end, wave2_end, target_levels)

    def get_wave4_entry_zone(
        self,
        wave3_start: float,
        wave3_end: float,
        wave1_end: float,
        is_extended: bool = False
    ) -> Tuple[float, float, List[FibLevel]]:
        """
        Get entry zone for Wave 5 entry (Wave 4 retracement)

        Args:
            wave3_start: Wave 3 start (Wave 2 end)
            wave3_end: Wave 3 end
            wave1_end: Wave 1 end (invalidation level)
            is_extended: Whether Wave 3 was extended

        Returns:
            Tuple of (zone_top, zone_bottom, fib_levels)
        """
        if is_extended:
            entry_levels = [0.236, 0.382]
        else:
            entry_levels = [0.382, 0.5]

        retracements = self.calculate_retracements(wave3_start, wave3_end, entry_levels)

        prices = [fib.price for fib in retracements]
        zone_top = max(prices) * (1 + self.tolerance_pct / 100)
        zone_bottom = min(prices) * (1 - self.tolerance_pct / 100)

        # Ensure Wave 4 doesn't enter Wave 1 territory
        if wave3_end > wave3_start:  # Uptrend
            zone_bottom = max(zone_bottom, wave1_end)
        else:  # Downtrend
            zone_top = min(zone_top, wave1_end)

        return zone_top, zone_bottom, retracements

    def get_wave5_targets(
        self,
        wave1_start: float,
        wave1_end: float,
        wave4_end: float,
        wave3_extended: bool = False
    ) -> List[FibLevel]:
        """
        Calculate Wave 5 profit targets

        Args:
            wave1_start: Wave 1 start
            wave1_end: Wave 1 end
            wave4_end: Wave 4 end (entry point)
            wave3_extended: Whether Wave 3 was extended

        Returns:
            List of target levels
        """
        if wave3_extended:
            # If W3 extended, W5 tends toward equality with W1
            target_levels = [0.618, 1.0]
        else:
            target_levels = [1.0, 1.618]

        return self.calculate_extensions(wave1_start, wave1_end, wave4_end, target_levels)

    def check_wave3_is_extended(
        self,
        wave1_range: float,
        wave3_range: float
    ) -> bool:
        """
        Check if Wave 3 is extended (>161.8% of Wave 1)

        Args:
            wave1_range: Absolute range of Wave 1
            wave3_range: Absolute range of Wave 3

        Returns:
            True if Wave 3 is extended
        """
        if wave1_range == 0:
            return False
        ratio = wave3_range / wave1_range
        return ratio > 1.618
