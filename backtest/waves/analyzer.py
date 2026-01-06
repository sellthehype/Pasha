"""
Wave analysis coordinator - combines all wave detection
"""

import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from ..indicators.zigzag import ZigzagIndicator, Pivot
from ..indicators.fibonacci import FibonacciCalculator
from .impulse import ImpulseWaveDetector, ImpulseWave, PartialImpulse, WaveDirection
from .corrective import (
    CorrectiveWaveDetector,
    ZigzagPattern,
    FlatPattern,
    TrianglePattern
)
from .validator import WaveValidator


@dataclass
class WaveStructure:
    """Complete analysis of wave structure at a point in time"""
    pivots: List[Pivot] = field(default_factory=list)

    # Impulse patterns
    complete_impulses: List[ImpulseWave] = field(default_factory=list)
    partial_impulses_wave3: List[PartialImpulse] = field(default_factory=list)  # Ready for W3
    partial_impulses_wave5: List[PartialImpulse] = field(default_factory=list)  # Ready for W5

    # Corrective patterns
    zigzags: List[ZigzagPattern] = field(default_factory=list)
    partial_zigzags: List[ZigzagPattern] = field(default_factory=list)
    flats: List[FlatPattern] = field(default_factory=list)
    partial_flats: List[FlatPattern] = field(default_factory=list)
    triangles: List[TrianglePattern] = field(default_factory=list)
    partial_triangles: List[TrianglePattern] = field(default_factory=list)

    # Diagonal flags
    diagonal_flags: List[Tuple[int, int]] = field(default_factory=list)  # (start_idx, end_idx)


class WaveAnalyzer:
    """
    Main wave analysis engine that coordinates all detection
    """

    def __init__(
        self,
        zigzag_atr_mult: float = 1.5,
        zigzag_atr_period: int = 20,
        min_wave_pct: float = 0.5,
        fib_tolerance: float = 2.5
    ):
        """
        Initialize wave analyzer

        Args:
            zigzag_atr_mult: ATR multiplier for zigzag
            zigzag_atr_period: ATR period
            min_wave_pct: Minimum wave size percentage
            fib_tolerance: Fibonacci level tolerance percentage
        """
        self.zigzag = ZigzagIndicator(
            atr_multiplier=zigzag_atr_mult,
            atr_period=zigzag_atr_period,
            min_wave_pct=min_wave_pct
        )
        self.impulse_detector = ImpulseWaveDetector()
        self.corrective_detector = CorrectiveWaveDetector()
        self.validator = WaveValidator()
        self.fib_calc = FibonacciCalculator(tolerance_pct=fib_tolerance)

        self._cached_pivots: List[Pivot] = []
        self._last_processed_idx: int = -1

    def analyze(self, df: pd.DataFrame) -> WaveStructure:
        """
        Analyze complete wave structure of price data

        Args:
            df: DataFrame with OHLCV data

        Returns:
            WaveStructure with all identified patterns
        """
        # Find pivots
        pivots = self.zigzag.find_pivots(df)

        if len(pivots) < 3:
            return WaveStructure(pivots=pivots)

        structure = WaveStructure(pivots=pivots)

        # Find impulse patterns
        structure.complete_impulses = self.impulse_detector.find_impulse_waves(pivots)
        partial_impulses = self.impulse_detector.find_partial_impulses(pivots)

        # Separate by current wave
        for partial in partial_impulses:
            if partial.current_wave == 3:
                structure.partial_impulses_wave3.append(partial)
            elif partial.current_wave == 5:
                structure.partial_impulses_wave5.append(partial)

        # Find corrective patterns
        structure.zigzags = self.corrective_detector.find_zigzags(pivots)
        structure.partial_zigzags = self.corrective_detector.find_partial_zigzags(pivots)
        structure.flats = self.corrective_detector.find_flats(pivots)
        structure.partial_flats = self.corrective_detector.find_partial_flats(pivots)
        structure.triangles = self.corrective_detector.find_triangles(pivots)
        structure.partial_triangles = self.corrective_detector.find_partial_triangles(pivots)

        # Flag diagonals
        for impulse in structure.complete_impulses:
            if impulse.is_diagonal:
                structure.diagonal_flags.append(
                    (impulse.wave1_start.index, impulse.wave5_end.index if impulse.wave5_end else impulse.wave4_end.index)
                )

        return structure

    def analyze_incremental(
        self,
        df: pd.DataFrame,
        previous_structure: Optional[WaveStructure] = None
    ) -> WaveStructure:
        """
        Incrementally analyze new data (for backtesting efficiency)

        Args:
            df: DataFrame with OHLCV data
            previous_structure: Previously calculated structure

        Returns:
            Updated WaveStructure
        """
        # For now, use full analysis (optimization can come later)
        return self.analyze(df)

    def get_wave3_setups(
        self,
        structure: WaveStructure,
        current_price: float
    ) -> List[Dict]:
        """
        Get actionable Wave 3 entry setups

        Args:
            structure: Current wave structure
            current_price: Current price

        Returns:
            List of setup dictionaries
        """
        setups = []

        for partial in structure.partial_impulses_wave3:
            if not partial.is_valid:
                continue

            # Calculate Fibonacci entry zone
            zone_top, zone_bottom, fib_levels = self.fib_calc.get_wave2_entry_zone(
                partial.wave1_start.price,
                partial.wave1_end.price
            )

            # Check if current price is in entry zone
            if partial.direction == WaveDirection.UP:
                in_zone = zone_bottom <= current_price <= zone_top
            else:
                in_zone = zone_top <= current_price <= zone_bottom

            if in_zone:
                # Calculate targets
                targets = self.fib_calc.get_wave3_targets(
                    partial.wave1_start.price,
                    partial.wave1_end.price,
                    partial.wave2_end.price if partial.wave2_end else current_price
                )

                # Get invalidation level
                invalidation = self.validator.get_invalidation_price(partial, "rule1")

                setups.append({
                    'type': 'wave3',
                    'direction': partial.direction,
                    'partial': partial,
                    'entry_zone': (zone_top, zone_bottom),
                    'fib_levels': fib_levels,
                    'targets': targets,
                    'invalidation': invalidation,
                    'current_price': current_price
                })

        return setups

    def get_wave5_setups(
        self,
        structure: WaveStructure,
        current_price: float
    ) -> List[Dict]:
        """
        Get actionable Wave 5 entry setups

        Args:
            structure: Current wave structure
            current_price: Current price

        Returns:
            List of setup dictionaries
        """
        setups = []

        for partial in structure.partial_impulses_wave5:
            if not partial.is_valid:
                continue

            # Check if Wave 3 was extended
            is_extended = self.impulse_detector.is_wave3_extended(partial)

            # Calculate entry zone
            zone_top, zone_bottom, fib_levels = self.fib_calc.get_wave4_entry_zone(
                partial.wave2_end.price,  # Wave 3 start
                partial.wave3_end.price,
                partial.wave1_end.price,
                is_extended
            )

            # Check if in zone
            if partial.direction == WaveDirection.UP:
                in_zone = zone_bottom <= current_price <= zone_top
            else:
                in_zone = zone_top <= current_price <= zone_bottom

            if in_zone:
                targets = self.fib_calc.get_wave5_targets(
                    partial.wave1_start.price,
                    partial.wave1_end.price,
                    partial.wave4_end.price if partial.wave4_end else current_price,
                    is_extended
                )

                invalidation = self.validator.get_invalidation_price(partial, "rule3")

                setups.append({
                    'type': 'wave5',
                    'direction': partial.direction,
                    'partial': partial,
                    'entry_zone': (zone_top, zone_bottom),
                    'fib_levels': fib_levels,
                    'targets': targets,
                    'invalidation': invalidation,
                    'wave3_extended': is_extended,
                    'current_price': current_price
                })

        return setups

    def get_corrective_setups(
        self,
        structure: WaveStructure,
        current_price: float
    ) -> List[Dict]:
        """
        Get actionable corrective wave entry setups

        Args:
            structure: Current wave structure
            current_price: Current price

        Returns:
            List of setup dictionaries
        """
        setups = []

        # Zigzag setups (Wave C entry)
        for zigzag in structure.partial_zigzags:
            if not zigzag.is_valid:
                continue

            # Entry at B completion for C wave
            # Direction: opposite of A
            if zigzag.wave_a_end.price < zigzag.wave_a_start.price:
                direction = WaveDirection.DOWN  # C will go down
            else:
                direction = WaveDirection.UP

            # Targets for Wave C
            c_targets = [
                zigzag.wave_b_end.price + (zigzag.wave_a_range * 0.618 * (1 if direction == WaveDirection.UP else -1)),
                zigzag.wave_b_end.price + (zigzag.wave_a_range * 1.0 * (1 if direction == WaveDirection.UP else -1)),
                zigzag.wave_b_end.price + (zigzag.wave_a_range * 1.618 * (1 if direction == WaveDirection.UP else -1)),
            ]

            setups.append({
                'type': 'zigzag_c',
                'direction': direction,
                'pattern': zigzag,
                'entry_price': zigzag.wave_b_end.price,
                'targets': c_targets,
                'invalidation': zigzag.wave_a_start.price,
                'current_price': current_price
            })

        # Flat setups
        for flat in structure.partial_flats:
            if not flat.is_valid:
                continue

            if flat.wave_a_end.price < flat.wave_a_start.price:
                direction = WaveDirection.DOWN
            else:
                direction = WaveDirection.UP

            c_targets = [
                flat.wave_b_end.price + (flat.wave_a_range * 1.0 * (1 if direction == WaveDirection.UP else -1)),
                flat.wave_b_end.price + (flat.wave_a_range * 1.618 * (1 if direction == WaveDirection.UP else -1)),
            ]

            setups.append({
                'type': 'flat_c',
                'direction': direction,
                'pattern': flat,
                'entry_price': flat.wave_b_end.price,
                'targets': c_targets,
                'invalidation': flat.wave_b_end.price,
                'current_price': current_price
            })

        # Triangle setups (thrust after E)
        for triangle in structure.partial_triangles:
            if not triangle.is_valid:
                continue

            direction = WaveDirection.UP if triangle.thrust_direction == 1 else WaveDirection.DOWN

            # Thrust target = triangle width at start
            thrust_target = triangle.width_at_start

            if direction == WaveDirection.UP:
                target = triangle.wave_d_end.price + thrust_target
            else:
                target = triangle.wave_d_end.price - thrust_target

            setups.append({
                'type': 'triangle_thrust',
                'direction': direction,
                'pattern': triangle,
                'entry_price': triangle.wave_d_end.price,
                'targets': [target * 0.5 + triangle.wave_d_end.price * 0.5, target],  # 50% and 100% of thrust
                'invalidation': triangle.wave_d_end.price,
                'current_price': current_price
            })

        return setups
