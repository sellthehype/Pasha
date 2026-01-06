"""
Impulse wave detection (5-wave motive patterns)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum
import numpy as np

from ..indicators.zigzag import Pivot, PivotType


class WaveDirection(Enum):
    UP = "up"
    DOWN = "down"


@dataclass
class ImpulseWave:
    """Represents a 5-wave impulse pattern"""
    direction: WaveDirection
    wave1_start: Pivot
    wave1_end: Pivot
    wave2_end: Pivot
    wave3_end: Pivot
    wave4_end: Pivot
    wave5_end: Optional[Pivot] = None

    # Calculated properties
    is_valid: bool = True
    is_diagonal: bool = False
    wave3_extended: bool = False
    invalidation_reason: Optional[str] = None

    @property
    def wave1_range(self) -> float:
        return abs(self.wave1_end.price - self.wave1_start.price)

    @property
    def wave2_range(self) -> float:
        return abs(self.wave2_end.price - self.wave1_end.price)

    @property
    def wave3_range(self) -> float:
        return abs(self.wave3_end.price - self.wave2_end.price)

    @property
    def wave4_range(self) -> float:
        return abs(self.wave4_end.price - self.wave3_end.price)

    @property
    def wave5_range(self) -> Optional[float]:
        if self.wave5_end:
            return abs(self.wave5_end.price - self.wave4_end.price)
        return None

    @property
    def wave2_retracement(self) -> float:
        """Wave 2 retracement as percentage of Wave 1"""
        if self.wave1_range == 0:
            return 0
        return self.wave2_range / self.wave1_range

    @property
    def wave4_retracement(self) -> float:
        """Wave 4 retracement as percentage of Wave 3"""
        if self.wave3_range == 0:
            return 0
        return self.wave4_range / self.wave3_range

    def check_wave3_extended(self) -> bool:
        """Check if Wave 3 is extended (>161.8% of Wave 1)"""
        if self.wave1_range == 0:
            return False
        ratio = self.wave3_range / self.wave1_range
        self.wave3_extended = ratio > 1.618
        return self.wave3_extended


@dataclass
class PartialImpulse:
    """Represents an incomplete impulse pattern (for entry detection)"""
    direction: WaveDirection
    wave1_start: Pivot
    wave1_end: Pivot
    wave2_end: Optional[Pivot] = None
    wave3_end: Optional[Pivot] = None
    wave4_end: Optional[Pivot] = None

    current_wave: int = 1  # Which wave we're currently in/looking for
    is_valid: bool = True


class ImpulseWaveDetector:
    """Detects impulse wave patterns from pivot points"""

    def __init__(self, min_wave_ratio: float = 0.382):
        """
        Initialize detector

        Args:
            min_wave_ratio: Minimum ratio for wave significance
        """
        self.min_wave_ratio = min_wave_ratio

    def find_impulse_waves(self, pivots: List[Pivot]) -> List[ImpulseWave]:
        """
        Find all complete impulse waves in pivot sequence

        Args:
            pivots: List of pivot points from zigzag

        Returns:
            List of identified impulse waves
        """
        if len(pivots) < 6:  # Need at least 6 pivots for 5 waves
            return []

        impulses = []

        # Try to find impulses starting from each pivot
        for i in range(len(pivots) - 5):
            impulse = self._try_identify_impulse(pivots[i:i + 6])
            if impulse and impulse.is_valid:
                impulses.append(impulse)

        return impulses

    def find_partial_impulses(self, pivots: List[Pivot]) -> List[PartialImpulse]:
        """
        Find incomplete impulse patterns (for entry identification)

        Args:
            pivots: List of pivot points

        Returns:
            List of partial impulses at various stages
        """
        partials = []

        if len(pivots) < 3:
            return partials

        # Look for Wave 1-2 complete (ready for Wave 3 entry)
        for i in range(len(pivots) - 2):
            partial = self._try_identify_wave12(pivots[i:])
            if partial and partial.is_valid:
                partials.append(partial)

        # Look for Wave 1-4 complete (ready for Wave 5 entry)
        if len(pivots) >= 5:
            for i in range(len(pivots) - 4):
                partial = self._try_identify_wave14(pivots[i:])
                if partial and partial.is_valid:
                    partials.append(partial)

        return partials

    def _try_identify_impulse(self, pivots: List[Pivot]) -> Optional[ImpulseWave]:
        """
        Try to identify a complete 5-wave impulse from 6 pivots

        Args:
            pivots: Exactly 6 pivots

        Returns:
            ImpulseWave if valid pattern found
        """
        if len(pivots) < 6:
            return None

        p0, p1, p2, p3, p4, p5 = pivots[:6]

        # Determine direction
        if p1.price > p0.price:
            direction = WaveDirection.UP
        else:
            direction = WaveDirection.DOWN

        impulse = ImpulseWave(
            direction=direction,
            wave1_start=p0,
            wave1_end=p1,
            wave2_end=p2,
            wave3_end=p3,
            wave4_end=p4,
            wave5_end=p5
        )

        # Validate Elliott Wave rules
        if not self._validate_impulse(impulse):
            return None

        impulse.check_wave3_extended()
        return impulse

    def _try_identify_wave12(self, pivots: List[Pivot]) -> Optional[PartialImpulse]:
        """
        Identify Wave 1-2 pattern (setup for Wave 3 entry)

        Args:
            pivots: Pivot sequence starting with potential Wave 1 start

        Returns:
            PartialImpulse if valid
        """
        if len(pivots) < 3:
            return None

        p0, p1, p2 = pivots[0], pivots[1], pivots[2]

        # Determine direction from Wave 1
        if p1.price > p0.price:
            direction = WaveDirection.UP
            # Wave 2 should retrace down
            if p2.price >= p1.price:
                return None  # Not a retracement
        else:
            direction = WaveDirection.DOWN
            # Wave 2 should retrace up
            if p2.price <= p1.price:
                return None  # Not a retracement

        partial = PartialImpulse(
            direction=direction,
            wave1_start=p0,
            wave1_end=p1,
            wave2_end=p2,
            current_wave=3  # Ready for Wave 3
        )

        # Validate Wave 2 doesn't exceed Wave 1 start
        if direction == WaveDirection.UP:
            if p2.price <= p0.price:
                partial.is_valid = False
        else:
            if p2.price >= p0.price:
                partial.is_valid = False

        return partial

    def _try_identify_wave14(self, pivots: List[Pivot]) -> Optional[PartialImpulse]:
        """
        Identify Wave 1-4 pattern (setup for Wave 5 entry)

        Args:
            pivots: Pivot sequence

        Returns:
            PartialImpulse if valid
        """
        if len(pivots) < 5:
            return None

        p0, p1, p2, p3, p4 = pivots[:5]

        # Determine direction
        if p1.price > p0.price:
            direction = WaveDirection.UP
        else:
            direction = WaveDirection.DOWN

        partial = PartialImpulse(
            direction=direction,
            wave1_start=p0,
            wave1_end=p1,
            wave2_end=p2,
            wave3_end=p3,
            wave4_end=p4,
            current_wave=5  # Ready for Wave 5
        )

        # Validate all rules for waves 1-4
        partial.is_valid = self._validate_waves_1_to_4(partial)

        return partial

    def _validate_impulse(self, impulse: ImpulseWave) -> bool:
        """
        Validate a complete impulse against Elliott Wave rules

        Args:
            impulse: ImpulseWave to validate

        Returns:
            True if valid
        """
        # Rule 1: Wave 2 never retraces more than 100% of Wave 1
        if impulse.direction == WaveDirection.UP:
            if impulse.wave2_end.price <= impulse.wave1_start.price:
                impulse.invalidation_reason = "Wave 2 exceeded 100% retracement"
                impulse.is_valid = False
                return False
        else:
            if impulse.wave2_end.price >= impulse.wave1_start.price:
                impulse.invalidation_reason = "Wave 2 exceeded 100% retracement"
                impulse.is_valid = False
                return False

        # Rule 2: Wave 3 cannot be the shortest
        wave1_range = impulse.wave1_range
        wave3_range = impulse.wave3_range
        wave5_range = impulse.wave5_range or 0

        if wave3_range < wave1_range and wave3_range < wave5_range:
            impulse.invalidation_reason = "Wave 3 is the shortest"
            impulse.is_valid = False
            return False

        # Rule 3: Wave 4 doesn't enter Wave 1 territory (except diagonals)
        if impulse.direction == WaveDirection.UP:
            if impulse.wave4_end.price <= impulse.wave1_end.price:
                # Check if it's a diagonal
                impulse.is_diagonal = True
                # For now, mark as valid but diagonal
        else:
            if impulse.wave4_end.price >= impulse.wave1_end.price:
                impulse.is_diagonal = True

        return True

    def _validate_waves_1_to_4(self, partial: PartialImpulse) -> bool:
        """
        Validate waves 1-4 for partial impulse

        Args:
            partial: PartialImpulse to validate

        Returns:
            True if valid
        """
        # Rule 1: Wave 2 < 100% of Wave 1
        if partial.direction == WaveDirection.UP:
            if partial.wave2_end.price <= partial.wave1_start.price:
                return False
        else:
            if partial.wave2_end.price >= partial.wave1_start.price:
                return False

        # Rule 3: Wave 4 doesn't enter Wave 1 territory
        if partial.direction == WaveDirection.UP:
            if partial.wave4_end.price <= partial.wave1_end.price:
                return False  # Overlap (could be diagonal, but skip for now)
        else:
            if partial.wave4_end.price >= partial.wave1_end.price:
                return False

        # Wave 3 should be significant
        wave1_range = abs(partial.wave1_end.price - partial.wave1_start.price)
        wave3_range = abs(partial.wave3_end.price - partial.wave2_end.price)

        if wave3_range < wave1_range * 0.5:
            return False  # Wave 3 too small

        return True

    def is_wave3_extended(self, partial: PartialImpulse) -> bool:
        """Check if Wave 3 is extended in a partial pattern"""
        if not partial.wave3_end:
            return False

        wave1_range = abs(partial.wave1_end.price - partial.wave1_start.price)
        wave3_range = abs(partial.wave3_end.price - partial.wave2_end.price)

        if wave1_range == 0:
            return False

        return wave3_range / wave1_range > 1.618
