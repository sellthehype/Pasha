"""
Corrective wave pattern detection (Zigzag, Flat, Triangle)
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum

from ..indicators.zigzag import Pivot, PivotType


class CorrectionType(Enum):
    ZIGZAG = "zigzag"
    FLAT_REGULAR = "flat_regular"
    FLAT_EXPANDED = "flat_expanded"
    FLAT_RUNNING = "flat_running"
    TRIANGLE_CONTRACTING = "triangle_contracting"
    TRIANGLE_EXPANDING = "triangle_expanding"


@dataclass
class ZigzagPattern:
    """A-B-C Zigzag correction pattern"""
    wave_a_start: Pivot
    wave_a_end: Pivot
    wave_b_end: Pivot
    wave_c_end: Optional[Pivot] = None

    is_complete: bool = False
    is_valid: bool = True
    b_retracement: float = 0.0  # B as % of A

    @property
    def wave_a_range(self) -> float:
        return abs(self.wave_a_end.price - self.wave_a_start.price)

    @property
    def wave_b_range(self) -> float:
        return abs(self.wave_b_end.price - self.wave_a_end.price)

    @property
    def wave_c_range(self) -> Optional[float]:
        if self.wave_c_end:
            return abs(self.wave_c_end.price - self.wave_b_end.price)
        return None

    def calculate_b_retracement(self):
        """Calculate B wave retracement percentage"""
        if self.wave_a_range > 0:
            self.b_retracement = self.wave_b_range / self.wave_a_range


@dataclass
class FlatPattern:
    """A-B-C Flat correction pattern"""
    wave_a_start: Pivot
    wave_a_end: Pivot
    wave_b_end: Pivot
    wave_c_end: Optional[Pivot] = None

    flat_type: CorrectionType = CorrectionType.FLAT_REGULAR
    is_complete: bool = False
    is_valid: bool = True
    b_retracement: float = 0.0

    @property
    def wave_a_range(self) -> float:
        return abs(self.wave_a_end.price - self.wave_a_start.price)

    @property
    def wave_b_range(self) -> float:
        return abs(self.wave_b_end.price - self.wave_a_end.price)


@dataclass
class TrianglePattern:
    """A-B-C-D-E Triangle pattern"""
    wave_a_start: Pivot
    wave_a_end: Pivot
    wave_b_end: Pivot
    wave_c_end: Pivot
    wave_d_end: Pivot
    wave_e_end: Optional[Pivot] = None

    triangle_type: CorrectionType = CorrectionType.TRIANGLE_CONTRACTING
    is_complete: bool = False
    is_valid: bool = True
    thrust_direction: int = 1  # 1 = up, -1 = down

    @property
    def width_at_start(self) -> float:
        """Triangle width at the beginning (A to B range)"""
        return abs(self.wave_b_end.price - self.wave_a_end.price)


class CorrectiveWaveDetector:
    """Detects corrective wave patterns from pivot points"""

    def __init__(self):
        # Zigzag B retracement range
        self.zigzag_b_min = 0.38
        self.zigzag_b_max = 0.79

        # Flat B retracement minimum
        self.flat_b_min = 0.90

        # Triangle wave relationship
        self.triangle_fib_min = 0.618
        self.triangle_fib_max = 0.786

    def find_zigzags(self, pivots: List[Pivot]) -> List[ZigzagPattern]:
        """
        Find zigzag patterns in pivot sequence

        Args:
            pivots: List of pivot points

        Returns:
            List of ZigzagPattern objects
        """
        patterns = []

        if len(pivots) < 4:
            return patterns

        for i in range(len(pivots) - 3):
            pattern = self._try_identify_zigzag(pivots[i:i + 4])
            if pattern and pattern.is_valid:
                patterns.append(pattern)

        return patterns

    def find_partial_zigzags(self, pivots: List[Pivot]) -> List[ZigzagPattern]:
        """
        Find incomplete zigzags (A-B complete, waiting for C)

        Args:
            pivots: List of pivot points

        Returns:
            List of partial ZigzagPattern objects
        """
        patterns = []

        if len(pivots) < 3:
            return patterns

        for i in range(len(pivots) - 2):
            pattern = self._try_identify_partial_zigzag(pivots[i:i + 3])
            if pattern and pattern.is_valid:
                patterns.append(pattern)

        return patterns

    def _try_identify_zigzag(self, pivots: List[Pivot]) -> Optional[ZigzagPattern]:
        """Try to identify a complete zigzag from 4 pivots"""
        if len(pivots) < 4:
            return None

        p0, p1, p2, p3 = pivots[:4]

        # Create pattern
        pattern = ZigzagPattern(
            wave_a_start=p0,
            wave_a_end=p1,
            wave_b_end=p2,
            wave_c_end=p3,
            is_complete=True
        )

        pattern.calculate_b_retracement()

        # Validate: B retracement should be 38-79% of A
        if not (self.zigzag_b_min <= pattern.b_retracement <= self.zigzag_b_max):
            pattern.is_valid = False

        # B should not exceed A start
        if p1.price > p0.price:  # A was down
            if p2.price >= p0.price:
                pattern.is_valid = False
        else:  # A was up
            if p2.price <= p0.price:
                pattern.is_valid = False

        return pattern

    def _try_identify_partial_zigzag(self, pivots: List[Pivot]) -> Optional[ZigzagPattern]:
        """Try to identify partial zigzag (A-B complete)"""
        if len(pivots) < 3:
            return None

        p0, p1, p2 = pivots[:3]

        pattern = ZigzagPattern(
            wave_a_start=p0,
            wave_a_end=p1,
            wave_b_end=p2,
            is_complete=False
        )

        pattern.calculate_b_retracement()

        # Validate B retracement
        if not (self.zigzag_b_min <= pattern.b_retracement <= self.zigzag_b_max):
            pattern.is_valid = False

        # B should not exceed A start
        if p1.price > p0.price:
            if p2.price >= p0.price:
                pattern.is_valid = False
        else:
            if p2.price <= p0.price:
                pattern.is_valid = False

        return pattern

    def find_flats(self, pivots: List[Pivot]) -> List[FlatPattern]:
        """
        Find flat patterns in pivot sequence

        Args:
            pivots: List of pivot points

        Returns:
            List of FlatPattern objects
        """
        patterns = []

        if len(pivots) < 4:
            return patterns

        for i in range(len(pivots) - 3):
            pattern = self._try_identify_flat(pivots[i:i + 4])
            if pattern and pattern.is_valid:
                patterns.append(pattern)

        return patterns

    def find_partial_flats(self, pivots: List[Pivot]) -> List[FlatPattern]:
        """Find incomplete flats (A-B complete)"""
        patterns = []

        if len(pivots) < 3:
            return patterns

        for i in range(len(pivots) - 2):
            pattern = self._try_identify_partial_flat(pivots[i:i + 3])
            if pattern and pattern.is_valid:
                patterns.append(pattern)

        return patterns

    def _try_identify_flat(self, pivots: List[Pivot]) -> Optional[FlatPattern]:
        """Try to identify a complete flat from 4 pivots"""
        if len(pivots) < 4:
            return None

        p0, p1, p2, p3 = pivots[:4]

        pattern = FlatPattern(
            wave_a_start=p0,
            wave_a_end=p1,
            wave_b_end=p2,
            wave_c_end=p3,
            is_complete=True
        )

        # Calculate B retracement
        if pattern.wave_a_range > 0:
            pattern.b_retracement = pattern.wave_b_range / pattern.wave_a_range

        # Validate: B should retrace at least 90% of A
        if pattern.b_retracement < self.flat_b_min:
            pattern.is_valid = False
            return pattern

        # Determine flat type
        if pattern.b_retracement >= 1.05:
            if p1.price > p0.price:  # A was up
                if p2.price > p0.price:  # B exceeds A start
                    pattern.flat_type = CorrectionType.FLAT_EXPANDED
            else:  # A was down
                if p2.price < p0.price:
                    pattern.flat_type = CorrectionType.FLAT_EXPANDED
        else:
            pattern.flat_type = CorrectionType.FLAT_REGULAR

        return pattern

    def _try_identify_partial_flat(self, pivots: List[Pivot]) -> Optional[FlatPattern]:
        """Try to identify partial flat (A-B complete)"""
        if len(pivots) < 3:
            return None

        p0, p1, p2 = pivots[:3]

        pattern = FlatPattern(
            wave_a_start=p0,
            wave_a_end=p1,
            wave_b_end=p2,
            is_complete=False
        )

        if pattern.wave_a_range > 0:
            pattern.b_retracement = pattern.wave_b_range / pattern.wave_a_range

        # B should retrace at least 90% of A
        if pattern.b_retracement < self.flat_b_min:
            pattern.is_valid = False

        return pattern

    def find_triangles(self, pivots: List[Pivot]) -> List[TrianglePattern]:
        """
        Find triangle patterns using Fibonacci relationships

        Args:
            pivots: List of pivot points

        Returns:
            List of TrianglePattern objects
        """
        patterns = []

        if len(pivots) < 6:
            return patterns

        for i in range(len(pivots) - 5):
            pattern = self._try_identify_triangle(pivots[i:i + 6])
            if pattern and pattern.is_valid:
                patterns.append(pattern)

        return patterns

    def find_partial_triangles(self, pivots: List[Pivot]) -> List[TrianglePattern]:
        """Find incomplete triangles (A-B-C-D complete, waiting for E)"""
        patterns = []

        if len(pivots) < 5:
            return patterns

        for i in range(len(pivots) - 4):
            pattern = self._try_identify_partial_triangle(pivots[i:i + 5])
            if pattern and pattern.is_valid:
                patterns.append(pattern)

        return patterns

    def _try_identify_triangle(self, pivots: List[Pivot]) -> Optional[TrianglePattern]:
        """Try to identify a triangle using Fib relationships"""
        if len(pivots) < 6:
            return None

        p0, p1, p2, p3, p4, p5 = pivots[:6]

        pattern = TrianglePattern(
            wave_a_start=p0,
            wave_a_end=p1,
            wave_b_end=p2,
            wave_c_end=p3,
            wave_d_end=p4,
            wave_e_end=p5,
            is_complete=True
        )

        # Determine thrust direction (same as before triangle)
        # Usually triangle appears in Wave 4 or Wave B, thrust continues prior trend
        if p1.price < p0.price:  # First move down
            pattern.thrust_direction = 1  # Thrust up after
        else:
            pattern.thrust_direction = -1

        # Validate using Fibonacci relationships
        # Each wave should retrace 61.8-78.6% of previous
        pattern.is_valid = self._validate_triangle_fib(pattern)

        return pattern

    def _try_identify_partial_triangle(self, pivots: List[Pivot]) -> Optional[TrianglePattern]:
        """Try to identify partial triangle (A-D complete)"""
        if len(pivots) < 5:
            return None

        p0, p1, p2, p3, p4 = pivots[:5]

        pattern = TrianglePattern(
            wave_a_start=p0,
            wave_a_end=p1,
            wave_b_end=p2,
            wave_c_end=p3,
            wave_d_end=p4,
            is_complete=False
        )

        if p1.price < p0.price:
            pattern.thrust_direction = 1
        else:
            pattern.thrust_direction = -1

        # Validate first 4 waves
        ranges = [
            abs(p1.price - p0.price),  # A
            abs(p2.price - p1.price),  # B
            abs(p3.price - p2.price),  # C
            abs(p4.price - p3.price),  # D
        ]

        # Check contracting (each wave smaller than previous)
        is_contracting = all(ranges[i] > ranges[i + 1] * 0.5 for i in range(len(ranges) - 1))

        if not is_contracting:
            pattern.is_valid = False

        return pattern

    def _validate_triangle_fib(self, pattern: TrianglePattern) -> bool:
        """
        Validate triangle using Fibonacci relationships

        Each wave should retrace 61.8-78.6% of previous wave
        """
        waves = [
            abs(pattern.wave_a_end.price - pattern.wave_a_start.price),
            abs(pattern.wave_b_end.price - pattern.wave_a_end.price),
            abs(pattern.wave_c_end.price - pattern.wave_b_end.price),
            abs(pattern.wave_d_end.price - pattern.wave_c_end.price),
        ]

        if pattern.wave_e_end:
            waves.append(abs(pattern.wave_e_end.price - pattern.wave_d_end.price))

        # Check Fibonacci relationships
        for i in range(1, len(waves)):
            if waves[i - 1] == 0:
                return False
            ratio = waves[i] / waves[i - 1]

            # Allow some flexibility around 61.8-78.6%
            if not (0.5 <= ratio <= 0.9):
                return False

        # Check overall contraction
        if len(waves) >= 4:
            if waves[-1] >= waves[0]:
                return False  # Not contracting

        return True
