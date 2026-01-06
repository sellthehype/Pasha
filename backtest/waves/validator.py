"""
Elliott Wave rule validation
"""

from typing import Optional, Tuple
from dataclasses import dataclass

from .impulse import ImpulseWave, PartialImpulse, WaveDirection


@dataclass
class ValidationResult:
    """Result of wave validation"""
    is_valid: bool
    rule_violated: Optional[str] = None
    confidence: float = 1.0


class WaveValidator:
    """Validates Elliott Wave patterns against the three golden rules"""

    def validate_wave2(
        self,
        wave1_start: float,
        wave1_end: float,
        wave2_end: float,
        direction: WaveDirection
    ) -> ValidationResult:
        """
        Validate Wave 2: Never retraces more than 100% of Wave 1

        Args:
            wave1_start: Start price of Wave 1
            wave1_end: End price of Wave 1
            wave2_end: End price of Wave 2
            direction: Direction of the impulse

        Returns:
            ValidationResult
        """
        if direction == WaveDirection.UP:
            # Wave 1 was up, Wave 2 should not go below Wave 1 start
            if wave2_end <= wave1_start:
                return ValidationResult(
                    is_valid=False,
                    rule_violated="Rule 1: Wave 2 retraced more than 100% of Wave 1"
                )
            # Calculate confidence based on how deep the retracement is
            wave1_range = wave1_end - wave1_start
            wave2_retrace = wave1_end - wave2_end
            if wave1_range > 0:
                retrace_pct = wave2_retrace / wave1_range
                # Higher confidence if retracement is in 50-78.6% range
                if 0.5 <= retrace_pct <= 0.786:
                    confidence = 1.0
                elif retrace_pct < 0.382:
                    confidence = 0.7  # Too shallow
                elif retrace_pct > 0.786:
                    confidence = 0.8  # Deep but valid
                else:
                    confidence = 0.9
            else:
                confidence = 0.5
        else:
            # Wave 1 was down, Wave 2 should not go above Wave 1 start
            if wave2_end >= wave1_start:
                return ValidationResult(
                    is_valid=False,
                    rule_violated="Rule 1: Wave 2 retraced more than 100% of Wave 1"
                )
            wave1_range = wave1_start - wave1_end
            wave2_retrace = wave2_end - wave1_end
            if wave1_range > 0:
                retrace_pct = wave2_retrace / wave1_range
                if 0.5 <= retrace_pct <= 0.786:
                    confidence = 1.0
                elif retrace_pct < 0.382:
                    confidence = 0.7
                elif retrace_pct > 0.786:
                    confidence = 0.8
                else:
                    confidence = 0.9
            else:
                confidence = 0.5

        return ValidationResult(is_valid=True, confidence=confidence)

    def validate_wave3_not_shortest(
        self,
        wave1_range: float,
        wave3_range: float,
        wave5_range: float
    ) -> ValidationResult:
        """
        Validate Wave 3: Cannot be the shortest impulse wave

        Args:
            wave1_range: Absolute range of Wave 1
            wave3_range: Absolute range of Wave 3
            wave5_range: Absolute range of Wave 5

        Returns:
            ValidationResult
        """
        if wave3_range < wave1_range and wave3_range < wave5_range:
            return ValidationResult(
                is_valid=False,
                rule_violated="Rule 2: Wave 3 is the shortest wave"
            )

        # Calculate confidence based on Wave 3 relative size
        avg_other = (wave1_range + wave5_range) / 2
        if avg_other > 0:
            ratio = wave3_range / avg_other
            if ratio > 1.618:
                confidence = 1.0  # Extended Wave 3
            elif ratio > 1.0:
                confidence = 0.9
            else:
                confidence = 0.7  # Wave 3 is smallest but still larger than one

        return ValidationResult(is_valid=True, confidence=confidence)

    def validate_wave4_no_overlap(
        self,
        wave1_end: float,
        wave4_end: float,
        direction: WaveDirection
    ) -> Tuple[ValidationResult, bool]:
        """
        Validate Wave 4: Cannot enter Wave 1 price territory (except diagonals)

        Args:
            wave1_end: End price of Wave 1
            wave4_end: End price of Wave 4
            direction: Direction of the impulse

        Returns:
            Tuple of (ValidationResult, is_diagonal)
        """
        is_diagonal = False

        if direction == WaveDirection.UP:
            # Wave 4 should not go below Wave 1 high
            if wave4_end <= wave1_end:
                is_diagonal = True
                return ValidationResult(
                    is_valid=True,  # Valid as diagonal
                    rule_violated="Wave 4 overlaps Wave 1 - possible diagonal",
                    confidence=0.6  # Lower confidence for diagonals
                ), is_diagonal
        else:
            # Wave 4 should not go above Wave 1 low
            if wave4_end >= wave1_end:
                is_diagonal = True
                return ValidationResult(
                    is_valid=True,
                    rule_violated="Wave 4 overlaps Wave 1 - possible diagonal",
                    confidence=0.6
                ), is_diagonal

        return ValidationResult(is_valid=True, confidence=1.0), is_diagonal

    def validate_impulse(self, impulse: ImpulseWave) -> ValidationResult:
        """
        Validate a complete impulse pattern

        Args:
            impulse: ImpulseWave to validate

        Returns:
            ValidationResult
        """
        # Rule 1: Wave 2
        result1 = self.validate_wave2(
            impulse.wave1_start.price,
            impulse.wave1_end.price,
            impulse.wave2_end.price,
            impulse.direction
        )
        if not result1.is_valid:
            return result1

        # Rule 2: Wave 3 not shortest
        result2 = self.validate_wave3_not_shortest(
            impulse.wave1_range,
            impulse.wave3_range,
            impulse.wave5_range or 0
        )
        if not result2.is_valid:
            return result2

        # Rule 3: Wave 4 overlap
        result3, is_diagonal = self.validate_wave4_no_overlap(
            impulse.wave1_end.price,
            impulse.wave4_end.price,
            impulse.direction
        )
        impulse.is_diagonal = is_diagonal

        # Combined confidence
        confidence = min(result1.confidence, result2.confidence, result3.confidence)

        return ValidationResult(is_valid=True, confidence=confidence)

    def validate_partial_impulse(self, partial: PartialImpulse) -> ValidationResult:
        """
        Validate a partial impulse pattern

        Args:
            partial: PartialImpulse to validate

        Returns:
            ValidationResult
        """
        # Rule 1: Wave 2 (if Wave 2 exists)
        if partial.wave2_end:
            result1 = self.validate_wave2(
                partial.wave1_start.price,
                partial.wave1_end.price,
                partial.wave2_end.price,
                partial.direction
            )
            if not result1.is_valid:
                return result1

        # Rule 3: Wave 4 overlap (if Wave 4 exists)
        if partial.wave4_end:
            result3, _ = self.validate_wave4_no_overlap(
                partial.wave1_end.price,
                partial.wave4_end.price,
                partial.direction
            )
            if not result3.is_valid:
                return result3

        return ValidationResult(is_valid=True, confidence=0.9)

    def get_invalidation_price(
        self,
        partial: PartialImpulse,
        rule: str = "all"
    ) -> float:
        """
        Get the price level that would invalidate the wave count

        Args:
            partial: Partial impulse pattern
            rule: Which rule to check ("rule1", "rule3", or "all")

        Returns:
            Price at which the pattern becomes invalid
        """
        invalidation = None

        if rule in ["rule1", "all"]:
            # Rule 1: Wave 2 > 100% of Wave 1
            invalidation = partial.wave1_start.price

        if rule in ["rule3", "all"] and partial.wave4_end:
            # Rule 3: Wave 4 enters Wave 1 territory
            if partial.direction == WaveDirection.UP:
                rule3_invalidation = partial.wave1_end.price
                if invalidation is None or rule3_invalidation > invalidation:
                    invalidation = rule3_invalidation
            else:
                rule3_invalidation = partial.wave1_end.price
                if invalidation is None or rule3_invalidation < invalidation:
                    invalidation = rule3_invalidation

        return invalidation
