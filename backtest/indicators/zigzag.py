"""
Zigzag indicator for swing point detection
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from .atr import calculate_atr


class PivotType(Enum):
    HIGH = "high"
    LOW = "low"


@dataclass
class Pivot:
    """Represents a swing high or low point"""
    index: int
    price: float
    pivot_type: PivotType
    timestamp: pd.Timestamp


class ZigzagIndicator:
    """
    Zigzag indicator that identifies swing highs and lows
    based on ATR-based reversal threshold
    """

    def __init__(
        self,
        atr_multiplier: float = 1.5,
        atr_period: int = 20,
        min_wave_pct: float = 0.5
    ):
        """
        Initialize Zigzag indicator

        Args:
            atr_multiplier: Multiplier for ATR to determine reversal threshold
            atr_period: Period for ATR calculation
            min_wave_pct: Minimum wave size as percentage
        """
        self.atr_multiplier = atr_multiplier
        self.atr_period = atr_period
        self.min_wave_pct = min_wave_pct

    def find_pivots(self, df: pd.DataFrame) -> List[Pivot]:
        """
        Find all pivot points (swing highs and lows) in the data

        Args:
            df: DataFrame with OHLC data

        Returns:
            List of Pivot objects
        """
        if len(df) < self.atr_period + 1:
            return []

        # Calculate ATR
        atr = calculate_atr(df, self.atr_period)

        pivots = []
        direction = None  # None = unknown, 1 = up, -1 = down

        # Initialize with first bar
        current_high_idx = 0
        current_high = df.iloc[0]['high']
        current_low_idx = 0
        current_low = df.iloc[0]['low']

        for i in range(1, len(df)):
            high = df.iloc[i]['high']
            low = df.iloc[i]['low']
            threshold = atr.iloc[i] * self.atr_multiplier if i < len(atr) else atr.iloc[-1] * self.atr_multiplier

            # Minimum threshold based on percentage
            min_threshold = df.iloc[i]['close'] * (self.min_wave_pct / 100)
            threshold = max(threshold, min_threshold)

            if direction is None:
                # Determine initial direction
                if high > current_high:
                    current_high = high
                    current_high_idx = i
                if low < current_low:
                    current_low = low
                    current_low_idx = i

                # Check if we have enough movement to establish direction
                if current_high - current_low >= threshold:
                    if current_high_idx > current_low_idx:
                        # Low came first, so uptrend
                        direction = 1
                        pivots.append(Pivot(
                            index=current_low_idx,
                            price=current_low,
                            pivot_type=PivotType.LOW,
                            timestamp=df.iloc[current_low_idx]['timestamp']
                        ))
                        current_high = high
                        current_high_idx = i
                    else:
                        # High came first, so downtrend
                        direction = -1
                        pivots.append(Pivot(
                            index=current_high_idx,
                            price=current_high,
                            pivot_type=PivotType.HIGH,
                            timestamp=df.iloc[current_high_idx]['timestamp']
                        ))
                        current_low = low
                        current_low_idx = i

            elif direction == 1:  # In uptrend, looking for higher high or reversal
                if high > current_high:
                    current_high = high
                    current_high_idx = i
                elif current_high - low >= threshold:
                    # Reversal down - mark high and switch direction
                    pivots.append(Pivot(
                        index=current_high_idx,
                        price=current_high,
                        pivot_type=PivotType.HIGH,
                        timestamp=df.iloc[current_high_idx]['timestamp']
                    ))
                    direction = -1
                    current_low = low
                    current_low_idx = i

            else:  # direction == -1, in downtrend
                if low < current_low:
                    current_low = low
                    current_low_idx = i
                elif high - current_low >= threshold:
                    # Reversal up - mark low and switch direction
                    pivots.append(Pivot(
                        index=current_low_idx,
                        price=current_low,
                        pivot_type=PivotType.LOW,
                        timestamp=df.iloc[current_low_idx]['timestamp']
                    ))
                    direction = 1
                    current_high = high
                    current_high_idx = i

        # Add final pivot if we have one
        if direction == 1 and current_high_idx > (pivots[-1].index if pivots else -1):
            pivots.append(Pivot(
                index=current_high_idx,
                price=current_high,
                pivot_type=PivotType.HIGH,
                timestamp=df.iloc[current_high_idx]['timestamp']
            ))
        elif direction == -1 and current_low_idx > (pivots[-1].index if pivots else -1):
            pivots.append(Pivot(
                index=current_low_idx,
                price=current_low,
                pivot_type=PivotType.LOW,
                timestamp=df.iloc[current_low_idx]['timestamp']
            ))

        return self._validate_pivots(pivots, df)

    def _validate_pivots(self, pivots: List[Pivot], df: pd.DataFrame) -> List[Pivot]:
        """
        Validate and filter pivots based on minimum wave size

        Args:
            pivots: List of pivot points
            df: Original DataFrame

        Returns:
            Filtered list of valid pivots
        """
        if len(pivots) < 2:
            return pivots

        valid_pivots = [pivots[0]]

        for i in range(1, len(pivots)):
            prev = valid_pivots[-1]
            curr = pivots[i]

            # Calculate wave size as percentage
            wave_size_pct = abs(curr.price - prev.price) / prev.price * 100

            if wave_size_pct >= self.min_wave_pct:
                valid_pivots.append(curr)
            else:
                # Skip this pivot, but update the last valid pivot if needed
                if curr.pivot_type == prev.pivot_type:
                    # Same type - update to more extreme
                    if curr.pivot_type == PivotType.HIGH and curr.price > prev.price:
                        valid_pivots[-1] = curr
                    elif curr.pivot_type == PivotType.LOW and curr.price < prev.price:
                        valid_pivots[-1] = curr

        return valid_pivots

    def get_pivots_as_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get pivots as a DataFrame for easier analysis

        Args:
            df: Original OHLC DataFrame

        Returns:
            DataFrame with pivot information
        """
        pivots = self.find_pivots(df)

        if not pivots:
            return pd.DataFrame(columns=['index', 'price', 'type', 'timestamp'])

        return pd.DataFrame([
            {
                'index': p.index,
                'price': p.price,
                'type': p.pivot_type.value,
                'timestamp': p.timestamp
            }
            for p in pivots
        ])

    def update_pivots(
        self,
        existing_pivots: List[Pivot],
        df: pd.DataFrame,
        last_processed_idx: int
    ) -> Tuple[List[Pivot], int]:
        """
        Incrementally update pivots with new data

        Args:
            existing_pivots: Previously identified pivots
            df: Full DataFrame including new data
            last_processed_idx: Last index that was processed

        Returns:
            Updated list of pivots and new last processed index
        """
        # For simplicity, recalculate from near the last pivot
        if existing_pivots and len(existing_pivots) >= 2:
            start_idx = max(0, existing_pivots[-2].index - 10)
            subset = df.iloc[start_idx:]
            new_pivots = self.find_pivots(subset)

            # Adjust indices
            for p in new_pivots:
                p.index += start_idx

            # Merge with existing
            cutoff_idx = existing_pivots[-2].index
            kept_pivots = [p for p in existing_pivots if p.index < cutoff_idx]
            return kept_pivots + new_pivots, len(df) - 1
        else:
            return self.find_pivots(df), len(df) - 1
