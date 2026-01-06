"""
Average True Range (ATR) indicator
"""

import pandas as pd
import numpy as np
from typing import Union


def calculate_atr(
    df: pd.DataFrame,
    period: int = 20,
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close'
) -> pd.Series:
    """
    Calculate Average True Range

    Args:
        df: DataFrame with OHLC data
        period: ATR period (default 20)
        high_col: Name of high column
        low_col: Name of low column
        close_col: Name of close column

    Returns:
        Series with ATR values
    """
    high = df[high_col]
    low = df[low_col]
    close = df[close_col]

    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Calculate ATR using exponential moving average
    atr = true_range.ewm(span=period, adjust=False).mean()

    return atr


def calculate_atr_at_index(
    df: pd.DataFrame,
    index: int,
    period: int = 20
) -> float:
    """
    Calculate ATR at a specific index

    Args:
        df: DataFrame with OHLC data
        index: Index to calculate ATR at
        period: ATR period

    Returns:
        ATR value at index
    """
    if index < period:
        # Not enough data, use available
        subset = df.iloc[:index + 1]
    else:
        subset = df.iloc[index - period + 1:index + 1]

    if len(subset) < 2:
        return df.iloc[index]['high'] - df.iloc[index]['low']

    atr = calculate_atr(subset, period=min(period, len(subset)))
    return atr.iloc[-1] if len(atr) > 0 else 0.0


def get_atr_series(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Get ATR as a series aligned with input DataFrame

    Args:
        df: DataFrame with OHLC data
        period: ATR period

    Returns:
        Series with ATR values
    """
    atr = calculate_atr(df, period)
    return atr
