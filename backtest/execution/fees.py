"""
Fee calculation for backtesting
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class FeeStructure:
    """Fee structure for trading"""
    maker_fee_pct: float = 0.02  # 0.02%
    taker_fee_pct: float = 0.04  # 0.04%


class FeeCalculator:
    """Calculates trading fees"""

    def __init__(
        self,
        maker_fee_pct: float = 0.02,
        taker_fee_pct: float = 0.04
    ):
        """
        Initialize fee calculator

        Args:
            maker_fee_pct: Maker fee percentage (e.g., 0.02 for 0.02%)
            taker_fee_pct: Taker fee percentage (e.g., 0.04 for 0.04%)
        """
        self.maker_fee_pct = maker_fee_pct / 100  # Convert to decimal
        self.taker_fee_pct = taker_fee_pct / 100

    def calculate_entry_fee(
        self,
        quantity: float,
        price: float,
        is_market_order: bool = False
    ) -> float:
        """
        Calculate fee for entry order

        Args:
            quantity: Position size
            price: Entry price
            is_market_order: True for market (taker), False for limit (maker)

        Returns:
            Fee amount
        """
        notional = quantity * price
        fee_rate = self.taker_fee_pct if is_market_order else self.maker_fee_pct
        return notional * fee_rate

    def calculate_exit_fee(
        self,
        quantity: float,
        price: float,
        is_stop_loss: bool = False
    ) -> float:
        """
        Calculate fee for exit order

        Args:
            quantity: Position size
            price: Exit price
            is_stop_loss: True for stop loss (taker), False for take profit (maker)

        Returns:
            Fee amount
        """
        notional = quantity * price
        fee_rate = self.taker_fee_pct if is_stop_loss else self.maker_fee_pct
        return notional * fee_rate

    def calculate_round_trip_fees(
        self,
        quantity: float,
        entry_price: float,
        exit_price: float,
        entry_is_market: bool = False,
        exit_is_stop: bool = False
    ) -> float:
        """
        Calculate total fees for a round trip trade

        Args:
            quantity: Position size
            entry_price: Entry price
            exit_price: Exit price
            entry_is_market: Entry order type
            exit_is_stop: Whether exit was stop loss

        Returns:
            Total fees
        """
        entry_fee = self.calculate_entry_fee(quantity, entry_price, entry_is_market)
        exit_fee = self.calculate_exit_fee(quantity, exit_price, exit_is_stop)
        return entry_fee + exit_fee

    def estimate_breakeven_move(
        self,
        entry_price: float,
        is_long: bool = True
    ) -> float:
        """
        Estimate price move needed to break even after fees

        Args:
            entry_price: Entry price
            is_long: Long or short position

        Returns:
            Breakeven price
        """
        # Assume maker entry, taker exit (worst case for stop)
        total_fee_pct = self.maker_fee_pct + self.taker_fee_pct

        if is_long:
            return entry_price * (1 + total_fee_pct)
        else:
            return entry_price * (1 - total_fee_pct)
