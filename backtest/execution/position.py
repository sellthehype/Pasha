"""
Position management for backtesting
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime
import uuid


class PositionStatus(Enum):
    PENDING = "pending"  # Waiting for confirmation
    OPEN = "open"
    PARTIALLY_CLOSED = "partially_closed"
    CLOSED = "closed"


class PositionSide(Enum):
    LONG = "long"
    SHORT = "short"


@dataclass
class PositionFill:
    """Records a fill (entry or exit) on a position"""
    timestamp: datetime
    bar_index: int
    price: float
    quantity: float
    is_entry: bool
    fee: float = 0.0
    reason: str = ""


@dataclass
class Position:
    """Represents a trading position"""
    id: str
    symbol: str
    side: PositionSide
    module: str  # A, B, or C

    # Entry info
    entry_price: float
    entry_timestamp: datetime
    entry_bar_index: int
    initial_quantity: float

    # Risk management
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    invalidation_price: float

    # Current state
    status: PositionStatus = PositionStatus.OPEN
    current_quantity: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_fees: float = 0.0

    # Fills
    fills: List[PositionFill] = field(default_factory=list)

    # Stage tracking (for probabilistic entry)
    stage: int = 1  # 1 = initial, 2 = confirmed
    awaiting_confirmation: bool = False
    confirmation_quantity: float = 0.0

    # Exit tracking
    tp1_hit: bool = False
    tp1_exit_quantity: float = 0.0
    exit_timestamp: Optional[datetime] = None
    exit_bar_index: Optional[int] = None
    exit_reason: str = ""

    # Stop behavior tracking
    stop_moved_to_breakeven: bool = False
    original_stop_loss: float = 0.0

    # Metadata
    signal_context: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.current_quantity == 0:
            self.current_quantity = self.initial_quantity
        if self.original_stop_loss == 0:
            self.original_stop_loss = self.stop_loss

    @property
    def is_long(self) -> bool:
        return self.side == PositionSide.LONG

    @property
    def is_open(self) -> bool:
        return self.status in [PositionStatus.OPEN, PositionStatus.PARTIALLY_CLOSED]

    @property
    def average_entry_price(self) -> float:
        """Calculate average entry price from fills"""
        entry_fills = [f for f in self.fills if f.is_entry]
        if not entry_fills:
            return self.entry_price

        total_value = sum(f.price * f.quantity for f in entry_fills)
        total_qty = sum(f.quantity for f in entry_fills)
        return total_value / total_qty if total_qty > 0 else self.entry_price

    def update_unrealized_pnl(self, current_price: float):
        """Update unrealized P&L based on current price"""
        if not self.is_open:
            self.unrealized_pnl = 0.0
            return

        if self.is_long:
            self.unrealized_pnl = (current_price - self.average_entry_price) * self.current_quantity
        else:
            self.unrealized_pnl = (self.average_entry_price - current_price) * self.current_quantity

    def record_fill(self, fill: PositionFill):
        """Record a fill on this position"""
        self.fills.append(fill)
        self.total_fees += fill.fee

    def partial_close(
        self,
        quantity: float,
        price: float,
        timestamp: datetime,
        bar_index: int,
        fee: float,
        reason: str = ""
    ) -> float:
        """
        Close part of the position

        Returns:
            Realized P&L from this close
        """
        if quantity > self.current_quantity:
            quantity = self.current_quantity

        # Calculate P&L
        if self.is_long:
            pnl = (price - self.average_entry_price) * quantity
        else:
            pnl = (self.average_entry_price - price) * quantity

        pnl -= fee  # Subtract fees
        self.realized_pnl += pnl
        self.current_quantity -= quantity
        self.total_fees += fee

        # Record fill
        self.record_fill(PositionFill(
            timestamp=timestamp,
            bar_index=bar_index,
            price=price,
            quantity=quantity,
            is_entry=False,
            fee=fee,
            reason=reason
        ))

        # Update status
        if self.current_quantity <= 0:
            self.status = PositionStatus.CLOSED
            self.exit_timestamp = timestamp
            self.exit_bar_index = bar_index
            self.exit_reason = reason
        else:
            self.status = PositionStatus.PARTIALLY_CLOSED

        return pnl

    def full_close(
        self,
        price: float,
        timestamp: datetime,
        bar_index: int,
        fee: float,
        reason: str = ""
    ) -> float:
        """Close entire position"""
        return self.partial_close(
            self.current_quantity, price, timestamp, bar_index, fee, reason
        )

    def move_stop_to_breakeven(self):
        """Move stop loss to breakeven (entry price)"""
        self.stop_loss = self.entry_price
        self.stop_moved_to_breakeven = True


class PositionManager:
    """Manages multiple positions"""

    def __init__(self, max_positions_per_asset: int = 10):
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        self.max_positions_per_asset = max_positions_per_asset

    def add_position(self, position: Position):
        """Add a new position"""
        self.positions[position.id] = position

    def get_position(self, position_id: str) -> Optional[Position]:
        """Get position by ID"""
        return self.positions.get(position_id)

    def get_open_positions(self) -> List[Position]:
        """Get all open positions"""
        return [p for p in self.positions.values() if p.is_open]

    def get_positions_for_asset(self, symbol: str) -> List[Position]:
        """Get open positions for specific asset"""
        return [p for p in self.get_open_positions() if p.symbol == symbol]

    def can_open_position(self, symbol: str) -> bool:
        """Check if we can open a new position for this asset"""
        current_count = len(self.get_positions_for_asset(symbol))
        return current_count < self.max_positions_per_asset

    def close_position(self, position_id: str):
        """Move position to closed list"""
        if position_id in self.positions:
            position = self.positions.pop(position_id)
            self.closed_positions.append(position)

    def get_total_exposure(self) -> float:
        """Get total value of open positions"""
        return sum(
            p.current_quantity * p.entry_price
            for p in self.get_open_positions()
        )

    def get_portfolio_heat(self, account_balance: float) -> float:
        """Get current portfolio heat (risk as % of account)"""
        total_risk = 0.0
        for position in self.get_open_positions():
            risk_per_unit = abs(position.entry_price - position.stop_loss)
            position_risk = risk_per_unit * position.current_quantity
            total_risk += position_risk

        return (total_risk / account_balance * 100) if account_balance > 0 else 0.0

    def update_all_pnl(self, prices: Dict[str, float]):
        """Update unrealized P&L for all positions"""
        for position in self.get_open_positions():
            if position.symbol in prices:
                position.update_unrealized_pnl(prices[position.symbol])

    def get_all_trades(self) -> List[Position]:
        """Get all trades (open and closed)"""
        return list(self.positions.values()) + self.closed_positions

    def create_position(
        self,
        symbol: str,
        side: PositionSide,
        module: str,
        entry_price: float,
        entry_timestamp: datetime,
        entry_bar_index: int,
        quantity: float,
        stop_loss: float,
        take_profit_1: float,
        take_profit_2: float,
        invalidation_price: float,
        entry_fee: float = 0.0,
        signal_context: Dict = None
    ) -> Position:
        """Create and add a new position"""
        position = Position(
            id=str(uuid.uuid4()),
            symbol=symbol,
            side=side,
            module=module,
            entry_price=entry_price,
            entry_timestamp=entry_timestamp,
            entry_bar_index=entry_bar_index,
            initial_quantity=quantity,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            invalidation_price=invalidation_price,
            signal_context=signal_context or {}
        )

        # Record entry fill
        position.record_fill(PositionFill(
            timestamp=entry_timestamp,
            bar_index=entry_bar_index,
            price=entry_price,
            quantity=quantity,
            is_entry=True,
            fee=entry_fee,
            reason="Initial entry"
        ))
        position.total_fees = entry_fee

        self.add_position(position)
        return position
