"""
Order management for backtesting
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum
from datetime import datetime


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Represents a trading order"""
    id: str
    symbol: str
    order_type: OrderType
    side: OrderSide
    quantity: float
    price: Optional[float] = None  # For limit orders
    stop_price: Optional[float] = None  # For stop orders

    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    fee: float = 0.0

    created_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None

    position_id: Optional[str] = None  # Link to position
    is_reduce_only: bool = False  # For closing positions

    def fill(
        self,
        price: float,
        quantity: float,
        fee: float,
        timestamp: datetime
    ):
        """Fill the order"""
        self.filled_price = price
        self.filled_quantity = quantity
        self.fee = fee
        self.filled_at = timestamp

        if quantity >= self.quantity:
            self.status = OrderStatus.FILLED
        else:
            self.status = OrderStatus.PARTIALLY_FILLED

    def cancel(self):
        """Cancel the order"""
        self.status = OrderStatus.CANCELLED


class OrderManager:
    """Manages orders in backtest"""

    def __init__(self):
        self.pending_orders = []
        self.filled_orders = []
        self.cancelled_orders = []

    def add_order(self, order: Order):
        """Add new order"""
        self.pending_orders.append(order)

    def process_orders(
        self,
        current_bar: dict,
        bar_index: int,
        timestamp: datetime
    ) -> list:
        """
        Process pending orders against current bar

        Returns:
            List of filled orders
        """
        filled = []
        remaining = []

        for order in self.pending_orders:
            if self._should_fill(order, current_bar):
                fill_price = self._get_fill_price(order, current_bar)
                order.fill(
                    price=fill_price,
                    quantity=order.quantity,
                    fee=0,  # Fee calculated separately
                    timestamp=timestamp
                )
                filled.append(order)
                self.filled_orders.append(order)
            else:
                remaining.append(order)

        self.pending_orders = remaining
        return filled

    def _should_fill(self, order: Order, bar: dict) -> bool:
        """Check if order should fill on this bar"""
        high = bar['high']
        low = bar['low']

        if order.order_type == OrderType.MARKET:
            return True

        elif order.order_type == OrderType.LIMIT:
            if order.side == OrderSide.BUY:
                return low <= order.price
            else:
                return high >= order.price

        elif order.order_type == OrderType.STOP:
            if order.side == OrderSide.BUY:
                return high >= order.stop_price
            else:
                return low <= order.stop_price

        return False

    def _get_fill_price(self, order: Order, bar: dict) -> float:
        """Get fill price for order"""
        if order.order_type == OrderType.MARKET:
            return bar['open']  # Fill at open

        elif order.order_type == OrderType.LIMIT:
            return order.price

        elif order.order_type == OrderType.STOP:
            # Stop orders fill at stop price (simplified)
            return order.stop_price

        return bar['close']

    def cancel_orders_for_position(self, position_id: str):
        """Cancel all pending orders for a position"""
        remaining = []
        for order in self.pending_orders:
            if order.position_id == position_id:
                order.cancel()
                self.cancelled_orders.append(order)
            else:
                remaining.append(order)
        self.pending_orders = remaining
