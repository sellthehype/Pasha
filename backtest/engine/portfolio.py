"""
Portfolio tracking for backtesting
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
import pandas as pd


@dataclass
class EquityPoint:
    """Single point on equity curve"""
    timestamp: datetime
    bar_index: int
    equity: float
    cash: float
    unrealized_pnl: float
    realized_pnl: float
    drawdown: float
    drawdown_pct: float


class Portfolio:
    """Tracks portfolio state over time"""

    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.cash = initial_balance
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0

        self.equity_curve: List[EquityPoint] = []
        self.peak_equity = initial_balance
        self.max_drawdown = 0.0
        self.max_drawdown_pct = 0.0

        # Trade tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.gross_profit = 0.0
        self.gross_loss = 0.0

    @property
    def equity(self) -> float:
        """Current equity (cash + unrealized P&L)"""
        return self.cash + self.unrealized_pnl

    @property
    def total_return(self) -> float:
        """Total return as decimal"""
        return (self.equity - self.initial_balance) / self.initial_balance

    @property
    def total_return_pct(self) -> float:
        """Total return as percentage"""
        return self.total_return * 100

    @property
    def win_rate(self) -> float:
        """Win rate as decimal"""
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades

    @property
    def profit_factor(self) -> float:
        """Profit factor (gross profit / gross loss)"""
        if self.gross_loss == 0:
            return float('inf') if self.gross_profit > 0 else 0.0
        return self.gross_profit / abs(self.gross_loss)

    @property
    def average_win(self) -> float:
        """Average winning trade"""
        if self.winning_trades == 0:
            return 0.0
        return self.gross_profit / self.winning_trades

    @property
    def average_loss(self) -> float:
        """Average losing trade"""
        if self.losing_trades == 0:
            return 0.0
        return self.gross_loss / self.losing_trades

    @property
    def expectancy(self) -> float:
        """Expected value per trade"""
        if self.total_trades == 0:
            return 0.0
        return (self.win_rate * self.average_win) + ((1 - self.win_rate) * self.average_loss)

    def record_equity(
        self,
        timestamp: datetime,
        bar_index: int,
        unrealized_pnl: float = 0.0
    ):
        """Record current equity state"""
        self.unrealized_pnl = unrealized_pnl
        current_equity = self.equity

        # Update peak and drawdown
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        drawdown = self.peak_equity - current_equity
        drawdown_pct = (drawdown / self.peak_equity * 100) if self.peak_equity > 0 else 0

        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
        if drawdown_pct > self.max_drawdown_pct:
            self.max_drawdown_pct = drawdown_pct

        point = EquityPoint(
            timestamp=timestamp,
            bar_index=bar_index,
            equity=current_equity,
            cash=self.cash,
            unrealized_pnl=self.unrealized_pnl,
            realized_pnl=self.realized_pnl,
            drawdown=drawdown,
            drawdown_pct=drawdown_pct
        )
        self.equity_curve.append(point)

    def record_trade_result(self, pnl: float, fees: float = 0.0):
        """Record result of a closed trade"""
        net_pnl = pnl - fees
        self.realized_pnl += net_pnl
        self.cash += net_pnl
        self.total_trades += 1

        if net_pnl > 0:
            self.winning_trades += 1
            self.gross_profit += net_pnl
        else:
            self.losing_trades += 1
            self.gross_loss += net_pnl  # Already negative

    def deduct_entry_cost(self, cost: float):
        """Deduct cost when opening position"""
        # For margin trading, we don't deduct full cost
        # Just track fees
        pass

    def get_equity_df(self) -> pd.DataFrame:
        """Get equity curve as DataFrame"""
        if not self.equity_curve:
            return pd.DataFrame()

        return pd.DataFrame([
            {
                'timestamp': p.timestamp,
                'bar_index': p.bar_index,
                'equity': p.equity,
                'cash': p.cash,
                'unrealized_pnl': p.unrealized_pnl,
                'realized_pnl': p.realized_pnl,
                'drawdown': p.drawdown,
                'drawdown_pct': p.drawdown_pct
            }
            for p in self.equity_curve
        ])

    def get_summary(self) -> Dict:
        """Get portfolio summary"""
        return {
            'initial_balance': self.initial_balance,
            'final_equity': self.equity,
            'total_return_pct': self.total_return_pct,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'average_win': self.average_win,
            'average_loss': self.average_loss,
            'expectancy': self.expectancy,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_pct': self.max_drawdown_pct,
            'gross_profit': self.gross_profit,
            'gross_loss': self.gross_loss,
            'realized_pnl': self.realized_pnl
        }
