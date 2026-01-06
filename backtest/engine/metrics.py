"""
Performance metrics calculation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class PerformanceMetrics:
    """Complete performance metrics"""
    # Returns
    total_return: float = 0.0
    total_return_pct: float = 0.0
    cagr: float = 0.0

    # Risk-adjusted
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Drawdown
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_drawdown: float = 0.0
    max_drawdown_duration: int = 0  # in bars

    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    expectancy: float = 0.0
    avg_trade_duration: float = 0.0

    # By module
    module_metrics: Dict[str, Dict] = field(default_factory=dict)

    # Additional
    recovery_factor: float = 0.0
    risk_reward_ratio: float = 0.0


class MetricsCalculator:
    """Calculates performance metrics from backtest results"""

    def __init__(self, risk_free_rate: float = 0.0):
        """
        Initialize calculator

        Args:
            risk_free_rate: Annual risk-free rate (default 0)
        """
        self.risk_free_rate = risk_free_rate

    def calculate_all(
        self,
        equity_curve: pd.DataFrame,
        trades: List,
        initial_balance: float
    ) -> PerformanceMetrics:
        """
        Calculate all performance metrics

        Args:
            equity_curve: DataFrame with equity over time
            trades: List of closed trades (Position objects)
            initial_balance: Starting balance

        Returns:
            PerformanceMetrics object
        """
        metrics = PerformanceMetrics()

        if equity_curve.empty:
            return metrics

        # Returns
        final_equity = equity_curve['equity'].iloc[-1]
        metrics.total_return = final_equity - initial_balance
        metrics.total_return_pct = (metrics.total_return / initial_balance) * 100

        # Calculate time period
        if 'timestamp' in equity_curve.columns:
            start_date = equity_curve['timestamp'].iloc[0]
            end_date = equity_curve['timestamp'].iloc[-1]
            years = (end_date - start_date).days / 365.25
            if years > 0:
                metrics.cagr = ((final_equity / initial_balance) ** (1 / years) - 1) * 100

        # Drawdown metrics
        metrics.max_drawdown = equity_curve['drawdown'].max()
        metrics.max_drawdown_pct = equity_curve['drawdown_pct'].max()
        metrics.avg_drawdown = equity_curve['drawdown'].mean()

        # Calculate max drawdown duration
        in_drawdown = equity_curve['drawdown'] > 0
        if in_drawdown.any():
            drawdown_groups = (in_drawdown != in_drawdown.shift()).cumsum()
            drawdown_lengths = in_drawdown.groupby(drawdown_groups).sum()
            metrics.max_drawdown_duration = int(drawdown_lengths.max())

        # Risk-adjusted returns
        returns = equity_curve['equity'].pct_change().dropna()
        if len(returns) > 1:
            metrics.sharpe_ratio = self._calculate_sharpe(returns)
            metrics.sortino_ratio = self._calculate_sortino(returns)

        if metrics.max_drawdown_pct > 0:
            metrics.calmar_ratio = metrics.cagr / metrics.max_drawdown_pct

        # Trade statistics
        if trades:
            metrics = self._calculate_trade_metrics(metrics, trades)

        # Recovery factor
        if metrics.max_drawdown > 0:
            metrics.recovery_factor = metrics.total_return / metrics.max_drawdown

        # Risk/Reward
        if metrics.average_loss != 0:
            metrics.risk_reward_ratio = abs(metrics.average_win / metrics.average_loss)

        # Module-specific metrics
        metrics.module_metrics = self._calculate_module_metrics(trades)

        return metrics

    def _calculate_sharpe(self, returns: pd.Series, periods: int = 252) -> float:
        """Calculate annualized Sharpe ratio"""
        if returns.std() == 0:
            return 0.0

        excess_returns = returns - (self.risk_free_rate / periods)
        return np.sqrt(periods) * (excess_returns.mean() / returns.std())

    def _calculate_sortino(self, returns: pd.Series, periods: int = 252) -> float:
        """Calculate annualized Sortino ratio"""
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0

        excess_returns = returns.mean() - (self.risk_free_rate / periods)
        return np.sqrt(periods) * (excess_returns / downside_returns.std())

    def _calculate_trade_metrics(
        self,
        metrics: PerformanceMetrics,
        trades: List
    ) -> PerformanceMetrics:
        """Calculate trade-based metrics"""
        metrics.total_trades = len(trades)

        wins = []
        losses = []
        durations = []

        for trade in trades:
            pnl = trade.realized_pnl
            if pnl > 0:
                metrics.winning_trades += 1
                wins.append(pnl)
            else:
                metrics.losing_trades += 1
                losses.append(pnl)

            # Duration
            if trade.exit_bar_index and trade.entry_bar_index:
                durations.append(trade.exit_bar_index - trade.entry_bar_index)

        if metrics.total_trades > 0:
            metrics.win_rate = metrics.winning_trades / metrics.total_trades

        if wins:
            metrics.average_win = sum(wins) / len(wins)
            metrics.largest_win = max(wins)
            metrics.gross_profit = sum(wins)

        if losses:
            metrics.average_loss = sum(losses) / len(losses)
            metrics.largest_loss = min(losses)
            metrics.gross_loss = sum(losses)

        if metrics.gross_loss != 0:
            metrics.profit_factor = abs(metrics.gross_profit / metrics.gross_loss)

        metrics.expectancy = (
            (metrics.win_rate * metrics.average_win) +
            ((1 - metrics.win_rate) * metrics.average_loss)
        )

        if durations:
            metrics.avg_trade_duration = sum(durations) / len(durations)

        return metrics

    def _calculate_module_metrics(self, trades: List) -> Dict[str, Dict]:
        """Calculate metrics by module"""
        module_trades = {'A': [], 'B': [], 'C': []}

        for trade in trades:
            module = trade.module
            if module in module_trades:
                module_trades[module].append(trade)

        results = {}
        for module, mtrades in module_trades.items():
            if not mtrades:
                results[module] = {
                    'total_trades': 0,
                    'win_rate': 0,
                    'profit_factor': 0,
                    'total_pnl': 0
                }
                continue

            wins = [t for t in mtrades if t.realized_pnl > 0]
            losses = [t for t in mtrades if t.realized_pnl <= 0]

            gross_profit = sum(t.realized_pnl for t in wins)
            gross_loss = sum(t.realized_pnl for t in losses)

            results[module] = {
                'total_trades': len(mtrades),
                'winning_trades': len(wins),
                'losing_trades': len(losses),
                'win_rate': len(wins) / len(mtrades) if mtrades else 0,
                'profit_factor': abs(gross_profit / gross_loss) if gross_loss != 0 else float('inf'),
                'total_pnl': sum(t.realized_pnl for t in mtrades),
                'average_pnl': sum(t.realized_pnl for t in mtrades) / len(mtrades)
            }

        return results


def calculate_metrics_by_timeframe(
    results: Dict[str, Dict]
) -> pd.DataFrame:
    """
    Create metrics comparison by timeframe

    Args:
        results: Dict mapping timeframe to backtest results

    Returns:
        DataFrame with metrics comparison
    """
    rows = []
    for tf, result in results.items():
        if 'metrics' in result:
            m = result['metrics']
            rows.append({
                'timeframe': tf,
                'total_return_pct': m.total_return_pct,
                'sharpe_ratio': m.sharpe_ratio,
                'sortino_ratio': m.sortino_ratio,
                'max_drawdown_pct': m.max_drawdown_pct,
                'total_trades': m.total_trades,
                'win_rate': m.win_rate,
                'profit_factor': m.profit_factor,
                'expectancy': m.expectancy
            })

    return pd.DataFrame(rows)
