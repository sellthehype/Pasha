"""
Trade statistics analysis
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class TradeStatistics:
    """Comprehensive trade statistics"""

    @staticmethod
    def analyze_trades(trades: List) -> Dict:
        """
        Analyze list of trades

        Args:
            trades: List of Position objects

        Returns:
            Dictionary with trade statistics
        """
        if not trades:
            return {'error': 'No trades to analyze'}

        # Basic counts
        total = len(trades)
        winners = [t for t in trades if t.realized_pnl > 0]
        losers = [t for t in trades if t.realized_pnl <= 0]

        # P&L statistics
        pnls = [t.realized_pnl for t in trades]
        winning_pnls = [t.realized_pnl for t in winners]
        losing_pnls = [t.realized_pnl for t in losers]

        # Duration statistics
        durations = []
        for t in trades:
            if t.exit_bar_index and t.entry_bar_index:
                durations.append(t.exit_bar_index - t.entry_bar_index)

        stats = {
            'total_trades': total,
            'winning_trades': len(winners),
            'losing_trades': len(losers),
            'win_rate': len(winners) / total if total > 0 else 0,

            # P&L
            'total_pnl': sum(pnls),
            'gross_profit': sum(winning_pnls) if winning_pnls else 0,
            'gross_loss': sum(losing_pnls) if losing_pnls else 0,
            'average_pnl': np.mean(pnls) if pnls else 0,
            'median_pnl': np.median(pnls) if pnls else 0,
            'std_pnl': np.std(pnls) if pnls else 0,

            # Winners
            'average_win': np.mean(winning_pnls) if winning_pnls else 0,
            'largest_win': max(winning_pnls) if winning_pnls else 0,
            'smallest_win': min(winning_pnls) if winning_pnls else 0,

            # Losers
            'average_loss': np.mean(losing_pnls) if losing_pnls else 0,
            'largest_loss': min(losing_pnls) if losing_pnls else 0,
            'smallest_loss': max(losing_pnls) if losing_pnls else 0,

            # Ratios
            'profit_factor': abs(sum(winning_pnls) / sum(losing_pnls)) if losing_pnls and sum(losing_pnls) != 0 else float('inf'),
            'payoff_ratio': abs(np.mean(winning_pnls) / np.mean(losing_pnls)) if losing_pnls and np.mean(losing_pnls) != 0 else float('inf'),

            # Duration
            'avg_duration': np.mean(durations) if durations else 0,
            'max_duration': max(durations) if durations else 0,
            'min_duration': min(durations) if durations else 0,

            # Fees
            'total_fees': sum(t.total_fees for t in trades),
        }

        # By exit reason
        exit_reasons = {}
        for t in trades:
            reason = t.exit_reason or 'Unknown'
            if reason not in exit_reasons:
                exit_reasons[reason] = {'count': 0, 'pnl': 0}
            exit_reasons[reason]['count'] += 1
            exit_reasons[reason]['pnl'] += t.realized_pnl

        stats['by_exit_reason'] = exit_reasons

        # By module
        by_module = {}
        for t in trades:
            module = t.module
            if module not in by_module:
                by_module[module] = {'count': 0, 'wins': 0, 'pnl': 0}
            by_module[module]['count'] += 1
            if t.realized_pnl > 0:
                by_module[module]['wins'] += 1
            by_module[module]['pnl'] += t.realized_pnl

        for module in by_module:
            by_module[module]['win_rate'] = by_module[module]['wins'] / by_module[module]['count']

        stats['by_module'] = by_module

        return stats

    @staticmethod
    def get_trades_df(trades: List) -> pd.DataFrame:
        """Convert trades to DataFrame"""
        if not trades:
            return pd.DataFrame()

        rows = []
        for t in trades:
            rows.append({
                'id': t.id,
                'symbol': t.symbol,
                'side': t.side.value,
                'module': t.module,
                'entry_price': t.entry_price,
                'entry_time': t.entry_timestamp,
                'exit_time': t.exit_timestamp,
                'exit_reason': t.exit_reason,
                'quantity': t.initial_quantity,
                'realized_pnl': t.realized_pnl,
                'fees': t.total_fees,
                'net_pnl': t.realized_pnl - t.total_fees,
                'duration_bars': t.exit_bar_index - t.entry_bar_index if t.exit_bar_index else None,
                'tp1_hit': t.tp1_hit,
                'stop_loss': t.original_stop_loss,
                'take_profit_1': t.take_profit_1,
                'take_profit_2': t.take_profit_2
            })

        return pd.DataFrame(rows)

    @staticmethod
    def calculate_streaks(trades: List) -> Dict:
        """Calculate winning/losing streaks"""
        if not trades:
            return {}

        results = [1 if t.realized_pnl > 0 else 0 for t in trades]

        max_win_streak = 0
        max_loss_streak = 0
        current_win = 0
        current_loss = 0

        for r in results:
            if r == 1:
                current_win += 1
                current_loss = 0
                max_win_streak = max(max_win_streak, current_win)
            else:
                current_loss += 1
                current_win = 0
                max_loss_streak = max(max_loss_streak, current_loss)

        return {
            'max_winning_streak': max_win_streak,
            'max_losing_streak': max_loss_streak,
            'current_streak': current_win if results[-1] == 1 else -current_loss
        }
