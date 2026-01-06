"""
Fibonacci level statistics analysis
"""

from typing import List, Dict
import numpy as np
from collections import defaultdict


class FibonacciStatistics:
    """Analyzes Fibonacci level hit rates and effectiveness"""

    @staticmethod
    def analyze_entry_levels(trades: List) -> Dict:
        """
        Analyze which Fibonacci entry levels perform best

        Args:
            trades: List of trades with fib_level_hit

        Returns:
            Statistics by entry Fib level
        """
        by_level = defaultdict(lambda: {'count': 0, 'wins': 0, 'pnl': 0, 'pnls': []})

        for t in trades:
            level = t.fib_level_hit if hasattr(t, 'fib_level_hit') else None
            if level is None:
                level = 'unknown'
            else:
                # Round to nearest standard level
                level = _round_to_fib(level)

            by_level[level]['count'] += 1
            if t.realized_pnl > 0:
                by_level[level]['wins'] += 1
            by_level[level]['pnl'] += t.realized_pnl
            by_level[level]['pnls'].append(t.realized_pnl)

        # Calculate statistics
        results = {}
        for level, data in by_level.items():
            count = data['count']
            results[level] = {
                'count': count,
                'wins': data['wins'],
                'win_rate': data['wins'] / count if count > 0 else 0,
                'total_pnl': data['pnl'],
                'avg_pnl': data['pnl'] / count if count > 0 else 0,
                'std_pnl': np.std(data['pnls']) if data['pnls'] else 0
            }

        return results

    @staticmethod
    def analyze_target_hit_rates(trades: List) -> Dict:
        """
        Analyze how often various TP levels are hit

        Args:
            trades: List of trades

        Returns:
            Target hit rate statistics
        """
        total = len(trades)
        if total == 0:
            return {'error': 'No trades'}

        tp1_hits = sum(1 for t in trades if t.tp1_hit)
        tp2_hits = sum(1 for t in trades if t.exit_reason == 'TP2')
        stop_hits = sum(1 for t in trades if t.exit_reason == 'Stop loss')
        breakeven_hits = sum(1 for t in trades if t.exit_reason == 'Breakeven')

        return {
            'total_trades': total,
            'tp1_hit_rate': tp1_hits / total,
            'tp2_hit_rate': tp2_hits / total,
            'stop_hit_rate': stop_hits / total,
            'breakeven_rate': breakeven_hits / total,
            'tp1_hits': tp1_hits,
            'tp2_hits': tp2_hits,
            'stop_hits': stop_hits,
            'breakeven_hits': breakeven_hits
        }

    @staticmethod
    def analyze_extension_accuracy(trades: List) -> Dict:
        """
        Analyze how accurately price reaches Fibonacci extensions

        Args:
            trades: List of trades

        Returns:
            Extension accuracy statistics
        """
        # This would require actual price data to fully implement
        # For now, use exit information as proxy

        wave3_trades = [t for t in trades if t.module == 'A']
        wave5_trades = [t for t in trades if t.module == 'B']

        results = {
            'wave3_extension_stats': {},
            'wave5_extension_stats': {}
        }

        if wave3_trades:
            # Wave 3 should typically reach 161.8%
            tp1_rate = sum(1 for t in wave3_trades if t.tp1_hit) / len(wave3_trades)
            tp2_rate = sum(1 for t in wave3_trades if t.exit_reason == 'TP2') / len(wave3_trades)

            results['wave3_extension_stats'] = {
                'total': len(wave3_trades),
                'reached_100pct': tp1_rate,  # TP1 is usually 100%
                'reached_161pct': tp2_rate,  # TP2 is usually 161.8%
            }

        if wave5_trades:
            tp1_rate = sum(1 for t in wave5_trades if t.tp1_hit) / len(wave5_trades)
            tp2_rate = sum(1 for t in wave5_trades if t.exit_reason == 'TP2') / len(wave5_trades)

            results['wave5_extension_stats'] = {
                'total': len(wave5_trades),
                'reached_tp1': tp1_rate,
                'reached_tp2': tp2_rate
            }

        return results


def _round_to_fib(value: float) -> str:
    """Round a value to nearest standard Fib level label"""
    standard_levels = [
        (0.236, '23.6%'),
        (0.382, '38.2%'),
        (0.5, '50%'),
        (0.618, '61.8%'),
        (0.786, '78.6%'),
        (1.0, '100%')
    ]

    closest = min(standard_levels, key=lambda x: abs(x[0] - value))
    return closest[1]
