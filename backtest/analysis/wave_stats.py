"""
Wave-specific statistics analysis
"""

from typing import List, Dict
import numpy as np
from collections import defaultdict


class WaveStatistics:
    """Analyzes wave-specific behavior"""

    @staticmethod
    def analyze_wave3_extensions(trades: List) -> Dict:
        """
        Analyze how often Wave 3 reaches various extension levels

        Args:
            trades: List of trades with wave context

        Returns:
            Statistics about Wave 3 extensions
        """
        wave3_trades = [t for t in trades if t.module == 'A']

        if not wave3_trades:
            return {'error': 'No Wave 3 trades'}

        # Track target hits
        tp1_hits = sum(1 for t in wave3_trades if t.tp1_hit)
        tp2_hits = sum(1 for t in wave3_trades if t.exit_reason == 'TP2')

        total = len(wave3_trades)

        return {
            'total_wave3_trades': total,
            'tp1_hit_rate': tp1_hits / total if total > 0 else 0,
            'tp2_hit_rate': tp2_hits / total if total > 0 else 0,
            'tp1_hits': tp1_hits,
            'tp2_hits': tp2_hits,
            'avg_pnl': np.mean([t.realized_pnl for t in wave3_trades]) if wave3_trades else 0
        }

    @staticmethod
    def analyze_wave5_truncations(trades: List) -> Dict:
        """
        Analyze Wave 5 truncation rate

        Args:
            trades: List of trades with wave context

        Returns:
            Statistics about Wave 5 behavior
        """
        wave5_trades = [t for t in trades if t.module == 'B']

        if not wave5_trades:
            return {'error': 'No Wave 5 trades'}

        # A truncation would be a trade that hit stop loss
        # without reaching TP1
        truncations = sum(
            1 for t in wave5_trades
            if t.exit_reason == 'Stop loss' and not t.tp1_hit
        )

        total = len(wave5_trades)

        return {
            'total_wave5_trades': total,
            'truncation_rate': truncations / total if total > 0 else 0,
            'truncations': truncations,
            'successful_trades': total - truncations,
            'avg_pnl': np.mean([t.realized_pnl for t in wave5_trades]) if wave5_trades else 0
        }

    @staticmethod
    def analyze_wave2_retracements(trades: List) -> Dict:
        """
        Analyze Wave 2 retracement depths from wave context

        Args:
            trades: List of trades

        Returns:
            Distribution of Wave 2 retracements
        """
        retracements = []

        for t in trades:
            if t.module == 'A' and 'wave_context' in dir(t):
                ctx = t.signal_context
                if 'wave1_start' in ctx and 'wave1_end' in ctx and 'wave2_end' in ctx:
                    w1_range = abs(ctx['wave1_end'] - ctx['wave1_start'])
                    w2_range = abs(ctx['wave2_end'] - ctx['wave1_end'])
                    if w1_range > 0:
                        retracements.append(w2_range / w1_range)

        if not retracements:
            return {'error': 'No Wave 2 data'}

        return {
            'count': len(retracements),
            'mean_retracement': np.mean(retracements),
            'median_retracement': np.median(retracements),
            'std_retracement': np.std(retracements),
            'min_retracement': min(retracements),
            'max_retracement': max(retracements),
            'pct_at_618': sum(1 for r in retracements if 0.58 <= r <= 0.66) / len(retracements),
            'pct_at_50': sum(1 for r in retracements if 0.47 <= r <= 0.53) / len(retracements),
            'pct_at_786': sum(1 for r in retracements if 0.75 <= r <= 0.82) / len(retracements)
        }

    @staticmethod
    def analyze_corrective_patterns(trades: List) -> Dict:
        """
        Analyze corrective pattern performance

        Args:
            trades: List of trades

        Returns:
            Statistics by corrective pattern type
        """
        corrective_trades = [t for t in trades if t.module == 'C']

        if not corrective_trades:
            return {'error': 'No corrective trades'}

        by_pattern = defaultdict(lambda: {'count': 0, 'wins': 0, 'pnl': 0})

        for t in corrective_trades:
            pattern_type = t.signal_context.get('pattern_type', 'unknown')
            by_pattern[pattern_type]['count'] += 1
            if t.realized_pnl > 0:
                by_pattern[pattern_type]['wins'] += 1
            by_pattern[pattern_type]['pnl'] += t.realized_pnl

        # Calculate win rates
        for pattern in by_pattern:
            count = by_pattern[pattern]['count']
            by_pattern[pattern]['win_rate'] = by_pattern[pattern]['wins'] / count if count > 0 else 0
            by_pattern[pattern]['avg_pnl'] = by_pattern[pattern]['pnl'] / count if count > 0 else 0

        return dict(by_pattern)
