"""
Main backtest engine
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from tqdm import tqdm

from ..config.settings import Config
from ..waves.analyzer import WaveAnalyzer
from ..strategy.module_a import ModuleAStrategy
from ..strategy.module_b import ModuleBStrategy
from ..strategy.module_c import ModuleCStrategy
from ..strategy.signals import Signal, SignalGenerator
from ..execution.position import PositionManager, Position, PositionSide, PositionStatus
from ..execution.fees import FeeCalculator
from ..indicators.atr import calculate_atr
from .portfolio import Portfolio
from .metrics import MetricsCalculator, PerformanceMetrics


@dataclass
class BacktestResult:
    """Results from a backtest run"""
    symbol: str
    timeframe: str
    config: Config

    # Core results
    portfolio: Portfolio
    metrics: PerformanceMetrics
    trades: List[Position]

    # Analysis data
    equity_curve: pd.DataFrame
    signals_generated: int
    signals_executed: int

    # Wave statistics
    wave_stats: Dict = None


class BacktestEngine:
    """
    Main backtesting engine that orchestrates all components
    """

    def __init__(self, config: Config):
        """
        Initialize backtest engine

        Args:
            config: Configuration object
        """
        self.config = config

        # Initialize components
        self.wave_analyzer = WaveAnalyzer(
            zigzag_atr_mult=config.zigzag_atr_multiplier,
            zigzag_atr_period=config.zigzag_atr_period,
            min_wave_pct=config.min_wave_size_pct,
            fib_tolerance=config.fib_tolerance_pct
        )

        self.module_a = ModuleAStrategy(
            fib_tolerance_pct=config.fib_tolerance_pct,
            entry_fib_levels=config.entry_fib_levels,
            initial_position_pct=config.initial_position_pct
        )

        self.module_b = ModuleBStrategy(
            fib_tolerance_pct=config.fib_tolerance_pct,
            initial_position_pct=config.initial_position_pct
        )

        self.module_c = ModuleCStrategy(
            fib_tolerance_pct=config.fib_tolerance_pct
        )

        self.signal_generator = SignalGenerator(
            module_a_enabled=config.module_a_enabled,
            module_b_enabled=config.module_b_enabled,
            module_c_enabled=config.module_c_enabled
        )

        self.fee_calc = FeeCalculator(
            maker_fee_pct=config.maker_fee_pct,
            taker_fee_pct=config.taker_fee_pct
        )

        self.metrics_calc = MetricsCalculator()

    def run(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        show_progress: bool = True
    ) -> BacktestResult:
        """
        Run backtest on data

        Args:
            df: DataFrame with OHLCV data
            symbol: Trading symbol
            timeframe: Timeframe string
            show_progress: Show progress bar

        Returns:
            BacktestResult object
        """
        # Initialize portfolio and position manager
        portfolio = Portfolio(initial_balance=self.config.initial_balance)
        position_manager = PositionManager()

        # Calculate ATR for the entire dataset
        atr_series = calculate_atr(df, self.config.zigzag_atr_period)

        # Tracking
        signals_generated = 0
        signals_executed = 0

        # Lookback for wave analysis
        lookback = max(100, self.config.zigzag_atr_period * 3)

        # Progress bar
        iterator = range(lookback, len(df))
        if show_progress:
            iterator = tqdm(iterator, desc=f"Backtesting {symbol} {timeframe}")

        for i in iterator:
            current_bar = df.iloc[i]
            historical = df.iloc[:i + 1]

            # 1. Analyze wave structure
            structure = self.wave_analyzer.analyze(historical)

            # 2. Generate signals from all modules
            signals = []

            if self.config.module_a_enabled:
                signals.extend(self.module_a.generate_signals(df, structure, i))

            if self.config.module_b_enabled:
                signals.extend(self.module_b.generate_signals(df, structure, i))

            if self.config.module_c_enabled:
                signals.extend(self.module_c.generate_signals(df, structure, i))

            # Filter signals
            signals = self.signal_generator.filter_signals(signals)
            signals = self.signal_generator.deduplicate_signals(signals)
            signals_generated += len(signals)

            # 3. Check existing positions for exits
            self._check_position_exits(
                position_manager, current_bar, i, portfolio
            )

            # 4. Check confirmations for pending entries
            self._check_confirmations(
                position_manager, df, i
            )

            # 5. Execute new signals
            for signal in signals:
                if position_manager.can_open_position(symbol):
                    # Check portfolio heat
                    current_heat = position_manager.get_portfolio_heat(portfolio.equity)
                    if current_heat < self.config.max_portfolio_heat_pct:
                        position = self._execute_signal(
                            signal, symbol, portfolio, position_manager,
                            current_bar, i, atr_series.iloc[i]
                        )
                        if position:
                            signals_executed += 1

            # 6. Update unrealized P&L
            unrealized = sum(
                p.unrealized_pnl for p in position_manager.get_open_positions()
            )
            for p in position_manager.get_open_positions():
                p.update_unrealized_pnl(current_bar['close'])

            # 7. Record equity
            portfolio.record_equity(
                timestamp=current_bar['timestamp'],
                bar_index=i,
                unrealized_pnl=unrealized
            )

        # Close any remaining positions at end
        self._close_remaining_positions(
            position_manager, df.iloc[-1], len(df) - 1, portfolio
        )

        # Calculate final metrics
        equity_df = portfolio.get_equity_df()
        all_trades = position_manager.get_all_trades()

        metrics = self.metrics_calc.calculate_all(
            equity_df,
            all_trades,
            self.config.initial_balance
        )

        return BacktestResult(
            symbol=symbol,
            timeframe=timeframe,
            config=self.config,
            portfolio=portfolio,
            metrics=metrics,
            trades=all_trades,
            equity_curve=equity_df,
            signals_generated=signals_generated,
            signals_executed=signals_executed
        )

    def _execute_signal(
        self,
        signal: Signal,
        symbol: str,
        portfolio: Portfolio,
        position_manager: PositionManager,
        current_bar: pd.Series,
        bar_index: int,
        current_atr: float
    ) -> Optional[Position]:
        """Execute a trading signal"""
        # Determine position side
        side = PositionSide.LONG if signal.is_long else PositionSide.SHORT

        # Calculate position size
        risk_amount = portfolio.equity * (self.config.base_risk_pct / 100)

        # Apply module multiplier
        if signal.module == 'A':
            risk_amount *= self.config.module_a_risk_mult
        elif signal.module == 'B':
            risk_amount *= self.config.module_b_risk_mult
        elif signal.module == 'C':
            risk_amount *= self.config.module_c_risk_mult

        # Risk per unit
        risk_per_unit = signal.risk_per_unit
        if risk_per_unit <= 0:
            return None

        # Position size
        quantity = risk_amount / risk_per_unit

        # Initial entry (probabilistic)
        initial_qty = quantity * (self.config.initial_position_pct / 100)

        # Calculate entry fee
        entry_fee = self.fee_calc.calculate_entry_fee(
            initial_qty, signal.entry_price, is_market_order=False
        )

        # Create position
        position = position_manager.create_position(
            symbol=symbol,
            side=side,
            module=signal.module,
            entry_price=signal.entry_price,
            entry_timestamp=signal.timestamp,
            entry_bar_index=bar_index,
            quantity=initial_qty,
            stop_loss=signal.stop_loss,
            take_profit_1=signal.take_profit_1,
            take_profit_2=signal.take_profit_2,
            invalidation_price=signal.stop_loss,
            entry_fee=entry_fee,
            signal_context=signal.wave_context
        )

        # Mark for confirmation entry
        position.awaiting_confirmation = True
        position.confirmation_quantity = quantity - initial_qty

        return position

    def _check_position_exits(
        self,
        position_manager: PositionManager,
        current_bar: pd.Series,
        bar_index: int,
        portfolio: Portfolio
    ):
        """Check all positions for exit conditions"""
        timestamp = current_bar['timestamp']
        high = current_bar['high']
        low = current_bar['low']
        close = current_bar['close']

        for position in position_manager.get_open_positions():
            # Check stop loss
            stop_hit = False
            if position.is_long:
                stop_hit = low <= position.stop_loss
            else:
                stop_hit = high >= position.stop_loss

            if stop_hit:
                exit_price = position.stop_loss
                exit_fee = self.fee_calc.calculate_exit_fee(
                    position.current_quantity, exit_price, is_stop_loss=True
                )
                pnl = position.full_close(
                    exit_price, timestamp, bar_index, exit_fee, "Stop loss"
                )
                portfolio.record_trade_result(pnl, exit_fee)
                position_manager.close_position(position.id)
                continue

            # Check TP1
            if not position.tp1_hit:
                tp1_hit = False
                if position.is_long:
                    tp1_hit = high >= position.take_profit_1
                else:
                    tp1_hit = low <= position.take_profit_1

                if tp1_hit:
                    position.tp1_hit = True
                    exit_qty = position.current_quantity * (self.config.tp1_pct / 100)
                    exit_fee = self.fee_calc.calculate_exit_fee(
                        exit_qty, position.take_profit_1, is_stop_loss=False
                    )
                    pnl = position.partial_close(
                        exit_qty, position.take_profit_1, timestamp,
                        bar_index, exit_fee, "TP1"
                    )
                    portfolio.record_trade_result(pnl, exit_fee)

                    # Move stop to breakeven
                    if self.config.stop_behavior == "breakeven_after_tp1":
                        position.move_stop_to_breakeven()

            # Check TP2 (remaining position)
            elif position.tp1_hit and position.current_quantity > 0:
                tp2_hit = False
                if position.is_long:
                    tp2_hit = high >= position.take_profit_2
                else:
                    tp2_hit = low <= position.take_profit_2

                if tp2_hit:
                    exit_fee = self.fee_calc.calculate_exit_fee(
                        position.current_quantity, position.take_profit_2, is_stop_loss=False
                    )
                    pnl = position.full_close(
                        position.take_profit_2, timestamp, bar_index, exit_fee, "TP2"
                    )
                    portfolio.record_trade_result(pnl, exit_fee)
                    position_manager.close_position(position.id)

            # Check breakeven exit after TP1
            if position.tp1_hit and position.stop_moved_to_breakeven:
                be_hit = False
                if position.is_long:
                    be_hit = low <= position.entry_price
                else:
                    be_hit = high >= position.entry_price

                if be_hit:
                    exit_fee = self.fee_calc.calculate_exit_fee(
                        position.current_quantity, position.entry_price, is_stop_loss=False
                    )
                    pnl = position.full_close(
                        position.entry_price, timestamp, bar_index, exit_fee, "Breakeven"
                    )
                    portfolio.record_trade_result(pnl, exit_fee)
                    position_manager.close_position(position.id)

    def _check_confirmations(
        self,
        position_manager: PositionManager,
        df: pd.DataFrame,
        current_index: int
    ):
        """Check for confirmation entries on pending positions"""
        for position in position_manager.get_open_positions():
            if not position.awaiting_confirmation:
                continue

            if position.confirmation_quantity <= 0:
                position.awaiting_confirmation = False
                continue

            # Check if we got higher low confirmation
            # For now, use simplified check - just add after a few bars
            bars_since_entry = current_index - position.entry_bar_index
            if bars_since_entry >= 3:
                # Add confirmation quantity
                current_bar = df.iloc[current_index]

                # Use current price for confirmation entry
                conf_price = current_bar['close']
                conf_fee = self.fee_calc.calculate_entry_fee(
                    position.confirmation_quantity, conf_price, is_market_order=False
                )

                position.record_fill(
                    PositionFill(
                        timestamp=current_bar['timestamp'],
                        bar_index=current_index,
                        price=conf_price,
                        quantity=position.confirmation_quantity,
                        is_entry=True,
                        fee=conf_fee,
                        reason="Confirmation entry"
                    )
                )

                position.current_quantity += position.confirmation_quantity
                position.total_fees += conf_fee
                position.stage = 2
                position.awaiting_confirmation = False
                position.confirmation_quantity = 0

    def _close_remaining_positions(
        self,
        position_manager: PositionManager,
        last_bar: pd.Series,
        bar_index: int,
        portfolio: Portfolio
    ):
        """Close any positions remaining at end of backtest"""
        timestamp = last_bar['timestamp']
        close_price = last_bar['close']

        for position in position_manager.get_open_positions():
            exit_fee = self.fee_calc.calculate_exit_fee(
                position.current_quantity, close_price, is_stop_loss=False
            )
            pnl = position.full_close(
                close_price, timestamp, bar_index, exit_fee, "End of backtest"
            )
            portfolio.record_trade_result(pnl, exit_fee)
            position_manager.close_position(position.id)


# Import for confirmation fill
from ..execution.position import PositionFill
