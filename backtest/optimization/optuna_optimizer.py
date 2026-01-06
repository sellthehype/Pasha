"""
Bayesian optimization using Optuna
"""

import optuna
from optuna.samplers import TPESampler
import pandas as pd
from typing import Dict, Callable, Optional
import warnings

from ..config.settings import Config
from ..engine.backtest import BacktestEngine, BacktestResult


class OptunaOptimizer:
    """Bayesian parameter optimization using Optuna"""

    def __init__(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        base_config: Config,
        n_trials: int = 200,
        metric: str = "sharpe_ratio"
    ):
        """
        Initialize optimizer

        Args:
            df: Price data
            symbol: Trading symbol
            timeframe: Timeframe
            base_config: Base configuration to modify
            n_trials: Number of optimization trials
            metric: Optimization metric (sharpe_ratio, total_return_pct, etc.)
        """
        self.df = df
        self.symbol = symbol
        self.timeframe = timeframe
        self.base_config = base_config
        self.n_trials = n_trials
        self.metric = metric

        # Split data
        split_idx = int(len(df) * base_config.train_test_split)
        self.train_df = df.iloc[:split_idx].copy()
        self.test_df = df.iloc[split_idx:].copy()

    def objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function

        Args:
            trial: Optuna trial object

        Returns:
            Optimization metric value
        """
        # Sample parameters
        config = Config(
            # Wave detection
            zigzag_atr_multiplier=trial.suggest_float("zigzag_atr_mult", 1.0, 2.5),

            # Entry
            fib_tolerance_pct=trial.suggest_float("fib_tolerance", 1.0, 5.0),
            initial_position_pct=trial.suggest_int("initial_position_pct", 30, 70),

            # Risk
            base_risk_pct=trial.suggest_float("base_risk_pct", 0.5, 2.0),

            # Modules
            module_a_enabled=self.base_config.module_a_enabled,
            module_b_enabled=self.base_config.module_b_enabled,
            module_c_enabled=self.base_config.module_c_enabled,

            # Fixed params
            initial_balance=self.base_config.initial_balance,
            maker_fee_pct=self.base_config.maker_fee_pct,
            taker_fee_pct=self.base_config.taker_fee_pct,
        )

        # Run backtest on training data
        try:
            engine = BacktestEngine(config)
            result = engine.run(
                self.train_df,
                self.symbol,
                self.timeframe,
                show_progress=False
            )

            # Constraints
            if result.metrics.max_drawdown_pct > 25:
                return -10.0

            if result.metrics.total_trades < 30:
                return -10.0

            if result.metrics.profit_factor < 1.0:
                return -5.0

            # Return optimization metric
            if self.metric == "sharpe_ratio":
                return result.metrics.sharpe_ratio
            elif self.metric == "total_return_pct":
                return result.metrics.total_return_pct
            elif self.metric == "profit_factor":
                return result.metrics.profit_factor
            else:
                return result.metrics.sharpe_ratio

        except Exception as e:
            warnings.warn(f"Trial failed: {e}")
            return -100.0

    def optimize(self, show_progress: bool = True) -> Dict:
        """
        Run optimization

        Args:
            show_progress: Show progress bar

        Returns:
            Dictionary with best params and results
        """
        # Create study
        sampler = TPESampler(seed=42)
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler
        )

        # Suppress Optuna logs if not showing progress
        if not show_progress:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Run optimization
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            show_progress_bar=show_progress
        )

        # Get best params
        best_params = study.best_params
        best_value = study.best_value

        # Create config with best params
        best_config = Config(
            zigzag_atr_multiplier=best_params.get("zigzag_atr_mult", 1.5),
            fib_tolerance_pct=best_params.get("fib_tolerance", 2.5),
            initial_position_pct=best_params.get("initial_position_pct", 50),
            base_risk_pct=best_params.get("base_risk_pct", 1.0),
            module_a_enabled=self.base_config.module_a_enabled,
            module_b_enabled=self.base_config.module_b_enabled,
            module_c_enabled=self.base_config.module_c_enabled,
            initial_balance=self.base_config.initial_balance,
            maker_fee_pct=self.base_config.maker_fee_pct,
            taker_fee_pct=self.base_config.taker_fee_pct,
        )

        # Run on test data
        engine = BacktestEngine(best_config)
        test_result = engine.run(
            self.test_df,
            self.symbol,
            self.timeframe,
            show_progress=False
        )

        return {
            'best_params': best_params,
            'best_train_metric': best_value,
            'test_result': test_result,
            'study': study,
            'n_trials': len(study.trials)
        }

    def get_optimization_history(self, study: optuna.Study) -> pd.DataFrame:
        """Get optimization history as DataFrame"""
        trials_data = []
        for trial in study.trials:
            row = {
                'trial': trial.number,
                'value': trial.value,
                'state': trial.state.name
            }
            row.update(trial.params)
            trials_data.append(row)

        return pd.DataFrame(trials_data)
