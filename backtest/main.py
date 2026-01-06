"""
Main entry point for Elliott Wave Backtest System
"""

import os
import sys
import argparse
from datetime import datetime, timedelta
from typing import List, Dict
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.config.settings import Config
from backtest.data.downloader import BinanceDataDownloader
from backtest.data.storage import DataStorage
from backtest.engine.backtest import BacktestEngine, BacktestResult
from backtest.visualization.dashboard import Dashboard
from backtest.analysis.statistics import TradeStatistics
from backtest.optimization.optuna_optimizer import OptunaOptimizer


def run_single_backtest(
    config: Config,
    symbol: str,
    timeframe: str,
    data_dir: str = "data"
) -> BacktestResult:
    """
    Run backtest for a single symbol/timeframe combination

    Args:
        config: Configuration object
        symbol: Trading symbol
        timeframe: Timeframe
        data_dir: Data directory

    Returns:
        BacktestResult
    """
    # Load data
    storage = DataStorage(data_dir)
    df = storage.load(symbol, timeframe)

    if df is None or len(df) == 0:
        raise ValueError(f"No data found for {symbol} {timeframe}")

    print(f"\nRunning backtest: {symbol} {timeframe}")
    print(f"Data: {len(df)} candles from {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

    # Run backtest
    engine = BacktestEngine(config)
    result = engine.run(df, symbol, timeframe, show_progress=True)

    # Print summary
    print(f"\n{'='*50}")
    print(f"Results: {symbol} {timeframe}")
    print(f"{'='*50}")
    print(f"Total Return: {result.metrics.total_return_pct:.2f}%")
    print(f"Sharpe Ratio: {result.metrics.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {result.metrics.max_drawdown_pct:.2f}%")
    print(f"Total Trades: {result.metrics.total_trades}")
    print(f"Win Rate: {result.metrics.win_rate*100:.1f}%")
    print(f"Profit Factor: {result.metrics.profit_factor:.2f}")
    print(f"Signals Generated: {result.signals_generated}")
    print(f"Signals Executed: {result.signals_executed}")

    return result


def run_multi_timeframe_backtest(
    config: Config,
    symbol: str,
    timeframes: List[str],
    data_dir: str = "data"
) -> Dict[str, BacktestResult]:
    """
    Run backtest across multiple timeframes

    Args:
        config: Configuration
        symbol: Symbol to test
        timeframes: List of timeframes
        data_dir: Data directory

    Returns:
        Dict mapping timeframe to results
    """
    results = {}

    for tf in timeframes:
        try:
            result = run_single_backtest(config, symbol, tf, data_dir)
            results[tf] = result
        except Exception as e:
            print(f"Error on {symbol} {tf}: {e}")

    return results


def download_data(
    config: Config,
    data_dir: str = "data",
    force: bool = False
):
    """
    Download historical data from Binance

    Args:
        config: Configuration with API keys
        data_dir: Directory to save data
        force: Force re-download
    """
    if not config.api_key or not config.api_secret:
        raise ValueError("API keys not configured. Check .env file.")

    downloader = BinanceDataDownloader(
        config.api_key,
        config.api_secret,
        data_dir
    )

    for symbol in config.assets:
        for tf in config.timeframes:
            print(f"Downloading {symbol} {tf}...")
            try:
                df = downloader.load_or_download(
                    symbol, tf,
                    days=config.history_days,
                    force_download=force
                )
                print(f"  Downloaded {len(df)} candles")
            except Exception as e:
                print(f"  Error: {e}")


def run_optimization(
    config: Config,
    symbol: str,
    timeframe: str,
    n_trials: int = 100,
    data_dir: str = "data"
) -> Dict:
    """
    Run parameter optimization

    Args:
        config: Base configuration
        symbol: Symbol to optimize on
        timeframe: Timeframe
        n_trials: Number of optimization trials
        data_dir: Data directory

    Returns:
        Optimization results
    """
    storage = DataStorage(data_dir)
    df = storage.load(symbol, timeframe)

    if df is None or len(df) == 0:
        raise ValueError(f"No data for {symbol} {timeframe}")

    print(f"\nOptimizing {symbol} {timeframe}")
    print(f"Trials: {n_trials}")

    optimizer = OptunaOptimizer(
        df=df,
        symbol=symbol,
        timeframe=timeframe,
        base_config=config,
        n_trials=n_trials,
        metric="sharpe_ratio"
    )

    results = optimizer.optimize(show_progress=True)

    print(f"\n{'='*50}")
    print("Optimization Results")
    print(f"{'='*50}")
    print(f"Best Parameters: {results['best_params']}")
    print(f"Best Train Sharpe: {results['best_train_metric']:.2f}")
    print(f"\nTest Set Performance:")
    test = results['test_result']
    print(f"  Return: {test.metrics.total_return_pct:.2f}%")
    print(f"  Sharpe: {test.metrics.sharpe_ratio:.2f}")
    print(f"  Max DD: {test.metrics.max_drawdown_pct:.2f}%")
    print(f"  Trades: {test.metrics.total_trades}")

    return results


def generate_report(
    result: BacktestResult,
    output_dir: str = "output/reports"
):
    """
    Generate HTML dashboard report

    Args:
        result: Backtest result
        output_dir: Output directory
    """
    dashboard = Dashboard(output_dir)
    filepath = dashboard.generate(
        result,
        filename=f"report_{result.symbol}_{result.timeframe}.html"
    )
    print(f"\nReport generated: {filepath}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Elliott Wave Backtest System"
    )

    parser.add_argument(
        "--action",
        choices=["download", "backtest", "optimize", "full"],
        default="backtest",
        help="Action to perform"
    )
    parser.add_argument(
        "--symbol",
        default="BTCUSDT",
        help="Trading symbol"
    )
    parser.add_argument(
        "--timeframe",
        default="1h",
        help="Timeframe (or 'all' for all timeframes)"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=100,
        help="Optimization trials"
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Data directory"
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force data re-download"
    )

    args = parser.parse_args()

    # Load config
    config = Config()

    if args.action == "download":
        download_data(config, args.data_dir, args.force_download)

    elif args.action == "backtest":
        if args.timeframe == "all":
            results = run_multi_timeframe_backtest(
                config, args.symbol, config.timeframes, args.data_dir
            )
            # Generate comparison report
            dashboard = Dashboard()
            dashboard.generate_comparison_report(
                results,
                f"comparison_{args.symbol}.html"
            )
        else:
            result = run_single_backtest(
                config, args.symbol, args.timeframe, args.data_dir
            )
            generate_report(result)

    elif args.action == "optimize":
        run_optimization(
            config, args.symbol, args.timeframe,
            args.trials, args.data_dir
        )

    elif args.action == "full":
        # Download, backtest all, and generate reports
        print("Downloading data...")
        download_data(config, args.data_dir)

        print("\nRunning backtests...")
        all_results = {}
        for symbol in config.assets:
            for tf in config.timeframes:
                try:
                    result = run_single_backtest(
                        config, symbol, tf, args.data_dir
                    )
                    all_results[f"{symbol}_{tf}"] = result
                    generate_report(result)
                except Exception as e:
                    print(f"Error on {symbol} {tf}: {e}")

        # Generate comparison
        if all_results:
            dashboard = Dashboard()
            dashboard.generate_comparison_report(
                all_results,
                "full_comparison_report.html"
            )


if __name__ == "__main__":
    main()
