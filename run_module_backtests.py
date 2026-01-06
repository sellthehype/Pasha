#!/usr/bin/env python3
"""
Run separate backtests for each module (A, B, C) on BTCUSDT 15m December 2025 data.
Generates 3 HTML verification files.
"""

import pandas as pd
import sys
import os

# Add project root to path
sys.path.insert(0, '/Users/jennieseline/Desktop/Pasha')

from backtest.config.settings import Config
from backtest.visualization.verification_chart import generate_verification_html


def load_and_filter_data(symbol: str, timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Load data and filter to date range"""
    data_path = f'/Users/jennieseline/Desktop/Pasha/data/{symbol}_{timeframe}.csv'
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Filter to date range
    mask = (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
    df_filtered = df[mask].reset_index(drop=True)

    print(f"Loaded {len(df_filtered)} candles from {start_date} to {end_date}")
    print(f"  Date range: {df_filtered['timestamp'].iloc[0]} to {df_filtered['timestamp'].iloc[-1]}")

    return df_filtered


def run_module_backtest(df: pd.DataFrame, symbol: str, timeframe: str, module: str):
    """Run backtest for a specific module and generate HTML"""

    # Create config with only the specified module enabled
    config = Config()

    # Disable all modules first
    config.module_a_enabled = False
    config.module_b_enabled = False
    config.module_c_enabled = False

    # Enable only the specified module
    if module == 'A':
        config.module_a_enabled = True
        module_name = 'ModuleA_Wave3'
    elif module == 'B':
        config.module_b_enabled = True
        module_name = 'ModuleB_Wave5'
    elif module == 'C':
        config.module_c_enabled = True
        config.module_c_zigzag_enabled = True
        config.module_c_flat_enabled = True
        config.module_c_triangle_enabled = True
        module_name = 'ModuleC_Corrective'
    else:
        raise ValueError(f"Unknown module: {module}")

    print(f"\n{'='*60}")
    print(f"Running backtest for Module {module} ({module_name})")
    print(f"{'='*60}")

    # Generate output path
    output_path = f'/Users/jennieseline/Desktop/Pasha/output/verification_{symbol}_{timeframe}_{module_name}_Dec2025.html'

    # Generate verification HTML
    result_path = generate_verification_html(
        df=df,
        symbol=symbol,
        timeframe=timeframe,
        config=config,
        output_path=output_path
    )

    print(f"Generated: {result_path}")
    return result_path


def main():
    # Configuration
    symbol = 'BTCUSDT'
    timeframe = '15m'
    start_date = '2025-12-01'
    end_date = '2025-12-31 23:59:59'

    print("="*60)
    print("BTCUSDT 15m December 2025 Module Backtests")
    print("="*60)

    # Load and filter data
    df = load_and_filter_data(symbol, timeframe, start_date, end_date)

    # Run backtests for each module
    results = []
    for module in ['A', 'B', 'C']:
        path = run_module_backtest(df, symbol, timeframe, module)
        results.append(path)

    print("\n" + "="*60)
    print("COMPLETED - Generated HTML files:")
    print("="*60)
    for path in results:
        print(f"  {path}")

    return results


if __name__ == '__main__':
    main()
