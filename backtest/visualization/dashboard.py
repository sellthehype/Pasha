"""
Dashboard generator for backtest results
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional
import os

from .charts import (
    create_equity_chart,
    create_trade_chart,
    create_monthly_returns_heatmap,
    create_win_loss_distribution
)
from ..engine.backtest import BacktestResult
from ..analysis.statistics import TradeStatistics


class Dashboard:
    """Generates interactive HTML dashboard for backtest results"""

    def __init__(self, output_dir: str = "output/reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate(
        self,
        result: BacktestResult,
        filename: str = "backtest_report.html"
    ) -> str:
        """
        Generate complete dashboard HTML

        Args:
            result: BacktestResult from backtest
            filename: Output filename

        Returns:
            Path to generated HTML file
        """
        # Create all charts
        equity_chart = create_equity_chart(
            result.equity_curve,
            title=f"Equity Curve - {result.symbol} {result.timeframe}"
        )

        monthly_chart = create_monthly_returns_heatmap(result.equity_curve)
        distribution_chart = create_win_loss_distribution(result.trades)

        # Get statistics
        stats = TradeStatistics.analyze_trades(result.trades)

        # Generate HTML
        html_content = self._generate_html(
            result, stats, equity_chart, monthly_chart, distribution_chart
        )

        # Save
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            f.write(html_content)

        return filepath

    def _generate_html(
        self,
        result: BacktestResult,
        stats: Dict,
        equity_chart: go.Figure,
        monthly_chart: go.Figure,
        distribution_chart: go.Figure
    ) -> str:
        """Generate full HTML content"""

        metrics = result.metrics
        portfolio = result.portfolio

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Backtest Report - {result.symbol} {result.timeframe}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        .header {{
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 28px;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.8;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-card h3 {{
            margin: 0;
            font-size: 14px;
            color: #666;
            text-transform: uppercase;
        }}
        .metric-card .value {{
            font-size: 28px;
            font-weight: bold;
            color: #1a1a2e;
            margin: 10px 0 0 0;
        }}
        .metric-card .value.positive {{
            color: #00c853;
        }}
        .metric-card .value.negative {{
            color: #ff1744;
        }}
        .chart-container {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .two-col {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        .stats-table {{
            width: 100%;
            border-collapse: collapse;
        }}
        .stats-table th, .stats-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        .stats-table th {{
            background-color: #f8f9fa;
            font-weight: 600;
        }}
        .module-stats {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin-top: 20px;
        }}
        .module-card {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .module-card h4 {{
            margin: 0 0 10px 0;
            color: #1a1a2e;
        }}
        @media (max-width: 768px) {{
            .two-col {{
                grid-template-columns: 1fr;
            }}
            .module-stats {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Elliott Wave Backtest Report</h1>
            <p>{result.symbol} | {result.timeframe} | {result.equity_curve['timestamp'].iloc[0].strftime('%Y-%m-%d')} to {result.equity_curve['timestamp'].iloc[-1].strftime('%Y-%m-%d')}</p>
        </div>

        <div class="metrics-grid">
            <div class="metric-card">
                <h3>Total Return</h3>
                <div class="value {'positive' if metrics.total_return_pct > 0 else 'negative'}">{metrics.total_return_pct:.2f}%</div>
            </div>
            <div class="metric-card">
                <h3>Sharpe Ratio</h3>
                <div class="value">{metrics.sharpe_ratio:.2f}</div>
            </div>
            <div class="metric-card">
                <h3>Max Drawdown</h3>
                <div class="value negative">-{metrics.max_drawdown_pct:.2f}%</div>
            </div>
            <div class="metric-card">
                <h3>Win Rate</h3>
                <div class="value">{metrics.win_rate*100:.1f}%</div>
            </div>
            <div class="metric-card">
                <h3>Profit Factor</h3>
                <div class="value">{metrics.profit_factor:.2f}</div>
            </div>
            <div class="metric-card">
                <h3>Total Trades</h3>
                <div class="value">{metrics.total_trades}</div>
            </div>
        </div>

        <div class="chart-container">
            <div id="equity-chart"></div>
        </div>

        <div class="two-col">
            <div class="chart-container">
                <div id="monthly-chart"></div>
            </div>
            <div class="chart-container">
                <div id="distribution-chart"></div>
            </div>
        </div>

        <div class="chart-container">
            <h2>Trade Statistics</h2>
            <table class="stats-table">
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Total Trades</td><td>{stats.get('total_trades', 0)}</td></tr>
                <tr><td>Winning Trades</td><td>{stats.get('winning_trades', 0)}</td></tr>
                <tr><td>Losing Trades</td><td>{stats.get('losing_trades', 0)}</td></tr>
                <tr><td>Win Rate</td><td>{stats.get('win_rate', 0)*100:.1f}%</td></tr>
                <tr><td>Average Win</td><td>${stats.get('average_win', 0):.2f}</td></tr>
                <tr><td>Average Loss</td><td>${stats.get('average_loss', 0):.2f}</td></tr>
                <tr><td>Largest Win</td><td>${stats.get('largest_win', 0):.2f}</td></tr>
                <tr><td>Largest Loss</td><td>${stats.get('largest_loss', 0):.2f}</td></tr>
                <tr><td>Expectancy</td><td>${metrics.expectancy:.2f}</td></tr>
                <tr><td>Total Fees</td><td>${stats.get('total_fees', 0):.2f}</td></tr>
            </table>

            <h3>Performance by Module</h3>
            <div class="module-stats">
                {self._generate_module_cards(stats.get('by_module', {}))}
            </div>
        </div>

    </div>

    <script>
        var equityData = {equity_chart.to_json()};
        Plotly.newPlot('equity-chart', equityData.data, equityData.layout);

        var monthlyData = {monthly_chart.to_json()};
        Plotly.newPlot('monthly-chart', monthlyData.data, monthlyData.layout);

        var distData = {distribution_chart.to_json()};
        Plotly.newPlot('distribution-chart', distData.data, distData.layout);
    </script>
</body>
</html>
"""
        return html

    def _generate_module_cards(self, by_module: Dict) -> str:
        """Generate HTML for module performance cards"""
        cards = ""
        for module, data in by_module.items():
            cards += f"""
            <div class="module-card">
                <h4>Module {module}</h4>
                <p>Trades: {data.get('count', 0)}</p>
                <p>Win Rate: {data.get('win_rate', 0)*100:.1f}%</p>
                <p>P&L: ${data.get('pnl', 0):.2f}</p>
            </div>
            """
        return cards

    def generate_comparison_report(
        self,
        results: Dict[str, BacktestResult],
        filename: str = "comparison_report.html"
    ) -> str:
        """
        Generate comparison report for multiple backtests

        Args:
            results: Dict mapping identifier to BacktestResult
            filename: Output filename

        Returns:
            Path to generated file
        """
        rows = []
        for key, result in results.items():
            m = result.metrics
            rows.append({
                'Backtest': key,
                'Return %': f"{m.total_return_pct:.2f}",
                'Sharpe': f"{m.sharpe_ratio:.2f}",
                'Max DD %': f"{m.max_drawdown_pct:.2f}",
                'Trades': m.total_trades,
                'Win Rate': f"{m.win_rate*100:.1f}%",
                'PF': f"{m.profit_factor:.2f}"
            })

        df = pd.DataFrame(rows)

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Backtest Comparison</title>
    <style>
        body {{ font-family: Arial, sans-serif; padding: 20px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #1a1a2e; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>Backtest Comparison Report</h1>
    {df.to_html(index=False)}
</body>
</html>
"""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            f.write(html)

        return filepath
