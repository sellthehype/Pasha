"""
Chart generation for backtest visualization
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Optional, Dict


def create_equity_chart(
    equity_df: pd.DataFrame,
    title: str = "Equity Curve"
) -> go.Figure:
    """
    Create interactive equity curve chart

    Args:
        equity_df: DataFrame with equity data
        title: Chart title

    Returns:
        Plotly Figure
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=(title, 'Drawdown %')
    )

    # Equity curve
    fig.add_trace(
        go.Scatter(
            x=equity_df['timestamp'],
            y=equity_df['equity'],
            mode='lines',
            name='Equity',
            line=dict(color='blue', width=1)
        ),
        row=1, col=1
    )

    # Drawdown
    fig.add_trace(
        go.Scatter(
            x=equity_df['timestamp'],
            y=-equity_df['drawdown_pct'],
            mode='lines',
            name='Drawdown %',
            fill='tozeroy',
            line=dict(color='red', width=1)
        ),
        row=2, col=1
    )

    fig.update_layout(
        height=600,
        showlegend=True,
        hovermode='x unified'
    )

    fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)

    return fig


def create_trade_chart(
    df: pd.DataFrame,
    trades: List,
    start_idx: int = 0,
    end_idx: int = None,
    title: str = "Price with Trades"
) -> go.Figure:
    """
    Create candlestick chart with trade markers

    Args:
        df: OHLCV DataFrame
        trades: List of Position objects
        start_idx: Start index for display
        end_idx: End index
        title: Chart title

    Returns:
        Plotly Figure
    """
    if end_idx is None:
        end_idx = len(df)

    subset = df.iloc[start_idx:end_idx]

    fig = go.Figure()

    # Candlesticks
    fig.add_trace(
        go.Candlestick(
            x=subset['timestamp'],
            open=subset['open'],
            high=subset['high'],
            low=subset['low'],
            close=subset['close'],
            name='Price'
        )
    )

    # Add trade markers
    for trade in trades:
        if trade.entry_bar_index < start_idx or trade.entry_bar_index >= end_idx:
            continue

        # Entry marker
        entry_color = 'green' if trade.is_long else 'red'
        fig.add_trace(
            go.Scatter(
                x=[trade.entry_timestamp],
                y=[trade.entry_price],
                mode='markers',
                marker=dict(
                    symbol='triangle-up' if trade.is_long else 'triangle-down',
                    size=12,
                    color=entry_color
                ),
                name=f'Entry {trade.module}',
                showlegend=False,
                hovertemplate=f"Entry<br>Price: {trade.entry_price:.2f}<br>Module: {trade.module}"
            )
        )

        # Exit marker (if available)
        if trade.exit_timestamp:
            exit_color = 'blue'
            fig.add_trace(
                go.Scatter(
                    x=[trade.exit_timestamp],
                    y=[trade.entry_price],  # Use entry price as proxy if exit price not stored
                    mode='markers',
                    marker=dict(
                        symbol='x',
                        size=10,
                        color=exit_color
                    ),
                    name=f'Exit',
                    showlegend=False,
                    hovertemplate=f"Exit<br>Reason: {trade.exit_reason}"
                )
            )

        # Stop loss line
        fig.add_hline(
            y=trade.original_stop_loss,
            line_dash="dash",
            line_color="red",
            opacity=0.3,
            annotation_text="SL"
        )

        # TP lines
        fig.add_hline(
            y=trade.take_profit_1,
            line_dash="dash",
            line_color="green",
            opacity=0.3,
            annotation_text="TP1"
        )

    fig.update_layout(
        title=title,
        height=600,
        xaxis_rangeslider_visible=False,
        showlegend=True
    )

    return fig


def create_monthly_returns_heatmap(equity_df: pd.DataFrame) -> go.Figure:
    """
    Create monthly returns heatmap

    Args:
        equity_df: DataFrame with equity data

    Returns:
        Plotly Figure
    """
    if 'timestamp' not in equity_df.columns or len(equity_df) < 2:
        return go.Figure()

    # Calculate monthly returns
    equity_df = equity_df.copy()
    equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
    equity_df.set_index('timestamp', inplace=True)

    monthly = equity_df['equity'].resample('M').last()
    monthly_returns = monthly.pct_change() * 100

    # Create pivot table
    monthly_df = pd.DataFrame({
        'year': monthly_returns.index.year,
        'month': monthly_returns.index.month,
        'return': monthly_returns.values
    })

    pivot = monthly_df.pivot(index='year', columns='month', values='return')

    # Month names
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=month_names[:len(pivot.columns)],
        y=pivot.index,
        colorscale='RdYlGn',
        zmid=0,
        text=pivot.values,
        texttemplate='%{text:.1f}%',
        hovertemplate='%{x} %{y}: %{z:.2f}%<extra></extra>'
    ))

    fig.update_layout(
        title='Monthly Returns (%)',
        height=400
    )

    return fig


def create_win_loss_distribution(trades: List) -> go.Figure:
    """
    Create distribution chart of trade P&L

    Args:
        trades: List of Position objects

    Returns:
        Plotly Figure
    """
    if not trades:
        return go.Figure()

    pnls = [t.realized_pnl for t in trades]
    colors = ['green' if p > 0 else 'red' for p in pnls]

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=pnls,
        nbinsx=30,
        marker_color='blue',
        opacity=0.7,
        name='P&L Distribution'
    ))

    fig.add_vline(x=0, line_dash="dash", line_color="black")

    fig.update_layout(
        title='Trade P&L Distribution',
        xaxis_title='P&L ($)',
        yaxis_title='Count',
        height=400
    )

    return fig
