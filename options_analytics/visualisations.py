"""
Plotly visualisations for the options_analytics library.

Functions
---------
plot_portfolio_value      : total portfolio value over time for one or more strategies
plot_cumulative_pnl       : cumulative P&L over time for one or more strategies
plot_daily_pnl            : daily P&L bar chart, one subplot per strategy
plot_portfolio_breakdown  : stacked area of cash, stock value, and option MTM
plot_volatility_smile     : implied volatility against strike for a single expiry
"""

from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

STRATEGY_COLOURS = {
    "long_call": "#636EFA",
    "long_put": "#EF553B",
    "covered_call": "#00CC96",
    "cash_secured_put": "#AB63FA",
    "wheel": "#FFA15A",
}

_DEFAULT_TEMPLATE = "plotly_white"
_LEGEND_LAYOUT = dict(
    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
)


def _strategy_label(strategy: str) -> str:
    """Convert snake_case strategy name to Title Case display label."""
    return strategy.replace("_", " ").title()


def _active(results: dict[str, pd.DataFrame]) -> list[tuple[str, pd.DataFrame]]:
    """Return only non-empty result DataFrames."""
    return [(s, df) for s, df in results.items() if not df.empty]


def plot_portfolio_value(
    results: dict[str, pd.DataFrame],
    ticker: str = "",
    start: str = "",
    end: str = "",
) -> go.Figure:
    """
    Plot total portfolio value over time for one or more strategies.

    Parameters
    ----------
    results : mapping of strategy name to backtester results DataFrame
    ticker  : ticker symbol for the chart title (optional)
    start   : backtest start date string for the chart title (optional)
    end     : backtest end date string for the chart title (optional)

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()

    for strategy, df in _active(results):
        colour = STRATEGY_COLOURS.get(strategy, "#888888")
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["total_value"],
                name=_strategy_label(strategy),
                line=dict(color=colour, width=2),
                hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>",
            )
        )

    initial_cash = next(
        (df["total_value"].iloc[0] for _, df in _active(results)), None
    )
    if initial_cash is not None:
        fig.add_hline(
            y=initial_cash,
            line_dash="dash",
            line_color="grey",
            annotation_text="Initial cash",
            annotation_position="bottom right",
        )

    title = "Portfolio Value"
    if ticker:
        title += f" — {ticker}"
    if start and end:
        title += f" {start} to {end}"

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode="x unified",
        template=_DEFAULT_TEMPLATE,
        legend=_LEGEND_LAYOUT,
    )

    return fig


def plot_cumulative_pnl(
    results: dict[str, pd.DataFrame],
    ticker: str = "",
    start: str = "",
    end: str = "",
) -> go.Figure:
    """
    Plot cumulative P&L over time for one or more strategies.

    Parameters
    ----------
    results : mapping of strategy name to backtester results DataFrame
    ticker  : ticker symbol for the chart title (optional)
    start   : backtest start date string for the chart title (optional)
    end     : backtest end date string for the chart title (optional)

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()

    for strategy, df in _active(results):
        colour = STRATEGY_COLOURS.get(strategy, "#888888")
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["cumulative_pnl"],
                name=_strategy_label(strategy),
                line=dict(color=colour, width=2),
                hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>",
            )
        )

    fig.add_hline(y=0, line_dash="dash", line_color="grey")

    title = "Cumulative P&L"
    if ticker:
        title += f" — {ticker}"
    if start and end:
        title += f" {start} to {end}"

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Cumulative P&L ($)",
        hovermode="x unified",
        template=_DEFAULT_TEMPLATE,
        legend=_LEGEND_LAYOUT,
    )

    return fig


def plot_daily_pnl(
    results: dict[str, pd.DataFrame],
    ticker: str = "",
    start: str = "",
    end: str = "",
) -> go.Figure:
    """
    Plot daily P&L as a bar chart, one subplot per strategy.

    Green bars indicate profitable days, red bars indicate losing days.

    Parameters
    ----------
    results : mapping of strategy name to backtester results DataFrame
    ticker  : ticker symbol for the chart title (optional)
    start   : backtest start date string for the chart title (optional)
    end     : backtest end date string for the chart title (optional)

    Returns
    -------
    go.Figure
    """
    active = _active(results)
    n = len(active)

    if n == 0:
        return go.Figure()

    fig = make_subplots(
        rows=n,
        cols=1,
        shared_xaxes=True,
        subplot_titles=[_strategy_label(s) for s, _ in active],
        vertical_spacing=0.06,
    )

    for i, (strategy, df) in enumerate(active, start=1):
        colours = ["#00CC96" if v >= 0 else "#EF553B" for v in df["daily_pnl"]]
        fig.add_trace(
            go.Bar(
                x=df["date"],
                y=df["daily_pnl"],
                marker_color=colours,
                showlegend=False,
                hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>",
            ),
            row=i,
            col=1,
        )

    title = "Daily P&L"
    if ticker:
        title += f" — {ticker}"
    if start and end:
        title += f" {start} to {end}"

    fig.update_layout(
        title=title,
        height=250 * n,
        template=_DEFAULT_TEMPLATE,
    )

    return fig


def plot_portfolio_breakdown(
    df: pd.DataFrame,
    strategy: str,
    ticker: str = "",
) -> go.Figure:
    """
    Plot a stacked area chart of cash, stock value, and option mark-to-market
    for a single strategy run.

    Parameters
    ----------
    df       : backtester results DataFrame for a single strategy
    strategy : strategy name (used in the chart title)
    ticker   : ticker symbol for the chart title (optional)

    Returns
    -------
    go.Figure
        Empty figure if df is empty (busted strategy).
    """
    if df.empty:
        return go.Figure()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["cash"],
            name="Cash",
            stackgroup="one",
            line=dict(color="#636EFA"),
            hovertemplate="$%{y:,.0f}<extra>Cash</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["stock_value"],
            name="Stock value",
            stackgroup="one",
            line=dict(color="#00CC96"),
            hovertemplate="$%{y:,.0f}<extra>Stock value</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["option_value"],
            name="Option MTM",
            stackgroup="one",
            line=dict(color="#FFA15A"),
            hovertemplate="$%{y:,.0f}<extra>Option MTM</extra>",
        )
    )

    title = f"Portfolio Breakdown — {_strategy_label(strategy)}"
    if ticker:
        title += f" ({ticker})"

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Value ($)",
        hovermode="x unified",
        template=_DEFAULT_TEMPLATE,
        legend=_LEGEND_LAYOUT,
    )

    return fig


def plot_volatility_smile(
    strikes: list[float],
    implied_vols: list[float],
    spot: Optional[float] = None,
    expiry_label: str = "",
) -> go.Figure:
    """
    Plot implied volatility against strike price for a single expiry.

    Parameters
    ----------
    strikes       : list of strike prices
    implied_vols  : list of implied volatilities corresponding to each strike
    spot          : current spot price — if provided, draws a vertical marker
    expiry_label  : expiry date or label string for the chart title (optional)

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=strikes,
            y=[iv * 100 for iv in implied_vols],
            mode="lines+markers",
            line=dict(color="#636EFA", width=2),
            marker=dict(size=6),
            hovertemplate="Strike: %{x:.1f}<br>IV: %{y:.2f}%<extra></extra>",
            name="Implied vol",
        )
    )

    if spot is not None:
        fig.add_vline(
            x=spot,
            line_dash="dash",
            line_color="grey",
            annotation_text="Spot",
            annotation_position="top right",
        )

    title = "Volatility Smile"
    if expiry_label:
        title += f" — {expiry_label}"

    fig.update_layout(
        title=title,
        xaxis_title="Strike",
        yaxis_title="Implied Volatility (%)",
        template=_DEFAULT_TEMPLATE,
        legend=_LEGEND_LAYOUT,
    )

    return fig
