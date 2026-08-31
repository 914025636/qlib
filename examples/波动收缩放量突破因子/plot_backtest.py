"""Backtest and visualize squeeze-breakout events on BTC/USDT minute data."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots


QLIB_ROOT = Path(__file__).resolve().parents[2]
FACTOR_SCRIPT = Path(__file__).resolve().parents[1] / "binance_btc_usdt_train_backtest.py"
DEFAULT_PROVIDER = QLIB_ROOT / ".qlib" / "qlib_data" / "binance_btc_usdt_1m"
DEFAULT_OUTPUT = QLIB_ROOT / ".qlib" / "experiments" / "squeeze_breakout_factor"
REPORT_NAME = "squeeze_breakout_backtest.html"
TRADES_NAME = "squeeze_breakout_trades.csv"
EQUITY_NAME = "squeeze_breakout_equity.csv"
METRICS_NAME = "squeeze_breakout_metrics.json"
REJECTIONS_NAME = "squeeze_breakout_rejections.csv"
REJECTION_COLUMNS = [
    "signal_row",
    "confirmation_row",
    "entry_row",
    "signal_time",
    "confirmation_time",
    "entry_time",
    "entry_price",
    "confirmation_price",
    "confirmation_upper_wick_ratio",
    "confirmation_close_position",
    "breakout_level",
    "reason",
]
TRADE_COLUMNS = [
    "signal_row",
    "confirmation_row",
    "entry_row",
    "exit_row",
    "signal_time",
    "confirmation_time",
    "entry_time",
    "confirmation_price",
    "exit_time",
    "entry_price",
    "exit_price",
    "gross_return",
    "net_return",
    "cost_drag",
    "equity_before",
    "equity",
    "holding_bars",
    "confirmation_upper_wick_ratio",
    "confirmation_close_position",
    "score",
    "compression_ratio",
    "volume_ratio",
    "breakout_strength",
    "close_position",
    "mfe",
    "mae",
]
FONT_FAMILY = '"Microsoft YaHei UI", "Noto Sans SC", sans-serif'
COLORS = {
    "strategy": "#147d64",
    "benchmark": "#d97706",
    "loss": "#c2414b",
    "price": "#39424e",
    "volume": "#6b7280",
    "grid": "#e5e7eb",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider-uri", type=Path, default=DEFAULT_PROVIDER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start", default="2026-04-01 00:00:00")
    parser.add_argument("--end", default=None)
    parser.add_argument("--horizon-bars", type=int, default=15)
    parser.add_argument("--fee-rate", type=float, default=0.0004)
    parser.add_argument("--slippage-rate", type=float, default=0.0001)
    parser.add_argument(
        "--max-entry-upper-wick-ratio",
        type=float,
        default=0.4,
        help="Maximum upper-wick/range ratio allowed on the confirmation candle",
    )
    parser.add_argument(
        "--min-entry-close-position",
        type=float,
        default=0.6,
        help="Minimum close position in the entry candle range",
    )
    parser.add_argument("--context-bars", type=int, default=30)
    parser.add_argument("--detail-trades", type=int, default=30)
    return parser.parse_args()


def load_factor_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("binance_factor_example", FACTOR_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load factor script: {FACTOR_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select_non_overlapping_trades(
    data: pd.DataFrame,
    features: pd.DataFrame,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    horizon_bars: int,
    fee_rate: float,
    slippage_rate: float,
    max_entry_upper_wick_ratio: float,
    min_entry_close_position: float,
) -> tuple[pd.DataFrame, int, int, int, int, pd.DataFrame]:
    feature_times = data.loc[features.index, "datetime"]
    candidate_mask = (
        feature_times.between(start_time, end_time)
        & features["squeeze_breakout_signal"].eq(1.0)
    )
    candidate_rows = features.index[candidate_mask]
    last_test_row = int(data.index[data["datetime"] <= end_time][-1])
    candle_range = (data["high"] - data["low"]).replace(0, np.nan)
    entry_close_position = (data["close"] - data["low"]) / candle_range
    entry_upper_wick_ratio = (
        data["high"] - data[["open", "close"]].max(axis=1)
    ) / candle_range
    prior_high_20 = data["high"].rolling(20).max().shift(1)
    entry_multiplier = (1 + slippage_rate) * (1 + fee_rate)
    exit_multiplier = (1 - slippage_rate) * (1 - fee_rate)
    cash = 1.0
    last_exit_row = -1
    entry_confirmation_rejected = 0
    incomplete_horizon_signals = 0
    overlapping_signals_skipped = 0
    trade_rows: list[dict[str, object]] = []
    rejection_rows: list[dict[str, object]] = []

    for signal_row in candidate_rows:
        signal_row = int(signal_row)
        confirmation_row = signal_row + 1
        entry_row = signal_row + 2
        exit_row = signal_row + horizon_bars + 2
        if exit_row > last_test_row:
            incomplete_horizon_signals += 1
            continue
        if entry_row <= last_exit_row:
            overlapping_signals_skipped += 1
            continue

        breakout_level = prior_high_20.at[signal_row]
        rejection_reasons: list[str] = []
        if pd.isna(entry_upper_wick_ratio.at[confirmation_row]):
            rejection_reasons.append("确认 K 线无有效波动范围")
        elif entry_upper_wick_ratio.at[confirmation_row] > max_entry_upper_wick_ratio:
            rejection_reasons.append("确认 K 线上影线过长")
        if pd.isna(entry_close_position.at[confirmation_row]):
            rejection_reasons.append("确认 K 线收盘位置无效")
        elif entry_close_position.at[confirmation_row] < min_entry_close_position:
            rejection_reasons.append("确认 K 线收盘位置过低")
        if pd.notna(breakout_level) and data.at[confirmation_row, "close"] < breakout_level:
            rejection_reasons.append("确认时收盘跌回突破位下方")
        if rejection_reasons:
            entry_confirmation_rejected += 1
            rejection_rows.append(
                {
                    "signal_row": signal_row,
                    "confirmation_row": confirmation_row,
                    "entry_row": entry_row,
                    "signal_time": data.at[signal_row, "datetime"],
                    "confirmation_time": data.at[confirmation_row, "datetime"],
                    "entry_time": data.at[entry_row, "datetime"],
                    "entry_price": float(data.at[entry_row, "open"]),
                    "confirmation_price": float(data.at[confirmation_row, "close"]),
                    "confirmation_upper_wick_ratio": float(
                        entry_upper_wick_ratio.at[confirmation_row]
                    ),
                    "confirmation_close_position": float(
                        entry_close_position.at[confirmation_row]
                    ),
                    "breakout_level": float(breakout_level) if pd.notna(breakout_level) else np.nan,
                    "reason": "；".join(rejection_reasons),
                }
            )
            continue

        entry_price = float(data.at[entry_row, "open"])
        exit_price = float(data.at[exit_row, "close"])
        equity_before = cash
        units = equity_before / (entry_price * entry_multiplier)
        cash = units * exit_price * exit_multiplier
        gross_return = exit_price / entry_price - 1
        net_return = cash / equity_before - 1
        holding_window = data.loc[entry_row:exit_row]

        trade_rows.append(
            {
                "signal_row": signal_row,
                "confirmation_row": confirmation_row,
                "entry_row": entry_row,
                "exit_row": exit_row,
                "signal_time": data.at[signal_row, "datetime"],
                "confirmation_time": data.at[confirmation_row, "datetime"],
                "entry_time": data.at[entry_row, "datetime"],
                "confirmation_price": float(data.at[confirmation_row, "close"]),
                "exit_time": data.at[exit_row, "datetime"],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_return": gross_return,
                "net_return": net_return,
                "cost_drag": gross_return - net_return,
                "equity_before": equity_before,
                "equity": cash,
                "holding_bars": horizon_bars,
                "confirmation_upper_wick_ratio": float(
                    entry_upper_wick_ratio.at[confirmation_row]
                ),
                "confirmation_close_position": float(
                    entry_close_position.at[confirmation_row]
                ),
                "score": float(features.at[signal_row, "squeeze_breakout_score"]),
                "compression_ratio": float(
                    features.at[signal_row, "volatility_compression_20_60"]
                ),
                "volume_ratio": float(features.at[signal_row, "volume_ratio_60_prev"]),
                "breakout_strength": float(features.at[signal_row, "breakout_strength_20"]),
                "close_position": float(features.at[signal_row, "close_position"]),
                "mfe": float(holding_window["high"].max() / entry_price - 1),
                "mae": float(holding_window["low"].min() / entry_price - 1),
            }
        )
        last_exit_row = exit_row

    return (
        pd.DataFrame(trade_rows, columns=TRADE_COLUMNS),
        int(len(candidate_rows)),
        incomplete_horizon_signals,
        entry_confirmation_rejected,
        overlapping_signals_skipped,
        pd.DataFrame(rejection_rows, columns=REJECTION_COLUMNS),
    )


def build_equity_curve(
    data: pd.DataFrame,
    trades: pd.DataFrame,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    fee_rate: float,
    slippage_rate: float,
) -> pd.DataFrame:
    test_data = data[data["datetime"].between(start_time, end_time)].copy()
    if test_data.empty:
        raise ValueError("No market data is available in the requested backtest period")

    entry_multiplier = (1 + slippage_rate) * (1 + fee_rate)
    exit_multiplier = (1 - slippage_rate) * (1 - fee_rate)
    equity = pd.Series(np.nan, index=test_data.index, dtype=float)
    position = pd.Series(0.0, index=test_data.index, dtype=float)
    cursor_row = int(test_data.index[0])
    cash = 1.0

    for trade in trades.to_dict("records"):
        entry_row = int(trade["entry_row"])
        exit_row = int(trade["exit_row"])
        equity.loc[(equity.index >= cursor_row) & (equity.index < entry_row)] = cash
        units = cash / (float(trade["entry_price"]) * entry_multiplier)
        holding_index = equity.index[(equity.index >= entry_row) & (equity.index <= exit_row)]
        equity.loc[holding_index] = (
            units * data.loc[holding_index, "close"].astype(float) * exit_multiplier
        )
        position.loc[(position.index >= entry_row) & (position.index < exit_row)] = 1.0
        cash = float(trade["equity"])
        cursor_row = exit_row + 1

    equity.loc[equity.index >= cursor_row] = cash
    equity = equity.ffill().fillna(1.0)
    first_close = float(test_data["close"].iloc[0])
    curve = pd.DataFrame(
        {
            "datetime": test_data["datetime"],
            "close": test_data["close"].astype(float),
            "equity": equity,
            "buy_hold": test_data["close"].astype(float) / first_close,
            "position": position,
        },
        index=test_data.index,
    )
    curve["drawdown"] = curve["equity"] / curve["equity"].cummax() - 1
    return curve.reset_index(drop=True)


def calculate_metrics(
    curve: pd.DataFrame,
    trades: pd.DataFrame,
    candidate_signals: int,
    incomplete_horizon_signals: int,
    entry_confirmation_rejected: int,
    overlapping_signals_skipped: int,
    fee_rate: float,
    slippage_rate: float,
    horizon_bars: int,
    max_entry_upper_wick_ratio: float,
    min_entry_close_position: float,
) -> dict[str, object]:
    net_returns = trades["net_return"] if not trades.empty else pd.Series(dtype=float)
    gross_returns = trades["gross_return"] if not trades.empty else pd.Series(dtype=float)
    positive_sum = float(net_returns[net_returns > 0].sum())
    negative_sum = float(-net_returns[net_returns < 0].sum())
    daily_equity = curve.set_index("datetime")["equity"].resample("1D").last().dropna()
    daily_returns = daily_equity.pct_change().dropna()
    daily_std = float(daily_returns.std(ddof=0)) if not daily_returns.empty else 0.0
    sharpe = (
        float(daily_returns.mean() / daily_std * np.sqrt(365))
        if daily_std > 0 and len(daily_returns) > 1
        else 0.0
    )
    gross_final = float((1 + gross_returns).prod()) if not gross_returns.empty else 1.0
    net_final = float(curve["equity"].iloc[-1])
    holding_bars = trades["holding_bars"] if not trades.empty else pd.Series(dtype=float)
    confirmation_candidates = candidate_signals - incomplete_horizon_signals
    entry_confirmation_passed = (
        confirmation_candidates
        - entry_confirmation_rejected
        - overlapping_signals_skipped
    )
    actionable_signals = entry_confirmation_passed
    return {
        "period_start": curve["datetime"].iloc[0].isoformat(),
        "period_end": curve["datetime"].iloc[-1].isoformat(),
        "horizon_bars": horizon_bars,
        "fee_rate_per_side": fee_rate,
        "slippage_rate_per_side": slippage_rate,
        "round_trip_break_even_return": (
            (1 + fee_rate) * (1 + slippage_rate)
            / ((1 - fee_rate) * (1 - slippage_rate))
            - 1
        ),
        "max_entry_upper_wick_ratio": max_entry_upper_wick_ratio,
        "min_entry_close_position": min_entry_close_position,
        "candidate_signals": candidate_signals,
        "incomplete_horizon_signals": incomplete_horizon_signals,
        "entry_confirmation_rejected": entry_confirmation_rejected,
        "entry_confirmation_passed": entry_confirmation_passed,
        "entry_confirmation_rejection_rate": (
            entry_confirmation_rejected / confirmation_candidates
            if confirmation_candidates
            else 0.0
        ),
        "actionable_signals": actionable_signals,
        "executed_trades": int(len(trades)),
        "skipped_overlapping_signals": overlapping_signals_skipped,
        "signal_accounting_total": (
            incomplete_horizon_signals
            + entry_confirmation_rejected
            + overlapping_signals_skipped
            + int(len(trades))
        ),
        "total_return": net_final - 1,
        "gross_compound_return": gross_final - 1,
        "cost_drag_on_equity": gross_final - net_final,
        "buy_hold_return": float(curve["buy_hold"].iloc[-1] - 1),
        "max_drawdown": float(curve["drawdown"].min()),
        "win_rate": float((net_returns > 0).mean()) if len(net_returns) else 0.0,
        "average_trade_return": float(net_returns.mean()) if len(net_returns) else 0.0,
        "median_trade_return": float(net_returns.median()) if len(net_returns) else 0.0,
        "average_holding_bars": float(holding_bars.mean()) if len(holding_bars) else 0.0,
        "median_holding_bars": float(holding_bars.median()) if len(holding_bars) else 0.0,
        "p90_holding_bars": float(holding_bars.quantile(0.9)) if len(holding_bars) else 0.0,
        "max_holding_bars": int(holding_bars.max()) if len(holding_bars) else 0,
        "profit_factor": positive_sum / negative_sum if negative_sum > 0 else 0.0,
        "daily_sharpe": sharpe,
        "long_exposure": float(curve["position"].mean()),
        "average_mfe": float(trades["mfe"].mean()) if len(trades) else 0.0,
        "average_mae": float(trades["mae"].mean()) if len(trades) else 0.0,
    }


def base_layout(figure: go.Figure, height: int) -> None:
    figure.update_layout(
        height=height,
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        font={"family": FONT_FAMILY, "color": "#202622"},
        hovermode="closest",
        legend={"orientation": "h", "y": 1.03, "x": 0},
        margin={"l": 66, "r": 34, "t": 76, "b": 48},
    )
    figure.update_xaxes(gridcolor=COLORS["grid"], zeroline=False)
    figure.update_yaxes(gridcolor=COLORS["grid"], zeroline=False)


def build_overview(
    curve: pd.DataFrame,
    trades: pd.DataFrame,
    rejections: pd.DataFrame,
) -> go.Figure:
    display = (
        curve.set_index("datetime")
        .resample("1h")
        .agg(
            equity=("equity", "last"),
            buy_hold=("buy_hold", "last"),
            drawdown=("drawdown", "min"),
            close=("close", "last"),
        )
        .dropna()
        .reset_index()
    )
    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.075,
        row_heights=[0.42, 0.20, 0.38],
        subplot_titles=("策略与买入持有净值", "策略回撤", "BTC/USDT 与因子交易点"),
    )
    figure.add_trace(
        go.Scattergl(
            x=display["datetime"],
            y=display["equity"],
            mode="lines",
            name="因子策略",
            line={"color": COLORS["strategy"], "width": 2.2},
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>策略净值=%{y:.6f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scattergl(
            x=display["datetime"],
            y=display["buy_hold"],
            mode="lines",
            name="买入持有",
            line={"color": COLORS["benchmark"], "width": 1.8},
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>买入持有净值=%{y:.6f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scattergl(
            x=display["datetime"],
            y=display["drawdown"],
            mode="lines",
            name="回撤",
            fill="tozeroy",
            line={"color": COLORS["loss"], "width": 1.4},
            fillcolor="rgba(194,65,75,0.18)",
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>回撤=%{y:.2%}<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    figure.add_trace(
        go.Scattergl(
            x=display["datetime"],
            y=display["close"],
            mode="lines",
            name="BTC/USDT",
            line={"color": COLORS["price"], "width": 1.2},
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>收盘价=%{y:.2f}<extra></extra>",
        ),
        row=3,
        col=1,
    )
    if not trades.empty:
        marker_custom = np.column_stack(
            [
                trades["net_return"].to_numpy(),
                trades["score"].to_numpy(),
                trades["volume_ratio"].to_numpy(),
                trades["compression_ratio"].to_numpy(),
                trades["confirmation_upper_wick_ratio"].to_numpy(),
                trades["confirmation_close_position"].to_numpy(),
            ]
        )
        figure.add_trace(
            go.Scattergl(
                x=trades["entry_time"],
                y=trades["entry_price"],
                mode="markers",
                name="买入",
                marker={"symbol": "triangle-up", "size": 8, "color": COLORS["strategy"]},
                customdata=marker_custom,
                hovertemplate=(
                    "买入=%{x|%Y-%m-%d %H:%M}<br>价格=%{y:.2f}<br>"
                    "本笔净收益=%{customdata[0]:.3%}<br>评分=%{customdata[1]:.4f}<br>"
                    "量比=%{customdata[2]:.2f}<br>压缩比=%{customdata[3]:.3f}<br>"
                    "确认上影线=%{customdata[4]:.1%}<br>确认收盘位置=%{customdata[5]:.1%}<extra></extra>"
                ),
            ),
            row=3,
            col=1,
        )
    if not rejections.empty:
        rejection_text = rejections.apply(
            lambda row: (
                f"确认拒绝={row['confirmation_time']:%Y-%m-%d %H:%M}<br>"
                f"价格={row['confirmation_price']:.2f}<br>"
                f"上影线/振幅={row['confirmation_upper_wick_ratio']:.1%}<br>"
                f"收盘位置={row['confirmation_close_position']:.1%}<br>"
                f"原因={row['reason']}"
            ),
            axis=1,
        )
        figure.add_trace(
            go.Scattergl(
                x=rejections["confirmation_time"],
                y=rejections["confirmation_price"],
                mode="markers",
                name="确认拒绝",
                marker={
                    "symbol": "x",
                    "size": 8,
                    "color": COLORS["loss"],
                    "line": {"width": 1.5, "color": COLORS["loss"]},
                },
                text=rejection_text,
                hovertemplate="%{text}<extra></extra>",
            ),
            row=3,
            col=1,
        )
    if not trades.empty:
        figure.add_trace(
            go.Scattergl(
                x=trades["exit_time"],
                y=trades["exit_price"],
                mode="markers",
                name="卖出",
                marker={"symbol": "triangle-down", "size": 8, "color": COLORS["loss"]},
                customdata=trades[["net_return"]].to_numpy(),
                hovertemplate=(
                    "卖出=%{x|%Y-%m-%d %H:%M}<br>价格=%{y:.2f}<br>"
                    "本笔净收益=%{customdata[0]:.3%}<extra></extra>"
                ),
            ),
            row=3,
            col=1,
        )
    figure.update_yaxes(title_text="净值", row=1, col=1)
    figure.update_yaxes(title_text="回撤", tickformat=".1%", row=2, col=1)
    figure.update_yaxes(title_text="价格（USDT）", row=3, col=1)
    figure.update_xaxes(title_text="时间", row=3, col=1)
    base_layout(figure, 940)
    return figure


def build_diagnostics(trades: pd.DataFrame) -> go.Figure:
    figure = make_subplots(
        rows=2,
        cols=2,
        vertical_spacing=0.16,
        horizontal_spacing=0.10,
        subplot_titles=("逐笔净收益分布", "因子评分与净收益", "月度复合收益", "评分分位收益"),
    )
    if trades.empty:
        figure.add_annotation(
            text="当前区间没有可执行事件",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
        base_layout(figure, 650)
        return figure

    return_bp = trades["net_return"] * 10_000
    figure.add_trace(
        go.Histogram(
            x=return_bp,
            nbinsx=45,
            marker={"color": COLORS["strategy"], "line": {"color": "#ffffff", "width": 0.5}},
            name="交易收益",
            hovertemplate="净收益=%{x:.2f} bp<br>笔数=%{y}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scattergl(
            x=trades["score"],
            y=return_bp,
            mode="markers",
            marker={
                "size": 7,
                "color": return_bp,
                "colorscale": "RdYlGn",
                "cmid": 0,
                "opacity": 0.72,
                "colorbar": {"title": "净收益 bp", "len": 0.37, "y": 0.82},
            },
            customdata=trades[["volume_ratio", "compression_ratio"]].to_numpy(),
            hovertemplate=(
                "评分=%{x:.4f}<br>净收益=%{y:.2f} bp<br>"
                "量比=%{customdata[0]:.2f}<br>压缩比=%{customdata[1]:.3f}<extra></extra>"
            ),
            name="事件",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    monthly = (
        trades.set_index("exit_time")["net_return"]
        .resample("ME")
        .apply(lambda values: (1 + values).prod() - 1)
    )
    monthly_colors = [COLORS["strategy"] if value >= 0 else COLORS["loss"] for value in monthly]
    figure.add_trace(
        go.Bar(
            x=monthly.index,
            y=monthly,
            marker_color=monthly_colors,
            name="月度收益",
            hovertemplate="%{x|%Y-%m}<br>复合收益=%{y:.2%}<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    quartiles = pd.qcut(trades["score"], q=4, duplicates="drop")
    for category in quartiles.cat.categories:
        values = return_bp[quartiles == category]
        figure.add_trace(
            go.Box(
                y=values,
                name=str(category),
                boxpoints="outliers",
                marker={"size": 4},
                line={"color": COLORS["benchmark"]},
                showlegend=False,
                hovertemplate="净收益=%{y:.2f} bp<extra></extra>",
            ),
            row=2,
            col=2,
        )
    figure.add_hline(y=0, line={"color": "#6b7280", "dash": "dot"}, row=1, col=1)
    figure.add_hline(y=0, line={"color": "#6b7280", "dash": "dot"}, row=1, col=2)
    figure.add_hline(y=0, line={"color": "#6b7280", "dash": "dot"}, row=2, col=1)
    figure.add_hline(y=0, line={"color": "#6b7280", "dash": "dot"}, row=2, col=2)
    figure.update_xaxes(title_text="净收益（bp）", row=1, col=1)
    figure.update_yaxes(title_text="笔数", row=1, col=1)
    figure.update_xaxes(title_text="连续评分", row=1, col=2)
    figure.update_yaxes(title_text="净收益（bp）", row=1, col=2)
    figure.update_yaxes(title_text="复合收益", tickformat=".1%", row=2, col=1)
    figure.update_xaxes(title_text="评分四分位", row=2, col=2)
    figure.update_yaxes(title_text="净收益（bp）", row=2, col=2)
    base_layout(figure, 760)
    return figure


def representative_trades(trades: pd.DataFrame, maximum: int) -> pd.DataFrame:
    if len(trades) <= maximum:
        return trades.copy()
    group_size = max(maximum // 3, 1)
    selected = pd.concat(
        [
            trades.nsmallest(group_size, "net_return"),
            trades.nlargest(group_size, "net_return"),
            trades.tail(maximum - 2 * group_size),
        ]
    )
    return selected.drop_duplicates("entry_time").sort_values("entry_time").head(maximum)


def build_trade_detail(
    data: pd.DataFrame,
    trades: pd.DataFrame,
    context_bars: int,
    maximum: int,
) -> go.Figure:
    selected = representative_trades(trades, maximum)
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.76, 0.24],
    )
    if selected.empty:
        figure.add_annotation(
            text="当前区间没有可展示的交易",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
        base_layout(figure, 420)
        return figure

    prior_high = data["high"].rolling(20).max().shift(1)
    trace_groups: list[list[int]] = []
    titles: list[str] = []
    x_ranges: list[list[pd.Timestamp]] = []
    price_ranges: list[list[float]] = []
    volume_ranges: list[list[float]] = []

    for display_number, (_, trade) in enumerate(selected.iterrows(), start=1):
        signal_row = int(trade["signal_row"])
        entry_row = int(trade["entry_row"])
        exit_row = int(trade["exit_row"])
        window_start_row = max(int(data.index[0]), signal_row - context_bars)
        window_end_row = min(int(data.index[-1]), exit_row + context_bars)
        window = data.loc[window_start_row:window_end_row]
        visible = display_number == 1
        group: list[int] = []
        candle_colors = np.where(window["close"] >= window["open"], "#3aa981", "#d95663")

        figure.add_trace(
            go.Candlestick(
                x=window["datetime"],
                open=window["open"],
                high=window["high"],
                low=window["low"],
                close=window["close"],
                increasing_line_color="#238b68",
                decreasing_line_color="#c2414b",
                increasing_fillcolor="#a7e3cf",
                decreasing_fillcolor="#f0b8bd",
                visible=visible,
                showlegend=False,
                name="K 线",
            ),
            row=1,
            col=1,
        )
        group.append(len(figure.data) - 1)
        figure.add_trace(
            go.Scatter(
                x=window["datetime"],
                y=prior_high.loc[window.index],
                mode="lines",
                line={"color": COLORS["benchmark"], "width": 1.4, "dash": "dash"},
                name="前 20 根上沿",
                visible=visible,
                showlegend=False,
                hovertemplate="区间上沿=%{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        group.append(len(figure.data) - 1)
        figure.add_trace(
            go.Scatter(
                x=[trade["confirmation_time"]],
                y=[trade["confirmation_price"]],
                mode="markers+text",
                text=["确认"],
                textposition="top center",
                marker={
                    "symbol": "diamond",
                    "size": 11,
                    "color": COLORS["benchmark"],
                    "line": {"color": "#ffffff", "width": 1},
                },
                visible=visible,
                showlegend=False,
                customdata=[
                    [
                        trade["confirmation_upper_wick_ratio"],
                        trade["confirmation_close_position"],
                    ]
                ],
                hovertemplate=(
                    "确认时间=%{x|%Y-%m-%d %H:%M}<br>确认收盘=%{y:.2f}<br>"
                    "上影线/振幅=%{customdata[0]:.1%}<br>"
                    "收盘位置=%{customdata[1]:.1%}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
        group.append(len(figure.data) - 1)
        figure.add_trace(
            go.Scatter(
                x=[trade["entry_time"], trade["exit_time"]],
                y=[trade["entry_price"], trade["exit_price"]],
                mode="markers+text",
                text=["买入", "卖出"],
                textposition=["top center", "bottom center"],
                marker={
                    "symbol": ["triangle-up", "triangle-down"],
                    "size": [15, 15],
                    "color": [COLORS["strategy"], COLORS["loss"]],
                    "line": {"color": "#ffffff", "width": 1},
                },
                visible=visible,
                showlegend=False,
                customdata=[[trade["net_return"]], [trade["net_return"]]],
                hovertemplate=(
                    "%{text}时间=%{x|%Y-%m-%d %H:%M}<br>价格=%{y:.2f}<br>"
                    "本笔净收益=%{customdata[0]:.3%}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
        group.append(len(figure.data) - 1)
        figure.add_trace(
            go.Bar(
                x=window["datetime"],
                y=window["volume"],
                marker_color=candle_colors,
                visible=visible,
                showlegend=False,
                name="成交量",
                hovertemplate="%{x|%Y-%m-%d %H:%M}<br>成交量=%{y:.4g}<extra></extra>",
            ),
            row=2,
            col=1,
        )
        group.append(len(figure.data) - 1)
        trace_groups.append(group)
        titles.append(
            f"代表交易 {display_number}/{len(selected)} | 信号 {pd.Timestamp(trade['signal_time']):%Y-%m-%d %H:%M} | "
            f"确认 {pd.Timestamp(trade['confirmation_time']):%H:%M} | 开仓 {pd.Timestamp(trade['entry_time']):%H:%M} | "
            f"净收益 {trade['net_return']:.3%} | 上影线 {trade['confirmation_upper_wick_ratio']:.1%}"
        )
        x_ranges.append([window["datetime"].iloc[0], window["datetime"].iloc[-1]])
        visible_prices = pd.concat(
            [window["low"], window["high"], prior_high.loc[window.index]]
        ).dropna()
        price_min = float(visible_prices.min())
        price_max = float(visible_prices.max())
        price_padding = max((price_max - price_min) * 0.08, price_max * 0.001)
        price_ranges.append([price_min - price_padding, price_max + price_padding])
        volume_ranges.append([0.0, float(window["volume"].max()) * 1.12])

    buttons = []
    for index, group in enumerate(trace_groups):
        visibility = [False] * len(figure.data)
        for trace_index in group:
            visibility[trace_index] = True
        buttons.append(
            {
                "label": titles[index],
                "method": "update",
                "args": [
                    {"visible": visibility},
                    {
                        "xaxis.range": x_ranges[index],
                        "xaxis.autorange": False,
                        "xaxis.rangeslider.visible": False,
                        "xaxis2.range": x_ranges[index],
                        "xaxis2.autorange": False,
                        "yaxis.range": price_ranges[index],
                        "yaxis.autorange": False,
                        "yaxis2.range": volume_ranges[index],
                        "yaxis2.autorange": False,
                    },
                ],
            }
        )
    figure.update_layout(
        updatemenus=[
            {
                "type": "dropdown",
                "direction": "down",
                "x": 0.5,
                "xanchor": "center",
                "y": 1.12,
                "buttons": buttons,
            }
        ],
    )
    figure.update_xaxes(range=x_ranges[0], autorange=False, rangeslider_visible=False)
    figure.update_xaxes(title_text="时间", row=2, col=1)
    figure.update_yaxes(
        title_text="价格（USDT）", range=price_ranges[0], autorange=False, row=1, col=1
    )
    figure.update_yaxes(
        title_text="成交量", range=volume_ranges[0], autorange=False, row=2, col=1
    )
    base_layout(figure, 720)
    return figure


def format_metric(value: object, kind: str) -> str:
    number = float(value)
    if kind == "percent":
        return f"{number:.2%}"
    if kind == "decimal":
        return f"{number:.3f}"
    return f"{int(number):,}"


def metric_cards(metrics: dict[str, object]) -> str:
    card_data = [
        ("候选信号 / 成交", f"{metrics['candidate_signals']:,} / {metrics['executed_trades']:,}", ""),
        ("策略总收益", format_metric(metrics["total_return"], "percent"), "positive" if metrics["total_return"] >= 0 else "negative"),
        ("买入持有", format_metric(metrics["buy_hold_return"], "percent"), "benchmark"),
        ("最大回撤", format_metric(metrics["max_drawdown"], "percent"), "negative"),
        ("交易胜率", format_metric(metrics["win_rate"], "percent"), ""),
        ("平均每笔", format_metric(metrics["average_trade_return"], "percent"), "positive" if metrics["average_trade_return"] >= 0 else "negative"),
        ("利润因子", format_metric(metrics["profit_factor"], "decimal"), ""),
        ("持仓暴露", format_metric(metrics["long_exposure"], "percent"), ""),
        ("平均持仓", f"{float(metrics['average_holding_bars']):.1f} 分钟", ""),
        ("中位 / P90 持仓", f"{float(metrics['median_holding_bars']):.1f} / {float(metrics['p90_holding_bars']):.1f} 分钟", ""),
        ("最长持仓", f"{int(metrics['max_holding_bars']):,} 分钟", ""),
    ]
    return "".join(
        f"<div class='metric'><span>{label}</span><strong class='{css_class}'>{value}</strong></div>"
        for label, value, css_class in card_data
    )


def trade_table_html(trades: pd.DataFrame) -> str:
    if trades.empty:
        return "<p class='empty'>当前区间没有可执行交易。</p>"
    display = trades.sort_values("entry_time", ascending=False).copy()
    display.insert(0, "trade", display.index + 1)
    display = display[
        [
            "trade",
            "signal_time",
            "confirmation_time",
            "entry_time",
            "exit_time",
            "holding_bars",
            "net_return",
            "confirmation_upper_wick_ratio",
            "confirmation_close_position",
            "score",
            "compression_ratio",
            "volume_ratio",
            "breakout_strength",
            "mfe",
            "mae",
            "equity",
        ]
    ]
    display.columns = [
        "交易",
        "信号时间",
        "确认时间",
        "买入时间",
        "卖出时间",
        "持仓分钟",
        "净收益",
        "确认上影线/振幅",
        "确认收盘位置",
        "评分",
        "压缩比",
        "量比",
        "突破强度",
        "MFE",
        "MAE",
        "交易后净值",
    ]
    for column in ["信号时间", "确认时间", "买入时间", "卖出时间"]:
        display[column] = pd.to_datetime(display[column]).dt.strftime("%Y-%m-%d %H:%M")
    for column in ["净收益", "确认上影线/振幅", "确认收盘位置", "MFE", "MAE"]:
        display[column] = display[column].map(lambda value: f"{value:.3%}")
    for column in ["评分", "压缩比", "突破强度"]:
        display[column] = display[column].map(lambda value: f"{value:.4f}")
    display["量比"] = display["量比"].map(lambda value: f"{value:.2f}")
    display["交易后净值"] = display["交易后净值"].map(lambda value: f"{value:.6f}")
    return display.to_html(index=False, classes="trade-table", border=0, escape=True)


def rejection_table(rejections: pd.DataFrame) -> str:
    if rejections.empty:
        return "<p class='empty'>没有被确认规则拒绝的信号。</p>"
    display = rejections.sort_values("confirmation_time", ascending=False).copy()
    display = display[
        [
            "signal_time",
            "confirmation_time",
            "entry_time",
            "entry_price",
            "confirmation_price",
            "confirmation_upper_wick_ratio",
            "confirmation_close_position",
            "breakout_level",
            "reason",
        ]
    ]
    display.columns = [
        "信号时间",
        "确认时间",
        "原计划开仓时间",
        "原计划开仓价",
        "确认收盘价",
        "确认上影线/振幅",
        "确认收盘位置",
        "突破位",
        "拒绝原因",
    ]
    for column in ["信号时间", "确认时间", "原计划开仓时间"]:
        display[column] = pd.to_datetime(display[column]).dt.strftime("%Y-%m-%d %H:%M")
    for column in ["确认上影线/振幅", "确认收盘位置"]:
        display[column] = display[column].map(lambda value: f"{value:.3%}")
    for column in ["原计划开仓价", "确认收盘价", "突破位"]:
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.2f}"
        )
    return display.to_html(index=False, classes="trade-table", border=0, escape=True)


def write_outputs(
    output_dir: Path,
    curve: pd.DataFrame,
    trades: pd.DataFrame,
    rejections: pd.DataFrame,
    metrics: dict[str, object],
    overview: go.Figure,
    diagnostics: go.Figure,
    detail: go.Figure,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    trades.to_csv(output_dir / TRADES_NAME, index=False)
    rejections.to_csv(output_dir / REJECTIONS_NAME, index=False)
    curve.to_csv(output_dir / EQUITY_NAME, index=False)
    (output_dir / METRICS_NAME).write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_config = {"displaylogo": False, "responsive": True, "scrollZoom": True}
    overview_html = pio.to_html(
        overview, full_html=False, include_plotlyjs=True, config=plot_config
    )
    diagnostics_html = pio.to_html(
        diagnostics, full_html=False, include_plotlyjs=False, config=plot_config
    )
    detail_html = pio.to_html(
        detail, full_html=False, include_plotlyjs=False, config=plot_config
    )
    total_return = float(metrics["total_return"])
    buy_hold_return = float(metrics["buy_hold_return"])
    if total_return > buy_hold_return:
        verdict_class = "good"
        verdict = "因子策略在当前区间跑赢买入持有。"
    elif total_return > 0:
        verdict_class = "watch"
        verdict = "因子策略取得正收益，但未跑赢买入持有。"
    else:
        verdict_class = "bad"
        verdict = "当前参数没有形成正向样本外收益，图表用于诊断而不是证明策略有效。"
    table_html = trade_table_html(trades)
    rejection_table_html = rejection_table(rejections)
    break_even = float(metrics["round_trip_break_even_return"])
    meta = (
        f"单边手续费 {float(metrics['fee_rate_per_side']):.2%}，单边滑点 "
        f"{float(metrics['slippage_rate_per_side']):.2%}，双边盈亏平衡约 "
        f"{break_even:.2%}；实际持仓中位数 {float(metrics['median_holding_bars']):.1f} 分钟，"
        f"最长 {int(metrics['max_holding_bars']):,} 分钟。"
    )
    report_path = output_dir / REPORT_NAME
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>波动收缩放量突破因子回测</title>
<style>
:root {{ --ink:#202622; --muted:#667069; --line:#dfe5df; --paper:#ffffff; --green:#147d64; --red:#b83d49; --amber:#b66708; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; color:var(--ink); font-family:{FONT_FAMILY}; background-color:#f4f6f2; background-image:linear-gradient(rgba(54,76,61,.035) 1px, transparent 1px),linear-gradient(90deg,rgba(54,76,61,.035) 1px,transparent 1px); background-size:28px 28px; }}
main {{ max-width:1540px; margin:0 auto; padding:24px 24px 52px; }}
header {{ display:flex; justify-content:space-between; gap:24px; align-items:end; padding:8px 0 18px; border-bottom:2px solid #2d3b32; }}
h1 {{ margin:0; font-size:28px; line-height:1.25; letter-spacing:0; }}
.period {{ color:var(--muted); font-size:13px; text-align:right; }}
.metrics {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; margin:18px 0 12px; }}
.metric {{ min-height:82px; background:var(--paper); border:1px solid var(--line); border-radius:7px; padding:13px 15px; }}
.metric span {{ display:block; color:var(--muted); font-size:12px; margin-bottom:8px; }}
.metric strong {{ display:block; font-size:23px; line-height:1; font-variant-numeric:tabular-nums; }}
.positive {{ color:var(--green); }} .negative {{ color:var(--red); }} .benchmark {{ color:var(--amber); }}
.verdict {{ border-left:4px solid; padding:12px 15px; margin:12px 0 20px; background:#fff; font-size:14px; }}
.verdict.good {{ border-color:var(--green); }} .verdict.watch {{ border-color:var(--amber); }} .verdict.bad {{ border-color:var(--red); }}
.toolbar {{ display:flex; gap:10px; justify-content:flex-end; margin:10px 0; }}
.toolbar a {{ color:#fff; background:#34473b; text-decoration:none; border-radius:6px; padding:8px 11px; font-size:12px; }}
.panel {{ background:var(--paper); border:1px solid var(--line); border-radius:7px; margin:16px 0; overflow:hidden; }}
.section-head {{ display:flex; align-items:baseline; justify-content:space-between; padding:18px 20px 0; }}
.section-head h2 {{ margin:0; font-size:18px; }}
.section-head span {{ color:var(--muted); font-size:12px; }}
.table-wrap {{ max-height:520px; overflow:auto; margin:15px 18px 20px; border:1px solid var(--line); }}
.trade-table {{ width:100%; border-collapse:collapse; white-space:nowrap; font-size:12px; font-variant-numeric:tabular-nums; }}
.trade-table th {{ position:sticky; top:0; z-index:1; color:#fff; background:#34473b; text-align:right; padding:9px 10px; }}
.trade-table td {{ border-bottom:1px solid #edf0ec; text-align:right; padding:8px 10px; }}
.trade-table tr:nth-child(even) td {{ background:#f7f9f6; }}
.empty {{ padding:24px; color:var(--muted); }}
@media(max-width:900px) {{ main {{ padding:14px 8px 36px; }} header {{ display:block; }} .period {{ text-align:left; margin-top:8px; }} .metrics {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} h1 {{ font-size:22px; }} .section-head {{ padding:14px 12px 0; }} }}
@media(max-width:480px) {{ .metrics {{ grid-template-columns:1fr 1fr; }} .metric {{ min-height:74px; padding:11px; }} .metric strong {{ font-size:18px; }} .toolbar {{ justify-content:stretch; }} .toolbar a {{ flex:1; text-align:center; }} }}
</style>
</head>
<body><main>
<header><div><h1>波动收缩放量突破因子回测</h1></div><div class="period">{metrics['period_start']}<br>至 {metrics['period_end']} · {metrics['horizon_bars']} 分钟持有期</div></header>
<section class="metrics">{metric_cards(metrics)}</section>
<div class="verdict {verdict_class}">{verdict}</div>
<p class="report-meta">{meta}</p>
<nav class="toolbar"><a href="{TRADES_NAME}">成交明细 CSV</a><a href="{REJECTIONS_NAME}">拒绝信号 CSV</a><a href="{EQUITY_NAME}">净值曲线 CSV</a><a href="{METRICS_NAME}">指标 JSON</a></nav>
<section class="panel">{overview_html}</section>
<section class="panel"><div class="section-head"><h2>收益与因子诊断</h2><span>收益均已计入双边手续费与滑点</span></div>{diagnostics_html}</section>
<section class="panel"><div class="section-head"><h2>代表交易 K 线</h2><span>下拉切换最好、最差及最近交易</span></div>{detail_html}</section>
<section class="panel"><div class="section-head"><h2>全部交易明细</h2><span>共 {len(trades):,} 笔</span></div><div class="table-wrap">{table_html}</div></section>
<section class="panel"><div class="section-head"><h2>确认拒绝信号</h2><span>共 {len(rejections):,} 笔，包含长上影和收盘位置不合格信号</span></div><div class="table-wrap">{rejection_table_html}</div></section>
</main></body></html>"""
    report_path.write_text(html, encoding="utf-8")
    return report_path


def main() -> None:
    args = parse_args()
    if args.horizon_bars < 1 or args.context_bars < 1 or args.detail_trades < 1:
        raise ValueError("horizon-bars, context-bars and detail-trades must be positive")
    if not 0 <= args.fee_rate < 1 or not 0 <= args.slippage_rate < 1:
        raise ValueError("fee-rate and slippage-rate must be in [0, 1)")
    if not 0 <= args.max_entry_upper_wick_ratio <= 1:
        raise ValueError("max-entry-upper-wick-ratio must be in [0, 1]")
    if not 0 <= args.min_entry_close_position <= 1:
        raise ValueError("min-entry-close-position must be in [0, 1]")

    factor_module = load_factor_module()
    start_time = pd.Timestamp(args.start)
    warmup_start = start_time - pd.DateOffset(days=2)
    data = factor_module.load_market_data(
        args.provider_uri,
        warmup_start.strftime("%Y-%m-%d %H:%M:%S"),
        args.end,
    )
    end_time = pd.Timestamp(args.end) if args.end else pd.Timestamp(data["datetime"].max())
    if end_time <= start_time:
        raise ValueError("end must be later than start")
    features, _ = factor_module.make_dataset(data, args.horizon_bars)
    (
        trades,
        candidate_signals,
        incomplete_horizon_signals,
        entry_confirmation_rejected,
        overlapping_signals_skipped,
        rejections,
    ) = select_non_overlapping_trades(
        data,
        features,
        start_time,
        end_time,
        args.horizon_bars,
        args.fee_rate,
        args.slippage_rate,
        args.max_entry_upper_wick_ratio,
        args.min_entry_close_position,
    )
    curve = build_equity_curve(
        data, trades, start_time, end_time, args.fee_rate, args.slippage_rate
    )
    metrics = calculate_metrics(
        curve,
        trades,
        candidate_signals,
        incomplete_horizon_signals,
        entry_confirmation_rejected,
        overlapping_signals_skipped,
        args.fee_rate,
        args.slippage_rate,
        args.horizon_bars,
        args.max_entry_upper_wick_ratio,
        args.min_entry_close_position,
    )
    overview = build_overview(curve, trades, rejections)
    diagnostics = build_diagnostics(trades)
    detail = build_trade_detail(data, trades, args.context_bars, args.detail_trades)
    report_path = write_outputs(
        args.output_dir,
        curve,
        trades,
        rejections,
        metrics,
        overview,
        diagnostics,
        detail,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Interactive report: {report_path.resolve()}")


if __name__ == "__main__":
    main()