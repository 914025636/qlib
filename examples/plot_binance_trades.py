"""Create an interactive price chart for the Binance backtest trades."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

import qlib
from qlib.data import D


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROVIDER = ROOT / ".qlib" / "qlib_data" / "binance_btc_usdt_1m"
DEFAULT_TRADES = ROOT / ".qlib" / "experiments" / "binance_btc_usdt_1m" / "trades.csv"
DEFAULT_OUTPUT = ROOT / ".qlib" / "experiments" / "binance_btc_usdt_1m"
FIELDS = ["$open", "$high", "$low", "$close", "$volume"]
ONE_MINUTE = pd.Timedelta(1, unit="min")
MAX_DETAIL_CANDLES = 900
CANDLE_INTERVALS = [
    ("1min", "1 分钟", pd.Timedelta(1, unit="min")),
    ("5min", "5 分钟", pd.Timedelta(5, unit="min")),
    ("15min", "15 分钟", pd.Timedelta(15, unit="min")),
    ("30min", "30 分钟", pd.Timedelta(30, unit="min")),
    ("1h", "1 小时", pd.Timedelta(1, unit="h")),
    ("2h", "2 小时", pd.Timedelta(2, unit="h")),
    ("4h", "4 小时", pd.Timedelta(4, unit="h")),
    ("8h", "8 小时", pd.Timedelta(8, unit="h")),
    ("12h", "12 小时", pd.Timedelta(12, unit="h")),
    ("1D", "1 天", pd.Timedelta(1, unit="D")),
]
EXIT_REASON_LABELS = {
    "prediction_exit": "预测转弱",
    "horizon_expired": "预测到期",
    "initial_stop": "初始止损",
    "trailing_stop": "移动止损",
    "breakout_failure": "突破失败",
    "end_of_test": "回测结束",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider-uri", type=Path, default=DEFAULT_PROVIDER)
    parser.add_argument("--trades-path", type=Path, default=DEFAULT_TRADES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--window-bars",
        type=int,
        default=30,
        help="Number of 1-minute bars shown before and after each execution",
    )
    return parser.parse_args()


def load_ohlcv(provider_uri: Path, start_time: pd.Timestamp, end_time: pd.Timestamp) -> pd.DataFrame:
    qlib.init(provider_uri=str(provider_uri.resolve()), auto_mount=False, redis_port=-1)
    data = D.features(
        ["BTC_USDT"],
        FIELDS,
        start_time=start_time.strftime("%Y-%m-%d %H:%M:%S"),
        end_time=end_time.strftime("%Y-%m-%d %H:%M:%S"),
        freq="1min",
        disk_cache=False,
    ).reset_index()
    data = data.rename(
        columns={
            "$open": "open",
            "$high": "high",
            "$low": "low",
            "$close": "close",
            "$volume": "volume",
        }
    )
    data["datetime"] = pd.to_datetime(data["datetime"])
    return data.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)


def close_at(data_by_time: pd.DataFrame, timestamp: pd.Timestamp) -> float | None:
    if timestamp not in data_by_time.index:
        return None
    value = data_by_time.at[timestamp, "close"]
    return None if pd.isna(value) else float(value)


def select_candle_interval(
    entry_time: pd.Timestamp, exit_time: pd.Timestamp, context_bars: int
) -> tuple[str, str, pd.Timedelta]:
    holding_duration = exit_time - entry_time
    for rule, label, interval in CANDLE_INTERVALS:
        estimated_candles = holding_duration / interval + 2 * context_bars
        if estimated_candles <= MAX_DETAIL_CANDLES:
            return rule, label, interval
    return CANDLE_INTERVALS[-1]


def get_trade_window(
    trade: pd.Series | dict, context_bars: int
) -> tuple[pd.Timestamp, pd.Timestamp, str, str, pd.Timedelta]:
    entry_time = pd.Timestamp(trade["entry_time"])
    exit_time = pd.Timestamp(trade["exit_time"])
    rule, label, interval = select_candle_interval(entry_time, exit_time, context_bars)
    context_duration = context_bars * interval
    return entry_time - context_duration, exit_time + context_duration, rule, label, interval


def resample_ohlcv(data: pd.DataFrame, rule: str) -> pd.DataFrame:
    if rule == "1min":
        return data.copy()
    return (
        data.set_index("datetime")
        .resample(rule, label="left", closed="left")
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .dropna(subset=["open", "high", "low", "close"])
        .reset_index()
    )


def add_price_context(trades: pd.DataFrame, market: pd.DataFrame, fee_rate: float, slippage_rate: float) -> pd.DataFrame:
    market_by_time = market.set_index("datetime").sort_index()
    enriched = trades.copy()
    numeric_columns = [
        "buy_before_close",
        "buy_close",
        "buy_after_close",
        "sell_before_close",
        "sell_close",
        "sell_after_close",
        "buy_reference_price",
        "sell_reference_price",
        "buy_effective_price",
        "sell_effective_price",
    ]
    for column in numeric_columns:
        enriched[column] = np.nan

    for index, trade in enriched.iterrows():
        entry_time = pd.Timestamp(trade["entry_time"])
        exit_time = pd.Timestamp(trade["exit_time"])
        buy_before = close_at(market_by_time, entry_time - ONE_MINUTE)
        buy_close = close_at(market_by_time, entry_time)
        buy_after = close_at(market_by_time, entry_time + ONE_MINUTE)
        sell_before = close_at(market_by_time, exit_time - ONE_MINUTE)
        sell_close = close_at(market_by_time, exit_time)
        sell_after = close_at(market_by_time, exit_time + ONE_MINUTE)
        buy_reference = trade.get("entry_reference_price", buy_close)
        sell_reference = trade.get("exit_reference_price", sell_close)
        buy_reference = buy_close if pd.isna(buy_reference) else float(buy_reference)
        sell_reference = sell_close if pd.isna(sell_reference) else float(sell_reference)
        enriched.loc[index, "buy_before_time"] = entry_time - ONE_MINUTE
        enriched.loc[index, "buy_after_time"] = entry_time + ONE_MINUTE
        enriched.loc[index, "sell_before_time"] = exit_time - ONE_MINUTE
        enriched.loc[index, "sell_after_time"] = exit_time + ONE_MINUTE
        enriched.loc[index, "buy_before_close"] = buy_before
        enriched.loc[index, "buy_close"] = buy_close
        enriched.loc[index, "buy_after_close"] = buy_after
        enriched.loc[index, "sell_before_close"] = sell_before
        enriched.loc[index, "sell_close"] = sell_close
        enriched.loc[index, "sell_after_close"] = sell_after
        enriched.loc[index, "buy_reference_price"] = buy_reference
        enriched.loc[index, "sell_reference_price"] = sell_reference
        if buy_reference is not None:
            enriched.loc[index, "buy_effective_price"] = buy_reference * (1 + slippage_rate)
        if sell_reference is not None:
            enriched.loc[index, "sell_effective_price"] = sell_reference * (1 - slippage_rate)

    enriched["before_to_buy_return"] = enriched["buy_close"] / enriched["buy_before_close"] - 1
    enriched["buy_to_after_return"] = enriched["buy_after_close"] / enriched["buy_close"] - 1
    enriched["before_to_sell_return"] = enriched["sell_close"] / enriched["sell_before_close"] - 1
    enriched["sell_to_after_return"] = enriched["sell_after_close"] / enriched["sell_close"] - 1
    enriched["fee_rate_per_side"] = fee_rate
    enriched["slippage_rate_per_side"] = slippage_rate
    return enriched


def build_overview(market: pd.DataFrame, trades: pd.DataFrame) -> go.Figure:
    chart_data = market.set_index("datetime")["close"].resample("15min").last().dropna().reset_index()
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.72, 0.28],
        subplot_titles=("BTC/USDT 价格与交易点（15 分钟聚合显示）", "每笔交易净值变化"),
    )
    figure.add_trace(
        go.Scattergl(
            x=chart_data["datetime"],
            y=chart_data["close"],
            mode="lines",
            name="BTC/USDT 收盘价",
            line={"color": "#334155", "width": 1.2},
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>收盘价=%{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=trades["entry_time"],
            y=trades["buy_reference_price"],
            mode="markers+text",
            text=[f"买{i + 1}" for i in range(len(trades))],
            textposition="top center",
            name="买入",
            marker={"symbol": "triangle-up", "size": 12, "color": "#16a34a", "line": {"width": 1, "color": "#14532d"}},
            customdata=trades[
                ["buy_before_close", "buy_close", "buy_after_close", "buy_effective_price"]
            ],
            hovertemplate=(
                "买入时间=%{x|%Y-%m-%d %H:%M}<br>"
                "买前收盘=%{customdata[0]:.2f}<br>K线收盘=%{customdata[1]:.2f}<br>"
                "回测参考价=%{y:.2f}<br>含滑点成交价=%{customdata[3]:.2f}<br>"
                "买后收盘=%{customdata[2]:.2f}<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=trades["exit_time"],
            y=trades["sell_reference_price"],
            mode="markers+text",
            text=[f"卖{i + 1}" for i in range(len(trades))],
            textposition="bottom center",
            name="卖出",
            marker={"symbol": "triangle-down", "size": 12, "color": "#dc2626", "line": {"width": 1, "color": "#7f1d1d"}},
            customdata=trades[
                ["sell_before_close", "sell_close", "sell_after_close", "sell_effective_price"]
            ],
            hovertemplate=(
                "卖出时间=%{x|%Y-%m-%d %H:%M}<br>"
                "卖前收盘=%{customdata[0]:.2f}<br>K线收盘=%{customdata[1]:.2f}<br>"
                "回测参考价=%{y:.2f}<br>含滑点成交价=%{customdata[3]:.2f}<br>"
                "卖后收盘=%{customdata[2]:.2f}<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=trades["exit_time"],
            y=trades["equity"],
            mode="lines+markers",
            name="策略净值",
            line={"color": "#2563eb", "width": 2},
            marker={"size": 7},
            hovertemplate="平仓时间=%{x|%Y-%m-%d %H:%M}<br>净值=%{y:.6f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    figure.update_yaxes(title_text="价格（USDT）", row=1, col=1)
    figure.update_yaxes(title_text="净值", row=2, col=1)
    figure.update_layout(
        height=820,
        template="plotly_white",
        hovermode="closest",
        legend={"orientation": "h", "y": 1.03, "x": 0},
        margin={"l": 70, "r": 40, "t": 100, "b": 50},
    )
    return figure


def build_detail(market: pd.DataFrame, trades: pd.DataFrame, window_bars: int) -> go.Figure:
    figure = go.Figure()
    if trades.empty:
        figure.add_annotation(
            text="没有预测收益能够覆盖交易成本与安全边际，因此本次回测未开仓。",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"size": 18, "color": "#475569"},
        )
        figure.update_layout(
            title="交易明细",
            height=360,
            template="plotly_white",
            xaxis={"visible": False},
            yaxis={"visible": False},
            margin={"l": 40, "r": 40, "t": 80, "b": 40},
        )
        return figure

    trace_groups: list[list[int]] = []
    trade_layouts = []
    for index, trade in trades.iterrows():
        entry_time = pd.Timestamp(trade["entry_time"])
        exit_time = pd.Timestamp(trade["exit_time"])
        window_start, window_end, rule, interval_label, _ = get_trade_window(trade, window_bars)
        window = market[
            (market["datetime"] >= window_start) & (market["datetime"] <= window_end)
        ]
        candles = resample_ohlcv(window, rule)
        if candles.empty:
            raise ValueError(f"No candles are available for trade {index + 1}")

        price_min = float(candles["low"].min())
        price_max = float(candles["high"].max())
        price_padding = max((price_max - price_min) * 0.08, price_max * 0.001)
        y_range = [price_min - price_padding, price_max + price_padding]
        holding_duration = exit_time - entry_time
        holding_hours = holding_duration.total_seconds() / 3600
        duration_label = f"{holding_hours / 24:.1f} 天" if holding_hours >= 48 else f"{holding_hours:.1f} 小时"
        group: list[int] = []
        figure.add_trace(
            go.Candlestick(
                x=candles["datetime"],
                open=candles["open"],
                high=candles["high"],
                low=candles["low"],
                close=candles["close"],
                name=f"第 {index + 1} 笔 K 线",
                increasing_line_color="#16a34a",
                decreasing_line_color="#dc2626",
                increasing_fillcolor="#bbf7d0",
                decreasing_fillcolor="#fecaca",
                visible=index == 0,
                showlegend=False,
                hoverlabel={"namelength": -1},
            )
        )
        group.append(len(figure.data) - 1)

        for event_time, event_price, effective_price, label, color, symbol, text_position in [
            (
                entry_time,
                float(trade["buy_reference_price"]),
                float(trade["buy_effective_price"]),
                "买入",
                "#16a34a",
                "triangle-up",
                "top center",
            ),
            (
                exit_time,
                float(trade["sell_reference_price"]),
                float(trade["sell_effective_price"]),
                "卖出",
                "#dc2626",
                "triangle-down",
                "bottom center",
            ),
        ]:
            figure.add_trace(
                go.Scatter(
                    x=[event_time, event_time],
                    y=y_range,
                    mode="lines",
                    name=f"{label}时间",
                    line={"color": color, "width": 1.5, "dash": "dot"},
                    visible=index == 0,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            group.append(len(figure.data) - 1)
            figure.add_trace(
                go.Scatter(
                    x=[event_time],
                    y=[event_price],
                    mode="markers+text",
                    text=[label],
                    textposition=text_position,
                    name=label,
                    marker={
                        "symbol": symbol,
                        "size": 15,
                        "color": color,
                        "line": {"width": 1, "color": "#0f172a"},
                    },
                    customdata=[[effective_price]],
                    hovertemplate=(
                        f"{label}时间=%{{x|%Y-%m-%d %H:%M}}<br>"
                        "回测参考价=%{y:.2f}<br>含滑点成交价=%{customdata[0]:.2f}<extra></extra>"
                    ),
                    visible=index == 0,
                    showlegend=False,
                )
            )
            group.append(len(figure.data) - 1)
        trace_groups.append(group)
        trade_layouts.append(
            {
                "title": (
                    f"第 {index + 1} 笔：买入到卖出完整 K 线 | "
                    f"{interval_label} K 线 | 持仓 {duration_label} | 前后各 {window_bars} 根"
                ),
                "x_range": [window_start, window_end],
                "y_range": y_range,
            }
        )

    buttons = []
    for index, (trade, group, trade_layout) in enumerate(
        zip(trades.to_dict("records"), trace_groups, trade_layouts)
    ):
        visibility = [False] * len(figure.data)
        for trace_index in group:
            visibility[trace_index] = True
        buttons.append(
            {
                "label": f"第 {index + 1} 笔：{trade['entry_time']} 买入",
                "method": "update",
                "args": [
                    {"visible": visibility},
                    {
                        "title": {"text": trade_layout["title"]},
                        "xaxis": {"range": trade_layout["x_range"], "autorange": False},
                        "yaxis": {"range": trade_layout["y_range"], "autorange": False},
                    },
                ],
            }
        )
    first_layout = trade_layouts[0]
    figure.update_xaxes(
        title_text="时间",
        rangeslider_visible=False,
        range=first_layout["x_range"],
        autorange=False,
    )
    figure.update_yaxes(
        title_text="价格（USDT）",
        range=first_layout["y_range"],
        autorange=False,
    )
    figure.update_layout(
        title={"text": first_layout["title"]},
        height=650,
        template="plotly_white",
        hovermode="closest",
        updatemenus=[
            {
                "type": "dropdown",
                "direction": "down",
                "x": 0.5,
                "xanchor": "center",
                "y": 1.16,
                "yanchor": "top",
                "buttons": buttons,
            }
        ],
        margin={"l": 75, "r": 40, "t": 120, "b": 60},
    )
    return figure


def build_table(trades: pd.DataFrame) -> go.Figure:
    display = trades.copy()
    display["exit_reason"] = display["exit_reason"].map(EXIT_REASON_LABELS).fillna(display["exit_reason"])
    display.insert(0, "trade", [f"第 {index + 1} 笔" for index in range(len(display))])
    columns = [
        ("trade", "交易"),
        ("entry_time", "买入时间"),
        ("buy_before_close", "买前价格"),
        ("buy_close", "买入 K 线收盘价"),
        ("buy_reference_price", "买入参考价"),
        ("buy_effective_price", "买入成交价（含滑点）"),
        ("buy_after_close", "买后价格"),
        ("exit_time", "卖出时间"),
        ("sell_before_close", "卖前价格"),
        ("sell_close", "卖出 K 线收盘价"),
        ("sell_reference_price", "卖出参考价"),
        ("sell_effective_price", "卖出成交价（含滑点）"),
        ("sell_after_close", "卖后价格"),
        ("exit_reason", "清仓原因"),
        ("holding_bars", "持仓分钟"),
        ("return", "交易收益"),
        ("equity", "交易后净值"),
    ]
    cell_values = []
    for field, _ in columns:
        values = display[field]
        if field.endswith("_time"):
            values = pd.to_datetime(values).dt.strftime("%Y-%m-%d %H:%M")
        elif field == "return":
            values = values.map(lambda value: "" if pd.isna(value) else f"{value:.2%}")
        elif field == "equity":
            values = values.map(lambda value: "" if pd.isna(value) else f"{value:.6f}")
        elif pd.api.types.is_numeric_dtype(values):
            values = values.map(lambda value: "" if pd.isna(value) else f"{value:.2f}")
        cell_values.append(values.tolist())
    figure = go.Figure(
        data=[
            go.Table(
                header={"values": [label for _, label in columns], "fill_color": "#1e293b", "font": {"color": "white", "size": 12}},
                cells={"values": cell_values, "fill_color": [["#f8fafc", "#e2e8f0"] * len(display)], "font": {"size": 11}},
            )
        ]
    )
    figure.update_layout(title="买卖前后价格明细", height=430, margin={"l": 10, "r": 10, "t": 60, "b": 20})
    return figure


def write_report(output_dir: Path, overview: go.Figure, detail: go.Figure, table: go.Figure, trades: pd.DataFrame, report: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    enriched_path = output_dir / "trades_with_prices.csv"
    trades.to_csv(enriched_path, index=False)
    report_path = output_dir / "trades_price_report.html"
    overview_html = pio.to_html(overview, full_html=False, include_plotlyjs=True)
    detail_html = pio.to_html(detail, full_html=False, include_plotlyjs=False)
    table_html = pio.to_html(table, full_html=False, include_plotlyjs=False)
    backtest = report.get("backtest", {})
    horizon_bars = int(backtest.get("horizon_bars", 0))
    average_holding = float(backtest.get("average_holding_bars", 0.0))
    median_holding = float(backtest.get("median_holding_bars", 0.0))
    p90_holding = float(backtest.get("p90_holding_bars", 0.0))
    max_holding = int(backtest.get("max_holding_bars", 0))
    entry_threshold = backtest.get("entry_prediction_threshold")
    exit_reason_counts = backtest.get("exit_reason_counts", {})
    exit_reasons = "、".join(
        f"{EXIT_REASON_LABELS.get(str(reason), reason)} {int(count)} 次"
        for reason, count in exit_reason_counts.items()
    ) or "暂无清仓记录"
    summary = (
        f"<div class='summary'>"
        f"<div><b>标的</b><span>BTC/USDT</span></div>"
        f"<div><b>交易笔数</b><span>{len(trades)}</span></div>"
        f"<div><b>测试集收益</b><span>{backtest.get('total_return', 0):.2%}</span></div>"
        f"<div><b>买入持有</b><span>{backtest.get('buy_and_hold_return', 0):.2%}</span></div>"
        f"<div><b>预测 / 到期窗口</b><span>{horizon_bars} 分钟</span></div>"
        f"<div><b>平均持仓</b><span>{average_holding:.1f} 分钟</span></div>"
        f"<div><b>中位 / P90 持仓</b><span>{median_holding:.1f} / {p90_holding:.1f} 分钟</span></div>"
        f"<div><b>最大持仓 / 续期</b><span>{max_holding} 分钟 / {backtest.get('total_renewals', 0)} 次</span></div>"
        f"</div>"
        f"<p class='report-meta'>开仓预测阈值："
        f"{float(entry_threshold):.4%}；清仓原因：{exit_reasons}</p>"
    )
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BTC/USDT 买卖前后价格报告</title>
<style>
body {{ margin: 0; background: #f1f5f9; color: #0f172a; font-family: "Microsoft YaHei", "Segoe UI", sans-serif; }}
main {{ max-width: 1500px; margin: 0 auto; padding: 28px 24px 48px; }}
h1 {{ margin: 0 0 8px; font-size: 28px; }}
p {{ color: #475569; margin: 0 0 18px; }}
.summary {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 18px 0; }}
.summary div {{ background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px 16px; }}
.summary b, .summary span {{ display: block; }}
.summary b {{ color: #64748b; font-size: 12px; font-weight: 600; margin-bottom: 6px; }}
.summary span {{ font-size: 20px; font-weight: 700; }}
.panel {{ background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 8px; margin-top: 18px; overflow: hidden; }}
@media (max-width: 800px) {{ .summary {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} main {{ padding: 18px 10px 32px; }} h1 {{ font-size: 22px; }} }}
</style>
</head>
<body><main>
<h1>BTC/USDT 买卖前后价格</h1>
<p>绿色向上三角为买入，红色向下三角为卖出；止损成交点使用回测参考价，普通成交点使用下一根 1 分钟 K 线收盘价。</p>
{summary}
<section class="panel">{overview_html}</section>
<section class="panel">{detail_html}</section>
<section class="panel">{table_html}</section>
</main></body></html>"""
    report_path.write_text(html, encoding="utf-8")
    return report_path


def main() -> None:
    args = parse_args()
    if args.window_bars < 1:
        raise ValueError("window-bars must be positive")
    trades = pd.read_csv(args.trades_path, parse_dates=["signal_time", "entry_time", "exit_time"])
    report_path = args.trades_path.with_name("report.json")
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    backtest = report.get("backtest", {})
    fee_rate = float(backtest.get("fee_rate_per_side", 0.0))
    slippage_rate = float(backtest.get("slippage_rate_per_side", 0.0))
    if trades.empty:
        test_period = report.get("test", {})
        if not test_period.get("start") or not test_period.get("end"):
            raise ValueError("The report must contain the test period when there are no trades")
        start_time = pd.Timestamp(test_period["start"])
        end_time = pd.Timestamp(test_period["end"])
    else:
        trade_windows = [get_trade_window(trade, args.window_bars) for _, trade in trades.iterrows()]
        start_time = min(window[0] for window in trade_windows) - 2 * ONE_MINUTE
        end_time = max(window[1] for window in trade_windows) + 2 * ONE_MINUTE
    market = load_ohlcv(args.provider_uri, start_time, end_time)
    if market.empty:
        raise ValueError("No market data was loaded for the trade timestamps")
    enriched = add_price_context(trades, market, fee_rate, slippage_rate)
    overview = build_overview(market, enriched)
    detail = build_detail(market, enriched, args.window_bars)
    table = build_table(enriched)
    output_path = write_report(args.output_dir, overview, detail, table, enriched, report)
    print(f"Price-enriched trades: {args.output_dir / 'trades_with_prices.csv'}")
    print(f"Interactive report: {output_path}")


if __name__ == "__main__":
    main()