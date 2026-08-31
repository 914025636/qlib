"""Build an interactive comparison dashboard for the Binance experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_15 = ROOT / ".qlib" / "experiments" / "binance_btc_usdt_1m"
DEFAULT_MODEL_60 = (
    ROOT
    / ".qlib"
    / "experiments"
    / "binance_btc_usdt_1m_h60_be_min_edge0_20260830"
)
DEFAULT_FACTOR_15 = ROOT / ".qlib" / "experiments" / "squeeze_breakout_factor"
DEFAULT_FACTOR_60 = ROOT / ".qlib" / "experiments" / "squeeze_breakout_factor_h60"
DEFAULT_OUTPUT = ROOT / ".qlib" / "experiments" / "binance_experiment_comparison"
HTML_NAME = "binance_experiment_comparison.html"
JSON_NAME = "binance_experiment_comparison.json"
COLORS = {
    "benchmark": "#b66708",
    "factor15": "#147d64",
    "factor60": "#2563a6",
    "model15": "#8b5cf6",
    "model60": "#c2414b",
    "muted": "#64748b",
    "grid": "#e2e8f0",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-15-dir", type=Path, default=DEFAULT_MODEL_15)
    parser.add_argument("--model-60-dir", type=Path, default=DEFAULT_MODEL_60)
    parser.add_argument("--factor-15-dir", type=Path, default=DEFAULT_FACTOR_15)
    parser.add_argument("--factor-60-dir", type=Path, default=DEFAULT_FACTOR_60)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def number(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def integer(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def load_model_experiment(directory: Path, label: str) -> dict[str, object]:
    report = read_json(directory / "report.json")
    backtest = report.get("backtest", {})
    test = report.get("test", {})
    model = report.get("model", {})
    return {
        "label": label,
        "kind": "模型回测",
        "horizon_bars": integer(backtest.get("horizon_bars")),
        "period_start": str(test.get("start", "")),
        "period_end": str(test.get("end", "")),
        "total_return": number(backtest.get("total_return")),
        "buy_hold_return": number(backtest.get("buy_and_hold_return")),
        "max_drawdown": number(backtest.get("max_drawdown")),
        "trades": integer(backtest.get("trades")),
        "win_rate": number(backtest.get("win_rate")),
        "long_exposure": number(backtest.get("long_exposure")),
        "average_holding_bars": number(backtest.get("average_holding_bars")),
        "median_holding_bars": number(backtest.get("median_holding_bars")),
        "p90_holding_bars": number(backtest.get("p90_holding_bars")),
        "prediction_mean": number(test.get("prediction_mean")),
        "directional_accuracy": number(test.get("directional_accuracy")),
        "entry_threshold": number(backtest.get("entry_prediction_threshold")),
        "best_iteration": integer(model.get("best_iteration")),
    }


def load_factor_experiment(directory: Path, label: str) -> dict[str, object]:
    metrics = read_json(directory / "squeeze_breakout_metrics.json")
    return {
        "label": label,
        "kind": "因子事件回测",
        "horizon_bars": integer(metrics.get("horizon_bars")),
        "period_start": str(metrics.get("period_start", "")),
        "period_end": str(metrics.get("period_end", "")),
        "total_return": number(metrics.get("total_return")),
        "buy_hold_return": number(metrics.get("buy_hold_return")),
        "max_drawdown": number(metrics.get("max_drawdown")),
        "trades": integer(metrics.get("executed_trades")),
        "win_rate": number(metrics.get("win_rate")),
        "long_exposure": number(metrics.get("long_exposure")),
        "average_holding_bars": number(metrics.get("average_holding_bars")),
        "median_holding_bars": number(metrics.get("median_holding_bars")),
        "p90_holding_bars": number(metrics.get("p90_holding_bars")),
        "prediction_mean": None,
        "directional_accuracy": None,
        "entry_threshold": None,
        "best_iteration": None,
    }


def load_curve(directory: Path) -> pd.DataFrame:
    curve_path = directory / "squeeze_breakout_equity.csv"
    curve = pd.read_csv(curve_path, parse_dates=["datetime"])
    return (
        curve.set_index("datetime")
        .resample("1h")
        .last()
        .dropna(subset=["equity", "buy_hold"])
        .reset_index()
    )


def build_equity_figure(curve_15: pd.DataFrame, curve_60: pd.DataFrame) -> go.Figure:
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.68, 0.32],
        subplot_titles=("测试集净值（归一化）", "回撤"),
    )
    for curve, name, color in [
        (curve_15, "因子 15 分钟", COLORS["factor15"]),
        (curve_60, "因子 60 分钟", COLORS["factor60"]),
    ]:
        figure.add_trace(
            go.Scattergl(
                x=curve["datetime"],
                y=curve["equity"],
                mode="lines",
                name=name,
                line={"color": color, "width": 2},
                hovertemplate=f"{name}<br>%{{x|%Y-%m-%d %H:%M}}<br>净值=%{{y:.4f}}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    figure.add_trace(
        go.Scattergl(
            x=curve_60["datetime"],
            y=curve_60["buy_hold"],
            mode="lines",
            name="买入持有",
            line={"color": COLORS["benchmark"], "width": 2, "dash": "dash"},
            hovertemplate="买入持有<br>%{x|%Y-%m-%d %H:%M}<br>净值=%{y:.4f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    for curve, name, color in [
        (curve_15, "因子 15 分钟回撤", COLORS["factor15"]),
        (curve_60, "因子 60 分钟回撤", COLORS["factor60"]),
    ]:
        figure.add_trace(
            go.Scattergl(
                x=curve["datetime"],
                y=curve["drawdown"],
                mode="lines",
                name=name,
                line={"color": color, "width": 1.5},
                showlegend=False,
                hovertemplate=f"{name}<br>%{{x|%Y-%m-%d %H:%M}}<br>回撤=%{{y:.2%}}<extra></extra>",
            ),
            row=2,
            col=1,
        )
    figure.add_trace(
        go.Scatter(
            x=curve_60["datetime"],
            y=[0.0] * len(curve_60),
            mode="lines",
            name="模型版本（均无交易）",
            line={"color": COLORS["model60"], "width": 1.3, "dash": "dot"},
            hovertemplate="模型回测<br>%{x|%Y-%m-%d %H:%M}<br>回撤=0.00%<extra></extra>",
        ),
        row=2,
        col=1,
    )
    figure.update_yaxes(title_text="净值", row=1, col=1)
    figure.update_yaxes(title_text="回撤", tickformat=".1%", row=2, col=1)
    figure.update_xaxes(title_text="时间", row=2, col=1)
    figure.update_layout(
        height=760,
        template="plotly_white",
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.06, "x": 0},
        margin={"l": 68, "r": 32, "t": 92, "b": 54},
    )
    return figure


def build_metric_figure(experiments: list[dict[str, object]]) -> go.Figure:
    labels = [str(item["label"]) for item in experiments]
    returns = [number(item["total_return"]) for item in experiments]
    trades = [integer(item["trades"]) for item in experiments]
    exposure = [number(item["long_exposure"]) for item in experiments]
    colors = [
        COLORS["model15"],
        COLORS["model60"],
        COLORS["factor15"],
        COLORS["factor60"],
    ]
    figure = make_subplots(
        rows=1,
        cols=3,
        horizontal_spacing=0.09,
        subplot_titles=("总收益", "交易笔数", "多头暴露"),
    )
    figure.add_trace(
        go.Bar(
            x=labels,
            y=returns,
            marker_color=colors,
            name="总收益",
            text=[f"{value:.2%}" for value in returns],
            textposition="outside",
            hovertemplate="%{x}<br>总收益=%{y:.2%}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Bar(
            x=labels,
            y=trades,
            marker_color=colors,
            name="交易笔数",
            text=[str(value) for value in trades],
            textposition="outside",
            hovertemplate="%{x}<br>交易笔数=%{y}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    figure.add_trace(
        go.Bar(
            x=labels,
            y=exposure,
            marker_color=colors,
            name="多头暴露",
            text=[f"{value:.1%}" for value in exposure],
            textposition="outside",
            hovertemplate="%{x}<br>多头暴露=%{y:.2%}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=3,
    )
    figure.add_hline(y=0, line={"color": COLORS["muted"], "dash": "dot"}, row=1, col=1)
    figure.update_yaxes(tickformat=".0%", row=1, col=1)
    figure.update_yaxes(title_text="笔数", row=1, col=2)
    figure.update_yaxes(tickformat=".0%", row=1, col=3)
    figure.update_layout(
        height=430,
        template="plotly_white",
        showlegend=False,
        margin={"l": 58, "r": 24, "t": 78, "b": 120},
    )
    return figure


def build_signal_figure(model_experiments: list[dict[str, object]]) -> go.Figure:
    labels = [str(item["label"]) for item in model_experiments]
    prediction_bp = [number(item["prediction_mean"]) * 10_000 for item in model_experiments]
    threshold_bp = [number(item["entry_threshold"]) * 10_000 for item in model_experiments]
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=labels,
            y=prediction_bp,
            name="测试集平均预测",
            marker_color=[COLORS["model15"], COLORS["model60"]],
            text=[f"{value:.3f} bp" for value in prediction_bp],
            textposition="outside",
            hovertemplate="%{x}<br>平均预测=%{y:.3f} bp<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=labels,
            y=threshold_bp,
            mode="lines+markers",
            name="开仓阈值",
            line={"color": COLORS["benchmark"], "width": 2, "dash": "dash"},
            hovertemplate="%{x}<br>开仓阈值=%{y:.3f} bp<extra></extra>",
        )
    )
    figure.update_yaxes(title_text="收益预测（bp）")
    figure.update_layout(
        title="模型预测幅度与成本门槛",
        height=430,
        template="plotly_white",
        hovermode="x unified",
        margin={"l": 68, "r": 24, "t": 78, "b": 70},
    )
    return figure


def format_percent(value: object) -> str:
    return f"{number(value):.2%}"


def format_optional_percent(value: object) -> str:
    return "-" if value is None else format_percent(value)


def format_holding(item: dict[str, object]) -> str:
    if integer(item.get("trades")) == 0:
        return "无交易"
    return (
        f"均值 {number(item.get('average_holding_bars')):.1f} / "
        f"中位 {number(item.get('median_holding_bars')):.1f} / "
        f"P90 {number(item.get('p90_holding_bars')):.1f} 分钟"
    )


def build_table(experiments: list[dict[str, object]]) -> str:
    rows = []
    for item in experiments:
        rows.append(
            "<tr>"
            f"<td>{item['label']}</td>"
            f"<td>{item['kind']}</td>"
            f"<td>{integer(item['horizon_bars'])} 分钟</td>"
            f"<td>{integer(item['trades']):,}</td>"
            f"<td>{format_percent(item['total_return'])}</td>"
            f"<td>{format_percent(item['buy_hold_return'])}</td>"
            f"<td>{format_percent(item['max_drawdown'])}</td>"
            f"<td>{format_optional_percent(item['win_rate'])}</td>"
            f"<td>{format_percent(item['long_exposure'])}</td>"
            f"<td>{format_holding(item)}</td>"
            "</tr>"
        )
    return (
        "<div class='table-wrap'><table><thead><tr>"
        "<th>实验</th><th>类型</th><th>窗口</th><th>交易数</th>"
        "<th>策略收益</th><th>买入持有</th><th>最大回撤</th>"
        "<th>胜率</th><th>多头暴露</th><th>持仓统计</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def build_timeline(experiments: list[dict[str, object]]) -> str:
    period_start = min(pd.Timestamp(item["period_start"]) for item in experiments if item["period_start"])
    period_end = max(pd.Timestamp(item["period_end"]) for item in experiments if item["period_end"])
    return (
        f"<div class='timeline'><span>统一测试区间</span>"
        f"<strong>{period_start:%Y-%m-%d %H:%M} 至 {period_end:%Y-%m-%d %H:%M}</strong>"
        "<div class='timeline-bar'><i></i></div>"
        "<small>所有结果均使用 BTC/USDT 1 分钟数据；手续费单边 0.04%，滑点单边 0.01%。</small></div>"
    )


def build_html(
    output_dir: Path,
    experiments: list[dict[str, object]],
    overview: go.Figure,
    metrics: go.Figure,
    signals: go.Figure,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    overview_html = pio.to_html(overview, full_html=False, include_plotlyjs=True)
    metrics_html = pio.to_html(metrics, full_html=False, include_plotlyjs=False)
    signals_html = pio.to_html(signals, full_html=False, include_plotlyjs=False)
    model_items = [item for item in experiments if item["kind"] == "模型回测"]
    h60_model = next(item for item in model_items if integer(item["horizon_bars"]) == 60)
    threshold_bp = number(h60_model["entry_threshold"]) * 10_000
    prediction_bp = number(h60_model["prediction_mean"]) * 10_000
    if integer(h60_model["trades"]) == 0:
        verdict = (
            f"60 分钟模型没有开仓：测试集平均预测仅 {prediction_bp:.3f} bp，"
            f"低于 {threshold_bp:.3f} bp 的成本门槛。"
        )
        verdict_class = "warning"
    else:
        verdict = "模型回测已产生交易，请结合下方曲线和回撤评估稳定性。"
        verdict_class = "neutral"
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BTC/USDT 实验对比总览</title>
<style>
:root {{ --ink:#17211b; --muted:#647067; --line:#dfe6e0; --paper:#fff; --bg:#f4f7f4; --green:#147d64; --blue:#2563a6; --amber:#b66708; --red:#c2414b; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; color:var(--ink); background:var(--bg); font-family:"Microsoft YaHei UI", "Noto Sans SC", sans-serif; background-image:linear-gradient(rgba(54,76,61,.035) 1px, transparent 1px),linear-gradient(90deg,rgba(54,76,61,.035) 1px,transparent 1px); background-size:28px 28px; }}
main {{ max-width:1560px; margin:0 auto; padding:26px 24px 54px; }}
header {{ display:flex; justify-content:space-between; align-items:end; gap:24px; padding:6px 0 18px; border-bottom:2px solid #2d3b32; }}
h1 {{ margin:0; font-size:30px; line-height:1.2; letter-spacing:0; }}
.subtitle {{ margin:7px 0 0; color:var(--muted); font-size:14px; }}
.stamp {{ text-align:right; color:var(--muted); font-size:13px; line-height:1.7; }}
.timeline {{ margin:18px 0; padding:16px 18px; background:var(--paper); border:1px solid var(--line); border-radius:7px; }}
.timeline span, .timeline strong, .timeline small {{ display:block; }}
.timeline span {{ color:var(--muted); font-size:12px; }}
.timeline strong {{ margin-top:4px; font-size:16px; }}
.timeline-bar {{ height:8px; margin:14px 0 10px; background:#e5e7eb; border-radius:4px; overflow:hidden; }}
.timeline-bar i {{ display:block; width:100%; height:100%; background:linear-gradient(90deg, var(--green) 0 58%, var(--blue) 58% 100%); }}
.timeline small {{ color:var(--muted); font-size:12px; }}
.verdict {{ margin:14px 0 20px; padding:13px 16px; border-left:4px solid; background:var(--paper); font-size:14px; }}
.verdict.warning {{ border-color:var(--amber); }} .verdict.neutral {{ border-color:var(--blue); }}
.section {{ margin:18px 0; background:var(--paper); border:1px solid var(--line); border-radius:7px; overflow:hidden; }}
.section-head {{ display:flex; justify-content:space-between; align-items:baseline; gap:18px; padding:17px 20px 0; }}
.section-head h2 {{ margin:0; font-size:19px; }}
.section-head p {{ margin:0; color:var(--muted); font-size:12px; }}
.plot {{ padding:4px 8px 8px; }}
.table-wrap {{ overflow:auto; margin:16px 18px 20px; border:1px solid var(--line); }}
table {{ width:100%; border-collapse:collapse; min-width:1080px; white-space:nowrap; font-size:12px; font-variant-numeric:tabular-nums; }}
th {{ position:sticky; top:0; color:#fff; background:#34473b; text-align:right; padding:10px 11px; }}
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align:left; }}
td {{ padding:9px 11px; border-bottom:1px solid #edf1ed; text-align:right; }}
tr:nth-child(even) td {{ background:#f8faf8; }}
.notes {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; padding:0 18px 20px; }}
.note {{ padding:14px 16px; border-top:3px solid var(--blue); background:#f8fafc; }}
.note h3 {{ margin:0 0 7px; font-size:14px; }}
.note p {{ margin:0; color:#475569; font-size:13px; line-height:1.65; }}
@media(max-width:900px) {{ main {{ padding:15px 9px 36px; }} header {{ display:block; }} .stamp {{ margin-top:8px; text-align:left; }} h1 {{ font-size:24px; }} .notes {{ grid-template-columns:1fr; }} .section-head {{ display:block; }} .section-head p {{ margin-top:5px; }} }}
</style>
</head>
<body><main>
<header><div><h1>BTC/USDT 实验对比总览</h1><p class="subtitle">1 分钟数据 · 模型回测与突破因子事件回测</p></div><div class="stamp">手续费单边 0.04%<br>滑点单边 0.01%</div></header>
{build_timeline(experiments)}
<div class="verdict {verdict_class}">{verdict}</div>
<section class="section"><div class="section-head"><h2>净值与回撤</h2><p>因子曲线按小时聚合；模型版本无交易时净值保持 1.0</p></div><div class="plot">{overview_html}</div></section>
<section class="section"><div class="section-head"><h2>关键回测指标</h2><p>正收益在收益图中位于零线以上</p></div><div class="plot">{metrics_html}</div></section>
<section class="section"><div class="section-head"><h2>模型信号诊断</h2><p>平均预测必须先覆盖手续费、滑点和开仓门槛</p></div><div class="plot">{signals_html}</div></section>
<section class="section"><div class="section-head"><h2>实验明细</h2><p>测试集指标均来自已落盘报告</p></div>{build_table(experiments)}</section>
<div class="notes"><div class="note"><h3>模型版本</h3><p>15 分钟和 60 分钟 LightGBM 模型均未产生交易。60 分钟版本的平均预测约为 {prediction_bp:.3f} bp，而完整交易成本门槛约为 {threshold_bp:.3f} bp，因此零交易是成本过滤的结果。</p></div><div class="note"><h3>因子版本</h3><p>直接固定持有 15 或 60 分钟的突破事件回测均为负收益。延长持有周期本身没有改善信号质量，应优先改进标签、模型和入场过滤。</p></div></div>
</main></body></html>"""
    html_path = output_dir / HTML_NAME
    html_path.write_text(html, encoding="utf-8")
    return html_path


def main() -> None:
    args = parse_args()
    experiments = [
        load_model_experiment(args.model_15_dir, "模型 15 分钟"),
        load_model_experiment(args.model_60_dir, "模型 60 分钟"),
        load_factor_experiment(args.factor_15_dir, "因子 15 分钟"),
        load_factor_experiment(args.factor_60_dir, "因子 60 分钟"),
    ]
    curve_15 = load_curve(args.factor_15_dir)
    curve_60 = load_curve(args.factor_60_dir)
    overview = build_equity_figure(curve_15, curve_60)
    metrics = build_metric_figure(experiments)
    signals = build_signal_figure(experiments[:2])
    output_path = build_html(args.output_dir, experiments, overview, metrics, signals)
    summary = {
        "experiments": experiments,
        "output": str(output_path.resolve()),
    }
    (args.output_dir / JSON_NAME).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Interactive comparison: {output_path.resolve()}")


if __name__ == "__main__":
    main()