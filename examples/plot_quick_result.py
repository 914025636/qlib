from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


EXAMPLES_DIR = Path(__file__).resolve().parent
MLRUNS_DIR = EXAMPLES_DIR.parent / "mlruns"
OUTPUT_PATH = EXAMPLES_DIR / "output" / "lightgbm_quick_report.html"


def find_latest_artifacts() -> Path:
    candidates = []
    for report_path in MLRUNS_DIR.glob("*/*/artifacts/portfolio_analysis/report_normal_1day.pkl"):
        artifacts_dir = report_path.parents[1]
        if (artifacts_dir / "pred.pkl").exists() and (artifacts_dir / "label.pkl").exists():
            candidates.append((report_path.stat().st_mtime, artifacts_dir))
    if not candidates:
        raise FileNotFoundError(f"No completed Qlib run found under {MLRUNS_DIR}")
    return max(candidates)[1]


def build_report(artifacts_dir: Path) -> go.Figure:
    report = pd.read_pickle(artifacts_dir / "portfolio_analysis" / "report_normal_1day.pkl")
    prediction = pd.read_pickle(artifacts_dir / "pred.pkl")
    label = pd.read_pickle(artifacts_dir / "label.pkl")

    strategy_return = (1 + report["return"].fillna(0)).cumprod() - 1
    benchmark_return = (1 + report["bench"].fillna(0)).cumprod() - 1
    net_return = (1 + report["return"].fillna(0) - report["cost"].fillna(0)).cumprod() - 1
    excess_return = (1 + report["return"].fillna(0) - report["cost"].fillna(0) - report["bench"].fillna(0)).cumprod() - 1
    net_value = 1 + net_return
    drawdown = net_value / net_value.cummax() - 1

    pred_label = prediction.join(label, how="inner")
    daily_ic = pred_label.groupby(level="datetime").apply(
        lambda frame: frame["score"].corr(frame["LABEL0"]), include_groups=False
    )

    figure = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=("累计收益", "扣费后超额收益", "回撤与换手率", "每日 IC"),
        specs=[[{}], [{}], [{"secondary_y": True}], [{}]],
    )
    figure.add_trace(go.Scatter(x=report.index, y=strategy_return, name="策略收益"), row=1, col=1)
    figure.add_trace(go.Scatter(x=report.index, y=net_return, name="策略收益（扣费）"), row=1, col=1)
    figure.add_trace(go.Scatter(x=report.index, y=benchmark_return, name="CSI300"), row=1, col=1)
    figure.add_trace(
        go.Scatter(x=report.index, y=excess_return, name="扣费后超额", line={"color": "#0f766e"}),
        row=2,
        col=1,
    )
    figure.add_hline(y=0, line_width=1, line_color="#94a3b8", row=2, col=1)
    figure.add_trace(
        go.Scatter(x=report.index, y=drawdown, name="回撤", fill="tozeroy", line={"color": "#dc2626"}),
        row=3,
        col=1,
        secondary_y=False,
    )
    figure.add_trace(
        go.Bar(x=report.index, y=report["turnover"], name="换手率", opacity=0.35, marker_color="#f59e0b"),
        row=3,
        col=1,
        secondary_y=True,
    )
    figure.add_trace(
        go.Bar(x=daily_ic.index, y=daily_ic, name="每日 IC", marker_color="#2563eb"),
        row=4,
        col=1,
    )
    figure.add_hline(y=0, line_width=1, line_color="#94a3b8", row=4, col=1)

    figure.update_yaxes(tickformat=".1%", row=1, col=1)
    figure.update_yaxes(tickformat=".1%", row=2, col=1)
    figure.update_yaxes(tickformat=".1%", row=3, col=1, secondary_y=False)
    figure.update_yaxes(tickformat=".1%", row=3, col=1, secondary_y=True)
    figure.update_layout(
        title="Qlib LightGBM 快速测试报告",
        height=1050,
        template="plotly_white",
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.03, "x": 0},
        margin={"l": 70, "r": 70, "t": 100, "b": 50},
    )
    return figure


def main() -> None:
    artifacts_dir = find_latest_artifacts()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    build_report(artifacts_dir).write_html(OUTPUT_PATH, include_plotlyjs=True)
    print(f"Recorder artifacts: {artifacts_dir}")
    print(f"Report written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
