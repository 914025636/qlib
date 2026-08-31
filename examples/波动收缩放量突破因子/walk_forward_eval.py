"""波动收缩放量突破因子的 purged walk-forward 滚动评估。

替代固定三段切分：在每个 fold 的 train 段搜索阈值、valid 段选参、test 段只评估一次，
最后合并所有互不重叠的 test 段交易，给出 alpha 的置信区间并与双边成本比较。

用法：
    python "examples\\波动收缩放量突破因子\\walk_forward_eval.py" --horizon-bars 60
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

EXAMPLES_DIR = Path(__file__).resolve().parents[1]
QLIB_ROOT = EXAMPLES_DIR.parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from btc_minute.config import SplitConfig  # noqa: E402
from btc_minute.labels import Fold, walk_forward_folds  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_backtest import load_factor_module, select_non_overlapping_trades  # noqa: E402

DEFAULT_PROVIDER = QLIB_ROOT / ".qlib" / "qlib_data" / "binance_btc_usdt_1m"
DEFAULT_OUTPUT = QLIB_ROOT / ".qlib" / "experiments" / "squeeze_breakout_walk_forward"
MINUTES_PER_MONTH = 60 * 24 * 30

# 信号在 t 根发出，t+2 根开盘成交，持有 horizon_bars 根后平仓
ENTRY_DELAY_BARS = 2


@dataclass(frozen=True)
class Thresholds:
    compression: float
    volume: float
    close_position: float

    def as_dict(self) -> dict[str, float]:
        return {
            "compression_max": self.compression,
            "volume_ratio_min": self.volume,
            "close_position_min": self.close_position,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider-uri", type=Path, default=DEFAULT_PROVIDER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start", default="2024-01-01 00:00:00")
    parser.add_argument("--end", default=None)
    parser.add_argument("--horizon-bars", type=int, default=60)
    parser.add_argument("--fee-rate", type=float, default=0.0004)
    parser.add_argument("--slippage-rate", type=float, default=0.0001)
    parser.add_argument("--max-entry-upper-wick-ratio", type=float, default=0.4)
    parser.add_argument("--min-entry-close-position", type=float, default=0.6)
    parser.add_argument("--train-months", type=int, default=12)
    parser.add_argument("--valid-months", type=int, default=3)
    parser.add_argument("--test-months", type=int, default=2)
    parser.add_argument(
        "--step-months",
        type=int,
        default=None,
        help="fold 步进，默认等于 test-months 以保证 test 段互不重叠",
    )
    parser.add_argument(
        "--embargo-minutes",
        type=int,
        default=60 * 24,
        help="须不小于最长特征回看窗口（本因子为 60 根 + shift(1)）",
    )
    parser.add_argument(
        "--compression-grid",
        type=float,
        nargs="+",
        default=[0.5, 0.7, 0.9],
    )
    parser.add_argument("--volume-grid", type=float, nargs="+", default=[1.2, 1.5, 2.5])
    parser.add_argument("--close-position-grid", type=float, nargs="+", default=[0.5, 0.6, 0.7])
    parser.add_argument(
        "--min-train-trades",
        type=int,
        default=30,
        help="train 段成交数低于该值的阈值组合直接淘汰",
    )
    parser.add_argument(
        "--min-valid-trades",
        type=int,
        default=10,
        help="valid 段成交数低于该值的候选不予选中",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="train 段按 t 值取前 k 个候选进入 valid 段复选",
    )
    return parser.parse_args()


def round_trip_break_even_return(fee_rate: float, slippage_rate: float) -> float:
    entry_multiplier = (1 + slippage_rate) * (1 + fee_rate)
    exit_multiplier = (1 - slippage_rate) * (1 - fee_rate)
    return entry_multiplier / exit_multiplier - 1


def build_split_config(args: argparse.Namespace) -> SplitConfig:
    step_months = args.step_months if args.step_months is not None else args.test_months
    return SplitConfig(
        train_minutes=args.train_months * MINUTES_PER_MONTH,
        valid_minutes=args.valid_months * MINUTES_PER_MONTH,
        test_minutes=args.test_months * MINUTES_PER_MONTH,
        step_minutes=step_months * MINUTES_PER_MONTH,
        embargo_minutes=args.embargo_minutes,
    )


def first_breakout_flags(data: pd.DataFrame) -> pd.Series:
    """与 make_dataset 中 first_breakout 保持同一定义，用于去除重复触发。"""
    close = data["close"].astype(float)
    prior_high_20 = data["high"].rolling(20).max().shift(1)
    return (close > prior_high_20) & (close.shift(1) <= prior_high_20.shift(1))


def build_signal(
    features: pd.DataFrame,
    static_mask: pd.Series,
    thresholds: Thresholds,
) -> pd.Series:
    mask = (
        static_mask
        & features["volatility_compression_20_60"].le(thresholds.compression)
        & features["volume_ratio_60_prev"].ge(thresholds.volume)
        & features["close_position"].clip(lower=0, upper=1).ge(thresholds.close_position)
    )
    return mask.astype(float)


def trade_stats(trades: pd.DataFrame, column: str) -> dict[str, float]:
    """按笔统计收益（bp），交易已在 select_non_overlapping_trades 中去重叠。"""
    if trades.empty:
        return {"trades": 0, "mean_bp": 0.0, "std_bp": 0.0, "se_bp": 0.0, "t_stat": 0.0}
    values = trades[column].astype(float) * 1e4
    n = int(len(values))
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if n > 1 else 0.0
    se = std / math.sqrt(n) if n > 1 and std > 0 else 0.0
    return {
        "trades": n,
        "mean_bp": mean,
        "std_bp": std,
        "se_bp": se,
        "t_stat": mean / se if se > 0 else 0.0,
        "ci95_low_bp": mean - 1.96 * se,
        "ci95_high_bp": mean + 1.96 * se,
        "win_rate": float((values > 0).mean()),
    }


def sign_test_p_value(positive: int, total: int) -> float:
    """零假设 p=0.5 下 P(X >= positive) 的单侧尾概率。"""
    if total <= 0:
        return 1.0
    tail = sum(math.comb(total, i) for i in range(positive, total + 1))
    return tail / (2**total)


def run_window(
    data: pd.DataFrame,
    features: pd.DataFrame,
    static_mask: pd.Series,
    window: pd.DatetimeIndex,
    thresholds: Thresholds,
    args: argparse.Namespace,
) -> pd.DataFrame:
    # 原地覆盖信号列，避免每次网格评估都复制百万行特征表
    features["squeeze_breakout_signal"] = build_signal(features, static_mask, thresholds)
    trades, *_ = select_non_overlapping_trades(
        data=data,
        features=features,
        start_time=window[0],
        end_time=window[-1],
        horizon_bars=args.horizon_bars,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        max_entry_upper_wick_ratio=args.max_entry_upper_wick_ratio,
        min_entry_close_position=args.min_entry_close_position,
    )
    return trades


def search_thresholds(
    data: pd.DataFrame,
    features: pd.DataFrame,
    static_mask: pd.Series,
    fold: Fold,
    grid: list[Thresholds],
    args: argparse.Namespace,
) -> tuple[Thresholds | None, list[dict[str, object]]]:
    """train 段搜参、valid 段选参；test 段在此函数内不被触碰。"""
    train_scores: list[tuple[float, Thresholds, dict[str, float]]] = []
    for thresholds in grid:
        stats = trade_stats(
            run_window(data, features, static_mask, fold.train, thresholds, args), "net_return"
        )
        if stats["trades"] < args.min_train_trades:
            continue
        train_scores.append((stats["t_stat"], thresholds, stats))

    train_scores.sort(key=lambda item: item[0], reverse=True)
    shortlist = train_scores[: args.top_k]

    trace: list[dict[str, object]] = []
    best: tuple[float, Thresholds] | None = None
    for train_t, thresholds, train_stat in shortlist:
        valid_stat = trade_stats(
            run_window(data, features, static_mask, fold.valid, thresholds, args), "net_return"
        )
        trace.append(
            {
                **thresholds.as_dict(),
                "train_trades": train_stat["trades"],
                "train_mean_bp": train_stat["mean_bp"],
                "train_t": train_t,
                "valid_trades": valid_stat["trades"],
                "valid_mean_bp": valid_stat["mean_bp"],
                "valid_t": valid_stat["t_stat"],
            }
        )
        if valid_stat["trades"] < args.min_valid_trades:
            continue
        if best is None or valid_stat["t_stat"] > best[0]:
            best = (valid_stat["t_stat"], thresholds)

    return (best[1] if best else None), trace


def summarize_folds(
    data: pd.DataFrame,
    features: pd.DataFrame,
    static_mask: pd.Series,
    folds: list[Fold],
    grid: list[Thresholds],
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], pd.DataFrame, list[dict[str, object]]]:
    fold_rows: list[dict[str, object]] = []
    search_rows: list[dict[str, object]] = []
    oos_trades: list[pd.DataFrame] = []

    for i, fold in enumerate(folds):
        thresholds, trace = search_thresholds(data, features, static_mask, fold, grid, args)
        for row in trace:
            search_rows.append({"fold": i, **row})

        if thresholds is None:
            fold_rows.append(
                {
                    "fold": i,
                    "test_start": fold.test[0].isoformat(),
                    "test_end": fold.test[-1].isoformat(),
                    "selected": False,
                    "reason": "train/valid 段成交数不足，无可用阈值组合",
                }
            )
            print(f"[fold {i}] 无可用阈值组合，跳过")
            continue

        trades = run_window(data, features, static_mask, fold.test, thresholds, args)
        trades = trades.assign(fold=i)
        oos_trades.append(trades)
        net = trade_stats(trades, "net_return")
        gross = trade_stats(trades, "gross_return")
        fold_rows.append(
            {
                "fold": i,
                "train_start": fold.train[0].isoformat(),
                "train_end": fold.train[-1].isoformat(),
                "valid_start": fold.valid[0].isoformat(),
                "valid_end": fold.valid[-1].isoformat(),
                "test_start": fold.test[0].isoformat(),
                "test_end": fold.test[-1].isoformat(),
                "selected": True,
                **thresholds.as_dict(),
                "test_trades": net["trades"],
                "test_net_mean_bp": net["mean_bp"],
                "test_net_t": net["t_stat"],
                "test_gross_mean_bp": gross["mean_bp"],
                "test_win_rate": net.get("win_rate", 0.0),
            }
        )
        print(
            f"[fold {i}] test={fold.test[0]:%Y-%m-%d}~{fold.test[-1]:%Y-%m-%d} "
            f"n={net['trades']} net={net['mean_bp']:+.2f}bp gross={gross['mean_bp']:+.2f}bp "
            f"t={net['t_stat']:+.2f}"
        )

    pooled = (
        pd.concat(oos_trades, ignore_index=True)
        if oos_trades
        else pd.DataFrame(columns=["net_return", "gross_return", "fold"])
    )
    return fold_rows, pooled, search_rows


def threshold_stability(fold_rows: list[dict[str, object]]) -> dict[str, object]:
    selected = [row for row in fold_rows if row.get("selected")]
    stability: dict[str, object] = {}
    for key in ("compression_max", "volume_ratio_min", "close_position_min"):
        values = [row[key] for row in selected]
        if not values:
            stability[key] = {"distinct": 0, "modal_share": 0.0}
            continue
        counts = pd.Series(values).value_counts()
        stability[key] = {
            "distinct": int(counts.size),
            "modal_value": float(counts.index[0]),
            "modal_share": float(counts.iloc[0] / len(values)),
        }
    return stability


def build_summary(
    fold_rows: list[dict[str, object]],
    pooled: pd.DataFrame,
    args: argparse.Namespace,
    split: SplitConfig,
    grid_size: int,
) -> dict[str, object]:
    net = trade_stats(pooled, "net_return")
    gross = trade_stats(pooled, "gross_return")
    evaluated = [row for row in fold_rows if row.get("selected")]
    positive = sum(1 for row in evaluated if float(row["test_net_mean_bp"]) > 0)
    break_even_bp = round_trip_break_even_return(args.fee_rate, args.slippage_rate) * 1e4

    return {
        "config": {
            "horizon_bars": args.horizon_bars,
            "entry_delay_bars": ENTRY_DELAY_BARS,
            "label_horizon_bars": ENTRY_DELAY_BARS + args.horizon_bars,
            "fee_rate_per_side": args.fee_rate,
            "slippage_rate_per_side": args.slippage_rate,
            "round_trip_break_even_bp": break_even_bp,
            "train_minutes": split.train_minutes,
            "valid_minutes": split.valid_minutes,
            "test_minutes": split.test_minutes,
            "step_minutes": split.step_minutes,
            "embargo_minutes": split.embargo_minutes,
            "purge_minutes": split.embargo_minutes + ENTRY_DELAY_BARS + args.horizon_bars,
        },
        "multiple_testing": {
            "grid_size": grid_size,
            "folds_searched": len(fold_rows),
            "total_train_evaluations": grid_size * len(fold_rows),
            "note": "train 段共评估上述次数，train 最优值必然含过拟合成分，仅 test 段结论可信",
        },
        "pooled_out_of_sample": {
            "net": net,
            "gross": gross,
            "gross_alpha_upper_bound_bp": gross.get("ci95_high_bp", 0.0),
            "cost_exceeds_alpha_upper_bound": gross.get("ci95_high_bp", 0.0) < break_even_bp,
        },
        "fold_consistency": {
            "folds_evaluated": len(evaluated),
            "folds_positive": positive,
            "sign_test_p_value": sign_test_p_value(positive, len(evaluated)),
            "note": "fold 数少时符号检验功效很低，仅在 >=7/8 或 <=1/8 时具有解释力",
        },
        "threshold_stability": threshold_stability(fold_rows),
        "folds": fold_rows,
    }


def print_report(summary: dict[str, object]) -> None:
    pooled = summary["pooled_out_of_sample"]
    net, gross = pooled["net"], pooled["gross"]
    cost = summary["config"]["round_trip_break_even_bp"]
    consistency = summary["fold_consistency"]

    print("\n===== 合并样本外结论 =====")
    print(f"非重叠交易数        : {net['trades']}")
    print(
        f"毛收益/笔           : {gross['mean_bp']:+.2f} bp "
        f"(95% CI {gross.get('ci95_low_bp', 0):+.2f} ~ {gross.get('ci95_high_bp', 0):+.2f})"
    )
    print(f"净收益/笔           : {net['mean_bp']:+.2f} bp  t={net['t_stat']:+.2f}")
    print(f"双边成本            : {cost:.2f} bp")
    print(f"毛 alpha 95% 上界   : {gross.get('ci95_high_bp', 0):+.2f} bp")
    verdict = "低于成本" if pooled["cost_exceeds_alpha_upper_bound"] else "未能排除高于成本"
    print(f"结论                : alpha 上界{verdict}")
    print(
        f"正收益 fold         : {consistency['folds_positive']}/{consistency['folds_evaluated']}"
        f"  符号检验 p={consistency['sign_test_p_value']:.3f}"
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    factor = load_factor_module()
    data = factor.load_market_data(args.provider_uri, args.start, args.end)
    features, _ = factor.make_dataset(data, args.horizon_bars)

    breakout = first_breakout_flags(data).reindex(features.index).fillna(False)
    static_mask = features["breakout_strength_20"].gt(0) & breakout

    split = build_split_config(args)
    folds = walk_forward_folds(
        pd.DatetimeIndex(data["datetime"]),
        split,
        ENTRY_DELAY_BARS + args.horizon_bars,
    )
    print(f"共构造 {len(folds)} 个 fold，OOS 覆盖 {folds[0].test[0]:%Y-%m} ~ {folds[-1].test[-1]:%Y-%m}")

    grid = [
        Thresholds(compression=c, volume=v, close_position=p)
        for c, v, p in itertools.product(
            args.compression_grid, args.volume_grid, args.close_position_grid
        )
    ]
    fold_rows, pooled, search_rows = summarize_folds(
        data, features, static_mask, folds, grid, args
    )
    summary = build_summary(fold_rows, pooled, args, split, len(grid))

    pd.DataFrame(fold_rows).to_csv(args.output_dir / "walk_forward_folds.csv", index=False)
    pd.DataFrame(search_rows).to_csv(args.output_dir / "walk_forward_search.csv", index=False)
    pooled.to_csv(args.output_dir / "walk_forward_trades.csv", index=False)
    (args.output_dir / "walk_forward_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print_report(summary)
    print(f"\n输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()
