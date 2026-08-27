"""端到端流水线：数据 -> 因子 -> 标签 -> walk-forward 训练 -> 回测。

用法:
    python -m btc_minute.pipeline screen     # 单因子体检
    python -m btc_minute.pipeline train      # 滚动训练 + 样本外回测
    python -m btc_minute.pipeline demo       # 用合成数据跑通全流程
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from .backtest import cost_sensitivity, run_backtest
from .config import CONFIG, ARTIFACT_DIR, Config
from .data import load_bars
from .factors import build_factors
from .labels import align_xy, vol_scaled_label, walk_forward_folds
from .model import (
    drop_collinear,
    evaluate,
    factor_screening,
    feature_importance,
    predict,
    train_lightgbm,
)


def prepare(cfg: Config, bars: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    bars = load_bars(cfg.data) if bars is None else bars
    features = build_factors(bars)
    label = vol_scaled_label(bars, cfg.label)
    x, y = align_xy(features, label)
    return x, y, bars.loc[x.index]


def screen(cfg: Config, bars: pd.DataFrame | None = None) -> pd.DataFrame:
    x, y, _ = prepare(cfg, bars)
    report = factor_screening(x, y)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    report.to_csv(ARTIFACT_DIR / "factor_screening.csv", index=False)
    return report


def train(cfg: Config, bars: pd.DataFrame | None = None) -> dict:
    x, y, bars = prepare(cfg, bars)
    folds = walk_forward_folds(x.index, cfg.split, cfg.label.execution_lag + cfg.label.horizon)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    metrics: list[dict] = []
    oos_pred: list[pd.Series] = []
    importances: list[pd.Series] = []

    for i, fold in enumerate(folds):
        cols = drop_collinear(x.loc[fold.train])
        booster, med = train_lightgbm(
            x.loc[fold.train, cols],
            y.loc[fold.train],
            x.loc[fold.valid, cols],
            y.loc[fold.valid],
            cfg.model,
        )
        pred = predict(booster, x.loc[fold.test, cols], med)
        metrics.append({"fold": i, "split": "test", **evaluate(pred, y.loc[fold.test])})
        metrics.append(
            {
                "fold": i,
                "split": "valid",
                **evaluate(predict(booster, x.loc[fold.valid, cols], med), y.loc[fold.valid]),
            }
        )
        oos_pred.append(pred)
        importances.append(feature_importance(booster, top=50))
        print(f"[fold {i}] {fold} test_ic={metrics[-2].get('ic'):.4f}")

    metric_df = pd.DataFrame(metrics)
    metric_df.to_csv(ARTIFACT_DIR / "fold_metrics.csv", index=False)
    pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False).to_csv(
        ARTIFACT_DIR / "feature_importance.csv"
    )

    full_pred = pd.concat(oos_pred).sort_index()
    full_pred = full_pred[~full_pred.index.duplicated(keep="last")]
    full_pred.to_frame().to_parquet(ARTIFACT_DIR / "oos_pred.parquet")

    result, perf = run_backtest(bars, full_pred, cfg.cost, cfg.trade)
    result.to_parquet(ARTIFACT_DIR / "backtest.parquet")
    sens = cost_sensitivity(bars, full_pred, cfg.cost, cfg.trade)
    sens.to_csv(ARTIFACT_DIR / "cost_sensitivity.csv", index=False)

    summary = {
        "n_folds": len(folds),
        "oos_ic_mean": float(metric_df.query("split=='test'")["ic"].mean()),
        "oos_ic_std": float(metric_df.query("split=='test'")["ic"].std()),
        "oos_hit_ratio": float(metric_df.query("split=='test'")["hit_ratio"].mean()),
        "backtest": perf,
    }
    (ARTIFACT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=float))
    print("\n成本敏感性:\n", sens[["cost_multiplier", "sharpe", "annual_return", "max_drawdown"]])
    return summary


def synthetic_bars(minutes: int = 60 * 24 * 200, seed: int = 0) -> pd.DataFrame:
    """生成带弱可预测性的合成分钟数据，用于跑通流程与做零假设基准。"""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=minutes, freq="1min", tz="UTC")
    flow = rng.standard_normal(minutes)
    noise = rng.standard_normal(minutes) * 6e-4
    # alpha 必须落在可交易窗口：t 的 flow 影响 t+2 起的收益（t+1 才成交），并持续至持仓期末
    persist = 30
    drift = sum(np.roll(flow, k) for k in range(2, 2 + persist)) / persist
    ret = 0.6 * 6e-4 * drift + noise
    ret[: 2 + persist] = 0.0
    close = 60000 * np.exp(np.cumsum(ret))
    spread = np.abs(rng.standard_normal(minutes)) * 4e-4 * close
    volume = np.abs(rng.lognormal(0, 1, minutes)) * 2
    amount = volume * close
    bars = pd.DataFrame(
        {
            "open": close / np.exp(ret),
            "high": close + spread,
            "low": close - spread,
            "close": close,
            "volume": volume,
            "amount": amount,
            "trade_count": rng.integers(20, 600, minutes),
            "vwap": close * (1 + rng.standard_normal(minutes) * 5e-5),
            "signed_amount": flow * amount * 0.3,
            "big_signed_amount": flow * amount * 0.1,
            "avg_trade_amount": amount / rng.integers(20, 600, minutes),
            "trade_imbalance": np.tanh(flow),
            "rv": ret**2 * rng.uniform(0.8, 1.5, minutes),
            "rv_up": np.where(ret > 0, ret**2, 0) * rng.uniform(0.8, 1.5, minutes),
            "rv_down": np.where(ret < 0, ret**2, 0) * rng.uniform(0.8, 1.5, minutes),
            "rskew": rng.standard_normal(minutes) * 0.5,
            "rkurt": 3 + np.abs(rng.standard_normal(minutes)),
            "mid": close,
            "micro_price": close * (1 + np.tanh(flow) * 2e-5),
            "spread_rel": spread / close,
            "obi1": np.tanh(flow * 0.8),
            "obi_top": np.tanh(flow * 0.5),
            "obi_top_last": np.tanh(flow * 0.5 + rng.standard_normal(minutes) * 0.1),
            "depth_bid": np.abs(rng.lognormal(0, 0.5, minutes)) * 10,
            "depth_ask": np.abs(rng.lognormal(0, 0.5, minutes)) * 10,
            "ofi": flow * 5,
            "is_gap": 0.0,
        },
        index=idx,
    )
    bars.index.name = "datetime"
    return bars


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("screen", "train", "demo"))
    args = parser.parse_args(argv)

    if args.command == "demo":
        from dataclasses import replace

        bars = synthetic_bars()
        cfg = replace(
            CONFIG,
            split=replace(
                CONFIG.split,
                train_minutes=60 * 24 * 60,
                valid_minutes=60 * 24 * 15,
                test_minutes=60 * 24 * 15,
                step_minutes=60 * 24 * 30,
            ),
        )
        print(screen(cfg, bars).head(15).to_string(index=False))
        train(cfg, bars)
        return 0

    if args.command == "screen":
        print(screen(CONFIG).head(30).to_string(index=False))
    else:
        train(CONFIG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
