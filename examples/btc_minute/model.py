"""模型训练与预测评估。

默认 LightGBM 回归；单标的时序问题下评估用 IC / 方向准确率 / 分层收益，
不使用横截面 RankIC。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .config import ModelConfig


def _fill(frame: pd.DataFrame, stats_: pd.Series | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """缺失值用训练集中位数填充，避免用全样本统计造成泄漏。"""
    med = stats_ if stats_ is not None else frame.median()
    return frame.fillna(med).fillna(0.0), med


def train_lightgbm(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    cfg: ModelConfig,
) -> tuple[object, pd.Series]:
    import lightgbm as lgb

    xt, med = _fill(x_train)
    xv, _ = _fill(x_valid, med)
    train_set = lgb.Dataset(xt, y_train)
    valid_set = lgb.Dataset(xv, y_valid, reference=train_set)
    booster = lgb.train(
        cfg.params,
        train_set,
        num_boost_round=cfg.num_boost_round,
        valid_sets=[valid_set],
        callbacks=[
            lgb.early_stopping(cfg.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    return booster, med


def predict(booster, x: pd.DataFrame, med: pd.Series) -> pd.Series:
    xf, _ = _fill(x, med)
    return pd.Series(booster.predict(xf, num_iteration=getattr(booster, "best_iteration", None)), index=x.index, name="pred")


def evaluate(pred: pd.Series, y: pd.Series, n_bins: int = 5) -> dict:
    df = pd.concat([pred.rename("pred"), y.rename("y")], axis=1).dropna()
    if df.empty:
        return {}
    ic = df["pred"].corr(df["y"])
    rank_ic = df["pred"].corr(df["y"], method="spearman")
    # IC 的时间稳定性：按天分组
    daily = df.groupby(df.index.floor("D")).apply(
        lambda g: g["pred"].corr(g["y"]) if len(g) > 30 else np.nan, include_groups=False
    )
    hit = float((np.sign(df["pred"]) == np.sign(df["y"])).mean())
    bins = pd.qcut(df["pred"], n_bins, labels=False, duplicates="drop")
    layered = df.groupby(bins)["y"].mean()
    return {
        "ic": float(ic),
        "rank_ic": float(rank_ic),
        "ic_ir": float(daily.mean() / (daily.std() + 1e-12)) if daily.notna().sum() > 3 else np.nan,
        "ic_daily_positive_ratio": float((daily > 0).mean()) if daily.notna().any() else np.nan,
        "hit_ratio": hit,
        "top_bin_mean": float(layered.iloc[-1]) if len(layered) else np.nan,
        "bottom_bin_mean": float(layered.iloc[0]) if len(layered) else np.nan,
        "long_short_spread": float(layered.iloc[-1] - layered.iloc[0]) if len(layered) > 1 else np.nan,
        "n_samples": int(len(df)),
    }


def feature_importance(booster, top: int = 30) -> pd.Series:
    gain = pd.Series(booster.feature_importance("gain"), index=booster.feature_name())
    return gain.sort_values(ascending=False).head(top)


def factor_screening(features: pd.DataFrame, y: pd.Series, min_abs_ic: float = 0.005) -> pd.DataFrame:
    """单因子体检：IC、t 值、缺失率、自相关（换手代理）。"""
    rows = []
    aligned = features.join(y.rename("y"), how="inner")
    target = aligned.pop("y")
    for col in aligned.columns:
        s = aligned[col]
        mask = s.notna() & target.notna()
        if mask.sum() < 500:
            continue
        ic = s[mask].corr(target[mask])
        rank_ic = s[mask].corr(target[mask], method="spearman")
        n = int(mask.sum())
        t_stat = ic * np.sqrt((n - 2) / max(1 - ic**2, 1e-12))
        rows.append(
            {
                "factor": col,
                "ic": ic,
                "rank_ic": rank_ic,
                "t_stat": t_stat,
                "p_value": float(2 * (1 - stats.norm.cdf(abs(t_stat)))),
                "nan_ratio": float(1 - mask.mean()),
                "autocorr_1": float(s.autocorr(1)),
                "n": n,
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["keep"] = result["ic"].abs() >= min_abs_ic
    return result.sort_values("ic", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


def drop_collinear(features: pd.DataFrame, threshold: float = 0.95) -> list[str]:
    """按相关性剔除冗余因子，保留列名列表。"""
    usable = features.loc[:, features.std() > 1e-12]
    corr = usable.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop = {col for col in upper.columns if (upper[col] > threshold).any()}
    return [c for c in usable.columns if c not in drop]
