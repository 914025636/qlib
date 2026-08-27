"""标签构造与 purged walk-forward 切分。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import LabelConfig, SplitConfig

EPS = 1e-12


def _forward_return(bars: pd.DataFrame, cfg: LabelConfig) -> pd.Series:
    """从 t+execution_lag 的执行价算到 t+execution_lag+horizon，剔除不可交易段。"""
    price = bars[cfg.price_col]
    entry = price.shift(-cfg.execution_lag)
    exit_ = price.shift(-(cfg.execution_lag + cfg.horizon))
    return np.log(exit_ / entry)


def make_label(bars: pd.DataFrame, cfg: LabelConfig) -> pd.Series:
    """未来 horizon 根 bar 的对数收益；执行价用 vwap 避免收盘价不可成交。"""
    fwd = _forward_return(bars, cfg)
    if cfg.threshold_in_vol is None:
        return fwd.rename("label")
    ret = np.log(bars["close"]).diff()
    vol = ret.rolling(cfg.vol_window, min_periods=cfg.vol_window // 2).std() * np.sqrt(cfg.horizon)
    thr = cfg.threshold_in_vol * vol
    label = pd.Series(0.0, index=bars.index)
    label[fwd > thr] = 1.0
    label[fwd < -thr] = -1.0
    label[fwd.isna() | vol.isna()] = np.nan
    return label.rename("label")


def vol_scaled_label(bars: pd.DataFrame, cfg: LabelConfig) -> pd.Series:
    """波动归一化收益，跨行情稳定，推荐作为回归目标。"""
    fwd = _forward_return(bars, cfg)
    ret = np.log(bars["close"]).diff()
    vol = ret.rolling(cfg.vol_window, min_periods=cfg.vol_window // 2).std() * np.sqrt(cfg.horizon)
    return (fwd / (vol + EPS)).clip(-5, 5).rename("label")


@dataclass(frozen=True)
class Fold:
    train: pd.DatetimeIndex
    valid: pd.DatetimeIndex
    test: pd.DatetimeIndex

    def __repr__(self) -> str:
        def rng(idx: pd.DatetimeIndex) -> str:
            return f"{idx[0]:%Y-%m-%d %H:%M}~{idx[-1]:%Y-%m-%d %H:%M}"

        return f"Fold(train={rng(self.train)}, valid={rng(self.valid)}, test={rng(self.test)})"


def walk_forward_folds(index: pd.DatetimeIndex, cfg: SplitConfig, label_horizon: int) -> list[Fold]:
    """滚动切分；相邻段之间剔除 embargo + horizon 根 bar 以隔断标签重叠。

    label_horizon 传 execution_lag + horizon。
    """
    purge = cfg.embargo_minutes + label_horizon
    n = len(index)
    span = cfg.train_minutes + purge + cfg.valid_minutes + purge + cfg.test_minutes
    if n < span:
        raise ValueError(f"样本不足：需要至少 {span} 根 bar，实际 {n}")

    folds: list[Fold] = []
    start = 0
    while start + span <= n:
        tr_end = start + cfg.train_minutes
        va_start = tr_end + purge
        va_end = va_start + cfg.valid_minutes
        te_start = va_end + purge
        te_end = te_start + cfg.test_minutes
        folds.append(
            Fold(
                train=index[start:tr_end],
                valid=index[va_start:va_end],
                test=index[te_start:te_end],
            )
        )
        start += cfg.step_minutes
    return folds


def align_xy(features: pd.DataFrame, label: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    joined = features.join(label, how="inner").dropna(subset=["label"])
    y = joined.pop("label")
    return joined, y
