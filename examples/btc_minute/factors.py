"""因子库：全部为截止当根 bar 收盘的可观测量，禁止使用未来信息。

分组：
    momentum   多周期动量 / 短周期反转
    volatility 已实现波动、半方差、跳跃、区间波动
    flow       主动买卖、大单、订单流不平衡
    liquidity  Amihud、Kyle lambda、价差、VWAP 偏离
    micro      盘口不平衡、微观价格偏离
    seasonal   7x24 时段周期编码
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12
SHORT_WINDOWS = (1, 3, 5, 10)
WINDOWS = (5, 15, 30, 60, 240, 720)


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0, np.nan)


def _zscore(series: pd.Series, window: int) -> pd.Series:
    roll = series.rolling(window, min_periods=window // 2)
    return (series - roll.mean()) / (roll.std() + EPS)


def momentum_factors(bars: pd.DataFrame) -> pd.DataFrame:
    close = bars["close"]
    logp = np.log(close)
    out = {}
    for w in WINDOWS:
        out[f"mom_{w}"] = logp - logp.shift(w)
        out[f"mom_z_{w}"] = _zscore(logp.diff(w), w * 2)
    for w in SHORT_WINDOWS:
        out[f"rev_{w}"] = -(logp - logp.shift(w))
    for w in (30, 120, 480):
        roll = close.rolling(w, min_periods=w // 2)
        rng = roll.max() - roll.min()
        out[f"pos_in_range_{w}"] = (close - roll.min()) / (rng + EPS)
        out[f"dist_ma_{w}"] = close / (roll.mean() + EPS) - 1
    ret = logp.diff()
    for lag in (1, 2, 3, 5):
        out[f"autocorr_{lag}"] = ret.rolling(240, min_periods=120).corr(ret.shift(lag))
    return pd.DataFrame(out, index=bars.index)


def volatility_factors(bars: pd.DataFrame) -> pd.DataFrame:
    ret = np.log(bars["close"]).diff()
    out = {}
    for w in WINDOWS:
        out[f"vol_{w}"] = ret.rolling(w, min_periods=w // 2).std()
    out["vol_ratio_15_240"] = _safe_div(out["vol_15"], out["vol_240"])
    out["vol_ratio_60_720"] = _safe_div(out["vol_60"], out["vol_720"])

    if "rv" in bars:
        rv = bars["rv"]
        for w in (15, 60, 240):
            out[f"rv_sum_{w}"] = rv.rolling(w, min_periods=w // 2).sum()
        out["rv_ratio_15_240"] = _safe_div(out["rv_sum_15"], out["rv_sum_240"])
        out["semivar_ratio"] = _safe_div(
            bars["rv_up"].rolling(60, min_periods=30).sum(),
            bars["rv_down"].rolling(60, min_periods=30).sum(),
        )
        # 跳跃代理：分钟内 RV 远大于近端中位数视为跳跃
        med = rv.rolling(240, min_periods=120).median()
        out["jump_ratio"] = _safe_div(rv, med)
    for col in ("rskew", "rkurt"):
        if col in bars:
            out[col] = bars[col]
            out[f"{col}_ma60"] = bars[col].rolling(60, min_periods=30).mean()

    hl = np.log(bars["high"] / bars["low"].clip(lower=EPS))
    out["parkinson_60"] = np.sqrt((hl**2).rolling(60, min_periods=30).mean() / (4 * np.log(2)))
    co = np.log(bars["close"] / bars["open"].clip(lower=EPS))
    out["garman_klass_60"] = np.sqrt(
        (0.5 * hl**2 - (2 * np.log(2) - 1) * co**2).rolling(60, min_periods=30).mean().clip(lower=0)
    )
    out["range_rel"] = (bars["high"] - bars["low"]) / bars["close"]
    return pd.DataFrame(out, index=bars.index)


def flow_factors(bars: pd.DataFrame) -> pd.DataFrame:
    out = {}
    amount = bars["amount"]
    for w in WINDOWS:
        out[f"amt_ratio_{w}"] = _safe_div(amount, amount.rolling(w, min_periods=w // 2).mean())
        out[f"amt_z_{w}"] = _zscore(np.log1p(amount), w)
    if "signed_amount" in bars:
        signed = bars["signed_amount"]
        for w in (5, 15, 60, 240):
            out[f"net_flow_{w}"] = _safe_div(
                signed.rolling(w, min_periods=w // 2).sum(),
                amount.rolling(w, min_periods=w // 2).sum(),
            )
        out["net_flow_1"] = bars.get("trade_imbalance")
    if "big_signed_amount" in bars:
        for w in (15, 60, 240):
            out[f"big_flow_{w}"] = _safe_div(
                bars["big_signed_amount"].rolling(w, min_periods=w // 2).sum(),
                amount.rolling(w, min_periods=w // 2).sum(),
            )
    if "trade_count" in bars:
        out["trade_count_z_60"] = _zscore(np.log1p(bars["trade_count"]), 60)
        out["avg_trade_z_60"] = _zscore(np.log1p(bars["avg_trade_amount"].fillna(0)), 60)
    if "ofi" in bars:
        ofi = bars["ofi"]
        for w in (1, 5, 15, 60):
            s = ofi if w == 1 else ofi.rolling(w, min_periods=1).sum()
            out[f"ofi_z_{w}"] = _zscore(s, 240)
    ret = np.log(bars["close"]).diff()
    out["corr_ret_vol_60"] = ret.rolling(60, min_periods=30).corr(np.log1p(amount).diff())
    out["corr_price_vol_60"] = bars["close"].rolling(60, min_periods=30).corr(np.log1p(amount))
    return pd.DataFrame(out, index=bars.index)


def liquidity_factors(bars: pd.DataFrame) -> pd.DataFrame:
    ret = np.log(bars["close"]).diff()
    amount = bars["amount"]
    out = {}
    illiq = ret.abs() / (amount + 1.0)
    for w in (60, 240):
        out[f"amihud_{w}"] = illiq.rolling(w, min_periods=w // 2).mean()
    if "signed_amount" in bars:
        signed = bars["signed_amount"] / 1e6
        cov = ret.rolling(240, min_periods=120).cov(signed)
        var = signed.rolling(240, min_periods=120).var()
        out["kyle_lambda_240"] = _safe_div(cov, var)
    out["vwap_dev"] = bars["close"] / bars["vwap"].replace(0, np.nan) - 1
    out["vwap_dev_ma15"] = out["vwap_dev"].rolling(15, min_periods=8).mean()
    if "spread_rel" in bars:
        out["spread_rel"] = bars["spread_rel"]
        out["spread_z_240"] = _zscore(bars["spread_rel"], 240)
    if "is_gap" in bars:
        out["gap_ratio_60"] = bars["is_gap"].rolling(60, min_periods=1).mean()
    return pd.DataFrame(out, index=bars.index)


def micro_factors(bars: pd.DataFrame) -> pd.DataFrame:
    if "mid" not in bars:
        return pd.DataFrame(index=bars.index)
    out = {}
    for col in ("obi1", "obi_top", "obi_top_last"):
        if col in bars:
            out[col] = bars[col]
            for w in (5, 15, 60):
                out[f"{col}_ma{w}"] = bars[col].rolling(w, min_periods=w // 2).mean()
    if "micro_price" in bars:
        out["micro_dev"] = bars["micro_price"] / bars["mid"] - 1
        out["micro_dev_z"] = _zscore(out["micro_dev"], 240)
        out["price_vs_mid"] = bars["close"] / bars["mid"] - 1
    if {"depth_bid", "depth_ask"} <= set(bars.columns):
        depth = bars["depth_bid"] + bars["depth_ask"]
        out["depth_z_240"] = _zscore(np.log1p(depth), 240)
        out["depth_skew"] = _safe_div(bars["depth_bid"] - bars["depth_ask"], depth)
    mid_ret = np.log(bars["mid"]).diff()
    out["mid_mom_15"] = mid_ret.rolling(15, min_periods=8).sum()
    return pd.DataFrame(out, index=bars.index)


def seasonal_factors(bars: pd.DataFrame) -> pd.DataFrame:
    idx = bars.index
    minute_of_day = idx.hour * 60 + idx.minute
    out = {
        "tod_sin": np.sin(2 * np.pi * minute_of_day / 1440),
        "tod_cos": np.cos(2 * np.pi * minute_of_day / 1440),
        "dow_sin": np.sin(2 * np.pi * idx.dayofweek / 7),
        "dow_cos": np.cos(2 * np.pi * idx.dayofweek / 7),
        # 主要交易时段（UTC）：亚洲 0-8、欧洲 7-16、美洲 13-21
        "sess_asia": ((idx.hour >= 0) & (idx.hour < 8)).astype(float),
        "sess_eu": ((idx.hour >= 7) & (idx.hour < 16)).astype(float),
        "sess_us": ((idx.hour >= 13) & (idx.hour < 21)).astype(float),
    }
    return pd.DataFrame(out, index=idx)


FACTOR_GROUPS = {
    "momentum": momentum_factors,
    "volatility": volatility_factors,
    "flow": flow_factors,
    "liquidity": liquidity_factors,
    "micro": micro_factors,
    "seasonal": seasonal_factors,
}


def build_factors(bars: pd.DataFrame, groups: list[str] | None = None) -> pd.DataFrame:
    names = groups or list(FACTOR_GROUPS)
    frames = [FACTOR_GROUPS[n](bars) for n in names]
    features = pd.concat([f for f in frames if not f.empty], axis=1)
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.loc[:, features.notna().mean() > 0.5]
    return features.loc[:, features.std(numeric_only=True) > EPS]
