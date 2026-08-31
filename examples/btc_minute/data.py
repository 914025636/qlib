"""数据层：逐笔成交与订单簿 -> 分钟 bar 特征表。

输入格式与工作区 `trades参考` / `orderbook参考` 一致：
    trades:    exchange, symbol, side, price, quantity, trade_id, timestamp
    orderbook: exchange, symbol, side, update_type, price, qty, sequence,
               source_update_count, timestamp
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DataConfig

TRADE_COLUMNS = ["side", "price", "quantity", "timestamp"]
BOOK_COLUMNS = ["side", "update_type", "price", "qty", "sequence", "timestamp"]


def _read(path: Path, columns: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name} 缺少列: {sorted(missing)}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    return frame.dropna(subset=["timestamp"])


def build_trade_bars(trades: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
    """成交流聚合为分钟 bar，附带主动买卖与大单结构。"""
    df = trades.sort_values("timestamp", kind="stable").copy()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df = df.dropna(subset=["price", "quantity"])
    df["amount"] = df["price"] * df["quantity"]
    is_buy = df["side"].astype(str).str.lower().isin(["buy", "b", "bid"])
    df["signed_amount"] = df["amount"].where(is_buy, -df["amount"])
    df["buy_amount"] = df["amount"].where(is_buy, 0.0)
    # 大单阈值用全样本分位数会引入前视，改为对数金额直接分层由模型学习
    big = df["amount"] >= df["amount"].expanding().quantile(0.9).shift().bfill()
    df["big_signed"] = df["signed_amount"].where(big, 0.0)

    grouped = df.set_index("timestamp").groupby(pd.Grouper(freq=freq))
    bars = pd.DataFrame(
        {
            "open": grouped["price"].first(),
            "high": grouped["price"].max(),
            "low": grouped["price"].min(),
            "close": grouped["price"].last(),
            "volume": grouped["quantity"].sum(),
            "amount": grouped["amount"].sum(),
            "trade_count": grouped["price"].size(),
            "signed_amount": grouped["signed_amount"].sum(),
            "buy_amount": grouped["buy_amount"].sum(),
            "big_signed_amount": grouped["big_signed"].sum(),
        }
    )
    bars = bars.dropna(subset=["close"])
    bars["vwap"] = bars["amount"] / bars["volume"].replace(0, np.nan)
    bars["vwap"] = bars["vwap"].fillna(bars["close"])
    bars["trade_imbalance"] = bars["signed_amount"] / bars["amount"].replace(0, np.nan)
    bars["big_imbalance"] = bars["big_signed_amount"] / bars["amount"].replace(0, np.nan)
    bars["avg_trade_amount"] = bars["amount"] / bars["trade_count"].replace(0, np.nan)

    # 分钟内已实现矩：用逐笔收益而非 bar 收益
    df["logret"] = np.log(df["price"]).diff()
    tick = df.set_index("timestamp")["logret"].groupby(pd.Grouper(freq=freq))
    bars["rv"] = tick.apply(lambda s: np.square(s).sum())
    bars["rv_up"] = tick.apply(lambda s: np.square(s[s > 0]).sum())
    bars["rv_down"] = tick.apply(lambda s: np.square(s[s < 0]).sum())
    bars["rskew"] = tick.apply(_realized_skew)
    bars["rkurt"] = tick.apply(_realized_kurt)
    return bars


def _realized_skew(series: pd.Series) -> float:
    r = series.dropna().to_numpy()
    rv = np.square(r).sum()
    if r.size < 3 or rv <= 0:
        return np.nan
    return float(np.sqrt(r.size) * np.power(r, 3).sum() / rv**1.5)


def _realized_kurt(series: pd.Series) -> float:
    r = series.dropna().to_numpy()
    rv = np.square(r).sum()
    if r.size < 4 or rv <= 0:
        return np.nan
    return float(r.size * np.power(r, 4).sum() / rv**2)


def replay_orderbook(book: pd.DataFrame, sample_freq: str, top_n: int) -> pd.DataFrame:
    """按 sequence 回放增量，输出每个采样桶末尾的盘口快照。"""
    df = book.sort_values(["sequence", "timestamp"], kind="stable").copy()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce")
    df = df.dropna(subset=["price", "qty"])
    df["bucket"] = df["timestamp"].dt.floor(sample_freq)
    df["is_bid"] = df["side"].astype(str).str.lower().isin(["bid", "buy", "b"])
    df["is_snapshot"] = df["update_type"].astype(str).str.lower() == "snapshot"

    bids: dict[float, float] = {}
    asks: dict[float, float] = {}
    rows: list[dict] = []
    current = None
    prev_snapshot: dict | None = None
    seen_snapshot = False

    for bucket, is_bid, price, qty, is_snap in df[
        ["bucket", "is_bid", "price", "qty", "is_snapshot"]
    ].itertuples(index=False, name=None):
        if current is not None and bucket != current:
            snap = _book_snapshot(current, bids, asks, top_n, prev_snapshot)
            if snap is not None:
                rows.append(snap)
                prev_snapshot = snap
        current = bucket
        if is_snap and not seen_snapshot:
            bids.clear()
            asks.clear()
            seen_snapshot = True
        side = bids if is_bid else asks
        if qty <= 0:
            side.pop(price, None)
        else:
            side[price] = qty
    if current is not None:
        snap = _book_snapshot(current, bids, asks, top_n, prev_snapshot)
        if snap is not None:
            rows.append(snap)
    return pd.DataFrame(rows).set_index("timestamp") if rows else pd.DataFrame()


def _book_snapshot(
    bucket: pd.Timestamp,
    bids: dict[float, float],
    asks: dict[float, float],
    top_n: int,
    prev: dict | None,
) -> dict | None:
    if not bids or not asks:
        return None
    top_bids = sorted(bids.items(), key=lambda kv: -kv[0])[:top_n]
    top_asks = sorted(asks.items(), key=lambda kv: kv[0])[:top_n]
    b1, bs1 = top_bids[0]
    a1, as1 = top_asks[0]
    if a1 <= b1:  # 交叉盘口视为脏数据
        return None
    mid = (a1 + b1) / 2
    bid_vol = sum(q for _, q in top_bids)
    ask_vol = sum(q for _, q in top_asks)
    snap = {
        "timestamp": bucket,
        "mid": mid,
        "micro_price": (b1 * as1 + a1 * bs1) / (bs1 + as1),
        "spread_rel": (a1 - b1) / mid,
        "obi1": (bs1 - as1) / (bs1 + as1),
        "obi_top": (bid_vol - ask_vol) / (bid_vol + ask_vol),
        "depth_bid": bid_vol,
        "depth_ask": ask_vol,
        "_bid1": b1,
        "_ask1": a1,
        "_bs1": bs1,
        "_as1": as1,
    }
    snap["ofi"] = _ofi(snap, prev)
    return snap


def _ofi(cur: dict, prev: dict | None) -> float:
    """一档订单流不平衡（Cont et al. 2014）。"""
    if prev is None:
        return 0.0
    e = 0.0
    if cur["_bid1"] > prev["_bid1"]:
        e += cur["_bs1"]
    elif cur["_bid1"] < prev["_bid1"]:
        e -= prev["_bs1"]
    else:
        e += cur["_bs1"] - prev["_bs1"]
    if cur["_ask1"] < prev["_ask1"]:
        e -= cur["_as1"]
    elif cur["_ask1"] > prev["_ask1"]:
        e += prev["_as1"]
    else:
        e -= cur["_as1"] - prev["_as1"]
    return float(e)


def build_book_bars(snapshots: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
    """盘口快照聚合到分钟：末值 + 均值 + 波动，OFI 用求和。"""
    if snapshots.empty:
        return pd.DataFrame()
    snap = snapshots.drop(columns=[c for c in snapshots.columns if c.startswith("_")])
    grouped = snap.groupby(pd.Grouper(freq=freq))
    out = pd.DataFrame(
        {
            "mid": grouped["mid"].last(),
            "micro_price": grouped["micro_price"].last(),
            "spread_rel": grouped["spread_rel"].mean(),
            "spread_rel_std": grouped["spread_rel"].std(),
            "obi1": grouped["obi1"].mean(),
            "obi_top": grouped["obi_top"].mean(),
            "obi_top_last": grouped["obi_top"].last(),
            "depth_bid": grouped["depth_bid"].mean(),
            "depth_ask": grouped["depth_ask"].mean(),
            "ofi": grouped["ofi"].sum(),
        }
    )
    return out.dropna(subset=["mid"])


def load_bars(cfg: DataConfig) -> pd.DataFrame:
    """读取原始文件并合成完整分钟表；已存在缓存则直接读缓存。"""
    if cfg.bar_path.exists():
        return pd.read_parquet(cfg.bar_path)

    trades_path = cfg.raw_dir / "trades.csv"
    if not trades_path.exists():
        raise FileNotFoundError(f"缺少成交数据: {trades_path}")
    bars = build_trade_bars(_read(trades_path, TRADE_COLUMNS), cfg.bar_freq)

    book_path = cfg.raw_dir / "orderbook.csv"
    if book_path.exists():
        snaps = replay_orderbook(_read(book_path, BOOK_COLUMNS), cfg.book_sample_freq, cfg.book_top_n)
        book_bars = build_book_bars(snaps, cfg.bar_freq)
        if not book_bars.empty:
            bars = bars.join(book_bars, how="left")

    bars = reindex_full_minutes(bars)
    cfg.bar_path.parent.mkdir(parents=True, exist_ok=True)
    bars.to_parquet(cfg.bar_path)
    return bars


def reindex_full_minutes(bars: pd.DataFrame) -> pd.DataFrame:
    """7x24 市场补齐空缺分钟：价格前向填充，成交量置 0。"""
    full = pd.date_range(bars.index.min(), bars.index.max(), freq="1min", tz="UTC")
    out = bars.reindex(full)
    price_cols = [c for c in ["open", "high", "low", "close", "vwap", "mid", "micro_price"] if c in out]
    out[price_cols] = out[price_cols].ffill()
    for col in ["open", "high", "low"]:
        if col in out:
            out[col] = out[col].fillna(out["close"])
    zero_cols = [
        c
        for c in [
            "volume",
            "amount",
            "trade_count",
            "signed_amount",
            "buy_amount",
            "big_signed_amount",
            "rv",
            "rv_up",
            "rv_down",
            "ofi",
        ]
        if c in out
    ]
    out[zero_cols] = out[zero_cols].fillna(0.0)
    out["is_gap"] = (~full.isin(bars.index)).astype(float)
    out = out.ffill().dropna(subset=["close"])
    out.index.name = "datetime"
    return out
