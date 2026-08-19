import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.joinpath("scripts")))
from questdb_to_qlib import aggregate_orderbook, aggregate_trades, write_symbol_csv


def test_aggregate_trades_ohlcv_and_vwap():
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02 09:30:00.100", "2024-01-02 09:30:00.800", "2024-01-02 09:30:01.100"]),
            "symbol": ["BTC", "BTC", "BTC"],
            "price": [100, 102, 101],
            "size": [2, 3, 1],
        }
    )
    result = aggregate_trades(frame, "1s")
    assert result[["open", "high", "low", "close", "volume", "trade_count"]].iloc[0].tolist() == [100, 102, 100, 102, 5, 2]
    assert result["vwap"].iloc[0] == 101.2
    assert len(result) == 2


def test_orderbook_absolute_updates_and_top_levels():
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02 09:30:00.1", "2024-01-02 09:30:00.2", "2024-01-02 09:30:01.1"]),
            "symbol": ["BTC"] * 3,
            "side": ["bid", "ask", "bid"],
            "price": [99, 101, 98],
            "size": [2, 3, 4],
        }
    )
    result = aggregate_orderbook(frame, "1s", top_n=2)
    assert result.loc[0, "bid_price_1"] == 99
    assert result.loc[0, "ask_price_1"] == 101
    assert result.loc[0, "mid"] == 100
    assert result.loc[1, "bid_price_1"] == 99
    assert result.loc[1, "bid_price_2"] == 98


def test_orderbook_delta_zero_removes_level_and_writes_one_file(tmp_path):
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02 09:30:00", "2024-01-02 09:30:00.5"]),
            "symbol": ["BTC", "BTC"],
            "side": ["bid", "bid"],
            "price": [99, 99],
            "size": [2, -2],
        }
    )
    result = aggregate_orderbook(frame, "1s", size_mode="delta")
    assert pd.isna(result.loc[0, "bid_price_1"])
    paths = write_symbol_csv(result, tmp_path)
    assert paths == [tmp_path / "btc.csv"]