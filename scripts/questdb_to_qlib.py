"""Convert QuestDB trades or order-book increments to Qlib-compatible CSV files.

The QuestDB client is optional.  Install ``psycopg[binary]`` when using the
database CLI; the aggregation functions can be imported and tested offline.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def validate_identifier(value: str, name: str) -> str:
    if not IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f"Invalid SQL identifier for {name}: {value!r}")
    return value


def _normalise_frame(frame: pd.DataFrame, timestamp_column: str, symbol_column: str) -> pd.DataFrame:
    result = frame.copy()
    result[timestamp_column] = pd.to_datetime(result[timestamp_column], utc=True, errors="coerce")
    result = result.dropna(subset=[timestamp_column, symbol_column])
    result[symbol_column] = result[symbol_column].astype(str)
    return result.sort_values([symbol_column, timestamp_column], kind="stable").reset_index(drop=True)


def aggregate_trades(
    frame: pd.DataFrame,
    frequency: str,
    timestamp_column: str = "timestamp",
    symbol_column: str = "symbol",
    price_column: str = "price",
    size_column: str = "size",
    amount_column: Optional[str] = None,
) -> pd.DataFrame:
    """Aggregate trades into OHLCV rows, retaining only non-empty buckets."""
    data = _normalise_frame(frame, timestamp_column, symbol_column)
    if data.empty:
        return pd.DataFrame(columns=["symbol", "date", "open", "high", "low", "close", "volume", "amount", "vwap", "trade_count"])

    data[price_column] = pd.to_numeric(data[price_column], errors="coerce")
    data[size_column] = pd.to_numeric(data[size_column], errors="coerce").fillna(0.0)
    data = data.dropna(subset=[price_column])
    data["_amount"] = (
        pd.to_numeric(data[amount_column], errors="coerce")
        if amount_column
        else data[price_column] * data[size_column]
    )
    data["_bucket"] = data[timestamp_column].dt.floor(frequency)
    grouped = data.groupby([symbol_column, "_bucket"], sort=True, observed=True)
    result = grouped[price_column].agg(open="first", high="max", low="min", close="last").reset_index()
    stats = grouped.agg(volume=(size_column, "sum"), amount=("_amount", "sum"), trade_count=(price_column, "size")).reset_index()
    result = result.merge(stats, on=[symbol_column, "_bucket"], how="left")
    result["vwap"] = result["amount"].div(result["volume"].replace(0, np.nan))
    return result.rename(columns={symbol_column: "symbol", "_bucket": "date"})[
        ["symbol", "date", "open", "high", "low", "close", "volume", "amount", "vwap", "trade_count"]
    ]


def _book_levels(book: Dict[str, Dict[float, float]], side: str, top_n: int) -> Iterable[tuple[float, float]]:
    levels = book[side].items()
    ordered = sorted(levels, key=lambda item: item[0], reverse=side == "bid")
    return ordered[:top_n]


def aggregate_orderbook(
    frame: pd.DataFrame,
    frequency: str,
    timestamp_column: str = "timestamp",
    symbol_column: str = "symbol",
    side_column: str = "side",
    price_column: str = "price",
    size_column: str = "size",
    size_mode: str = "absolute",
    top_n: int = 1,
) -> pd.DataFrame:
    """Replay order-book updates and emit the last book state in each bucket."""
    if size_mode not in {"absolute", "delta"}:
        raise ValueError("size_mode must be 'absolute' or 'delta'")
    if top_n < 1:
        raise ValueError("top_n must be at least 1")
    data = _normalise_frame(frame, timestamp_column, symbol_column)
    if data.empty:
        return pd.DataFrame()
    data[price_column] = pd.to_numeric(data[price_column], errors="coerce")
    data[size_column] = pd.to_numeric(data[size_column], errors="coerce")
    data = data.dropna(subset=[price_column, size_column])
    data["_bucket"] = data[timestamp_column].dt.floor(frequency)
    rows = []
    for symbol, symbol_data in data.groupby(symbol_column, sort=False, observed=True):
        book = {"bid": {}, "ask": {}}
        current_bucket = None
        updates = 0
        bucket_index = symbol_data.columns.get_loc("_bucket")
        side_index = symbol_data.columns.get_loc(side_column)
        price_index = symbol_data.columns.get_loc(price_column)
        size_index = symbol_data.columns.get_loc(size_column)
        for row in symbol_data.itertuples(index=False, name=None):
            bucket = row[bucket_index]
            if current_bucket is not None and bucket != current_bucket:
                rows.append(_book_row(symbol, current_bucket, book, updates, top_n))
                updates = 0
            current_bucket = bucket
            side = _normalise_side(row[side_index])
            price = float(row[price_index])
            value = float(row[size_index])
            old_value = book[side].get(price, 0.0)
            new_value = value if size_mode == "absolute" else old_value + value
            if new_value <= 0:
                book[side].pop(price, None)
            else:
                book[side][price] = new_value
            updates += 1
        if current_bucket is not None:
            rows.append(_book_row(symbol, current_bucket, book, updates, top_n))
    return pd.DataFrame(rows)


def _normalise_side(value: object) -> str:
    side = str(value).strip().lower()
    if side in {"bid", "buy", "b", "1"}:
        return "bid"
    if side in {"ask", "sell", "a", "-1", "2"}:
        return "ask"
    raise ValueError(f"Unsupported order-book side: {value!r}")


def _book_row(symbol: str, bucket: pd.Timestamp, book: Dict[str, Dict[float, float]], updates: int, top_n: int) -> dict:
    row = {"symbol": symbol, "date": bucket, "book_update_count": updates}
    bids = list(_book_levels(book, "bid", top_n))
    asks = list(_book_levels(book, "ask", top_n))
    for level in range(top_n):
        bid_price, bid_size = bids[level] if level < len(bids) else (np.nan, np.nan)
        ask_price, ask_size = asks[level] if level < len(asks) else (np.nan, np.nan)
        row[f"bid_price_{level + 1}"] = bid_price
        row[f"bid_size_{level + 1}"] = bid_size
        row[f"ask_price_{level + 1}"] = ask_price
        row[f"ask_size_{level + 1}"] = ask_size
    best_bid = row["bid_price_1"]
    best_ask = row["ask_price_1"]
    row["mid"] = (best_bid + best_ask) / 2 if pd.notna(best_bid) and pd.notna(best_ask) else np.nan
    row["spread"] = best_ask - best_bid if pd.notna(best_bid) and pd.notna(best_ask) else np.nan
    return row


def query_questdb(connection_options: dict, query: str, parameters: tuple) -> pd.DataFrame:
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError("QuestDB support requires 'psycopg[binary]'.") from exc
    with psycopg.connect(**connection_options) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, parameters)
            columns = [column.name for column in cursor.description]
            return pd.DataFrame(cursor.fetchall(), columns=columns)


def _build_query(args: argparse.Namespace, start: pd.Timestamp, end: pd.Timestamp) -> tuple[str, tuple]:
    timestamp_column = validate_identifier(args.timestamp_column, "timestamp column")
    symbol_column = validate_identifier(args.symbol_column, "symbol column")
    columns = [timestamp_column, symbol_column]
    if args.data_type == "trades":
        columns += [args.price_column, args.size_column]
        if args.amount_column:
            columns.append(args.amount_column)
    else:
        columns += [args.side_column, args.price_column, args.size_column]
    fields = ", ".join(validate_identifier(column, "column") for column in dict.fromkeys(columns))
    table = validate_identifier(args.table, "table")
    query = f'SELECT {fields} FROM "{table}" WHERE "{timestamp_column}" >= %s AND "{timestamp_column}" < %s'
    parameters = (start.to_pydatetime(), end.to_pydatetime())
    if args.symbol:
        query += f' AND "{symbol_column}" = %s'
        parameters += (args.symbol,)
    query += f' ORDER BY "{symbol_column}", "{timestamp_column}"'
    return query, parameters


def write_symbol_csv(frame: pd.DataFrame, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for symbol, symbol_data in frame.groupby("symbol", sort=True, observed=True):
        path = output_dir / f"{str(symbol).lower()}.csv"
        symbol_data.drop(columns=["symbol"]).sort_values("date").to_csv(path, index=False)
        written.append(path)
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8812)
    parser.add_argument("--user", default="admin")
    parser.add_argument("--password", default="quest")
    parser.add_argument("--database", default="qdb")
    parser.add_argument("--table", required=True)
    parser.add_argument("--date", required=True, help="Date in YYYY-MM-DD format")
    parser.add_argument("--freq", default="1s", help="Pandas frequency such as 1s or 5s")
    parser.add_argument("--data-type", choices=("trades", "orderbook"), required=True)
    parser.add_argument("--symbol")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--timestamp-column", default="timestamp")
    parser.add_argument("--symbol-column", default="symbol")
    parser.add_argument("--price-column", default="price")
    parser.add_argument("--size-column", default="size")
    parser.add_argument("--amount-column")
    parser.add_argument("--side-column", default="side")
    parser.add_argument("--book-size-mode", choices=("absolute", "delta"), default="absolute")
    parser.add_argument("--top-n", type=int, default=1)
    parser.add_argument("--dump-qlib", type=Path, help="Also write a Qlib data directory")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    day = pd.Timestamp(args.date, tz="UTC")
    query, parameters = _build_query(args, day, day + pd.Timedelta(days=1))
    connection_options = {"host": args.host, "port": args.port, "user": args.user, "password": args.password, "dbname": args.database}
    raw = query_questdb(connection_options, query, parameters)
    if args.data_type == "trades":
        result = aggregate_trades(raw, args.freq, args.timestamp_column, args.symbol_column, args.price_column, args.size_column, args.amount_column)
    else:
        result = aggregate_orderbook(raw, args.freq, args.timestamp_column, args.symbol_column, args.side_column, args.price_column, args.size_column, args.book_size_mode, args.top_n)
    paths = write_symbol_csv(result, args.output_dir)
    print(f"Wrote {len(paths)} CSV file(s) to {args.output_dir}")
    if args.dump_qlib:
        from dump_bin import DumpDataAll

        DumpDataAll(data_path=str(args.output_dir), qlib_dir=str(args.dump_qlib), freq=args.freq, date_field_name="date", file_suffix=".csv").dump()
        print(f"Wrote Qlib data to {args.dump_qlib}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())