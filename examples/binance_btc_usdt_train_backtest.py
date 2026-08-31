"""Train and backtest a long-only BTC/USDT minute model with Qlib data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

import qlib
from qlib.data import D


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROVIDER = ROOT / ".qlib" / "qlib_data" / "binance_btc_usdt_1m"
DEFAULT_OUTPUT = ROOT / ".qlib" / "experiments" / "binance_btc_usdt_1m"
FIELDS = ["$open", "$high", "$low", "$close", "$volume"]
FEATURE_NAMES = [
    "ret_1",
    "ret_5",
    "ret_15",
    "ret_60",
    "ret_240",
    "ret_1440",
    "oc_ret",
    "hl_spread",
    "close_position",
    "log_volume",
    "volume_ratio_60",
    "volatility_60",
    "volatility_240",
    "range_width_20",
    "volatility_compression_20_60",
    "volume_ratio_60_prev",
    "breakout_strength_20",
    "squeeze_breakout_score",
    "squeeze_breakout_signal",
    "rsi_60",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider-uri", type=Path, default=DEFAULT_PROVIDER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-start", default="2024-01-01 00:00:00")
    parser.add_argument("--train-end", default="2025-12-31 23:59:00")
    parser.add_argument("--valid-start", default="2026-01-01 00:00:00")
    parser.add_argument("--valid-end", default="2026-03-31 23:59:00")
    parser.add_argument("--test-start", default="2026-04-01 00:00:00")
    parser.add_argument("--test-end", default=None)
    parser.add_argument("--horizon-bars", type=int, default=15)
    parser.add_argument("--rebalance-bars", type=int, default=1)
    parser.add_argument("--fee-rate", type=float, default=0.0004)
    parser.add_argument("--slippage-rate", type=float, default=0.0001)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--min-edge", type=float, default=0.0002)
    parser.add_argument("--exit-threshold", type=float, default=0.0)
    parser.add_argument("--exit-confirm-bars", type=int, default=2)
    parser.add_argument("--atr-window", type=int, default=60)
    parser.add_argument("--initial-stop-atr", type=float, default=2.0)
    parser.add_argument("--trailing-stop-atr", type=float, default=3.0)
    parser.add_argument("--breakout-failure-bars", type=int, default=2)
    parser.add_argument("--num-boost-round", type=int, default=500)
    parser.add_argument("--n-jobs", type=int, default=8)
    return parser.parse_args()


def round_trip_break_even_return(fee_rate: float, slippage_rate: float) -> float:
    entry_multiplier = (1 + slippage_rate) * (1 + fee_rate)
    exit_multiplier = (1 - slippage_rate) * (1 - fee_rate)
    return entry_multiplier / exit_multiplier - 1


def label_exit_datetimes(data: pd.DataFrame, horizon_bars: int) -> pd.Series:
    return data["datetime"].shift(-(horizon_bars + 1))


def build_segment_masks(
    features: pd.DataFrame,
    label_exit_datetime: pd.Series,
    train_start: str,
    train_end: str,
    valid_start: str,
    valid_end: str,
    test_start: str,
    test_end: str,
) -> tuple[dict[str, pd.Series], dict[str, dict[str, object]]]:
    feature_datetime = features["datetime"]
    boundaries = {
        "train": {
            "start": pd.Timestamp(train_start),
            "end": pd.Timestamp(train_end),
            "next_start": pd.Timestamp(valid_start),
        },
        "valid": {
            "start": pd.Timestamp(valid_start),
            "end": pd.Timestamp(valid_end),
            "next_start": pd.Timestamp(test_start),
        },
        "test": {
            "start": pd.Timestamp(test_start),
            "end": pd.Timestamp(test_end),
            "next_start": None,
        },
    }
    masks: dict[str, pd.Series] = {}
    audit: dict[str, dict[str, object]] = {}
    for name, boundary in boundaries.items():
        requested = feature_datetime.between(boundary["start"], boundary["end"])
        if boundary["next_start"] is None:
            complete_labels = label_exit_datetime <= boundary["end"]
        else:
            complete_labels = label_exit_datetime < boundary["next_start"]
        mask = requested & complete_labels
        masks[name] = mask
        kept_datetime = feature_datetime[mask]
        kept_label_exit = label_exit_datetime[mask]
        audit[name] = {
            "requested_start": boundary["start"].strftime("%Y-%m-%d %H:%M:%S"),
            "requested_end": boundary["end"].strftime("%Y-%m-%d %H:%M:%S"),
            "candidate_samples": int(requested.sum()),
            "purged_samples": int((requested & ~complete_labels).sum()),
            "samples": int(mask.sum()),
            "first_signal": (
                kept_datetime.iloc[0].strftime("%Y-%m-%d %H:%M:%S")
                if not kept_datetime.empty
                else None
            ),
            "last_signal": (
                kept_datetime.iloc[-1].strftime("%Y-%m-%d %H:%M:%S")
                if not kept_datetime.empty
                else None
            ),
            "first_label_exit": (
                kept_label_exit.iloc[0].strftime("%Y-%m-%d %H:%M:%S")
                if not kept_label_exit.empty
                else None
            ),
            "last_label_exit": (
                kept_label_exit.iloc[-1].strftime("%Y-%m-%d %H:%M:%S")
                if not kept_label_exit.empty
                else None
            ),
        }
    return masks, audit


def load_market_data(provider_uri: Path, start_time: str, end_time: str | None) -> pd.DataFrame:
    qlib.init(provider_uri=str(provider_uri.resolve()), auto_mount=False, redis_port=-1)
    data = D.features(
        ["BTC_USDT"],
        FIELDS,
        start_time=start_time,
        end_time=end_time,
        freq="1min",
        disk_cache=False,
    ).reset_index()
    data = data.rename(
        columns={
            "$open": "open",
            "$high": "high",
            "$low": "low",
            "$close": "close",
            "$volume": "volume",
        }
    )
    data["datetime"] = pd.to_datetime(data["datetime"])
    data = data.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
    if data.empty:
        raise ValueError("No BTC_USDT data was loaded from the provider")
    if not data["datetime"].is_monotonic_increasing:
        raise ValueError("Loaded datetime values are not increasing")
    if data[["open", "high", "low", "close", "volume"]].isna().any().any():
        raise ValueError("Loaded OHLCV data contains null values")
    return data


def make_dataset(data: pd.DataFrame, horizon_bars: int) -> tuple[pd.DataFrame, pd.Series]:
    close = data["close"].astype(float)
    log_close = np.log(close)
    features = pd.DataFrame(index=data.index)
    for window in (1, 5, 15, 60, 240, 1440):
        features[f"ret_{window}"] = log_close.diff(window)

    candle_range = (data["high"] - data["low"]).replace(0, np.nan)
    features["oc_ret"] = np.log(data["close"] / data["open"])
    features["hl_spread"] = (data["high"] - data["low"]) / data["close"]
    features["close_position"] = (data["close"] - data["low"]) / candle_range
    features["log_volume"] = np.log1p(data["volume"])
    features["volume_ratio_60"] = data["volume"] / data["volume"].rolling(60).median()
    one_bar_return = log_close.diff()
    features["volatility_60"] = one_bar_return.rolling(60).std()
    features["volatility_240"] = one_bar_return.rolling(240).std()

    prior_high_20 = data["high"].rolling(20).max().shift(1)
    prior_low_20 = data["low"].rolling(20).min().shift(1)
    prior_close = close.shift(1)
    prior_channel_width = prior_high_20 - prior_low_20
    features["range_width_20"] = prior_channel_width / (prior_close + 1e-12)

    prior_short_vol = one_bar_return.rolling(20).std().shift(1)
    prior_long_vol = one_bar_return.rolling(60).std().shift(1)
    compression_ratio = prior_short_vol / (prior_long_vol + 1e-12)
    features["volatility_compression_20_60"] = compression_ratio

    prior_volume_median = data["volume"].rolling(60).median().shift(1)
    volume_ratio_prev = data["volume"] / (prior_volume_median + 1e-12)
    features["volume_ratio_60_prev"] = volume_ratio_prev

    breakout_strength = (close - prior_high_20) / (prior_channel_width + 1e-12)
    features["breakout_strength_20"] = breakout_strength

    compression_score = (1 - compression_ratio).clip(lower=0, upper=1)
    volume_score = np.tanh(np.log(volume_ratio_prev.clip(lower=1)))
    breakout_score = np.tanh(breakout_strength.clip(lower=0))
    close_strength = features["close_position"].clip(lower=0, upper=1)
    features["squeeze_breakout_score"] = (
        compression_score * volume_score * breakout_score * close_strength
    )
    first_breakout = (close > prior_high_20) & (prior_close <= prior_high_20.shift(1))
    features["squeeze_breakout_signal"] = (
        (compression_ratio <= 0.7)
        & (volume_ratio_prev >= 1.5)
        & (breakout_strength > 0)
        & (close_strength >= 0.6)
        & first_breakout
    ).astype(float)

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(60).mean()
    loss = (-delta.clip(upper=0)).rolling(60).mean()
    features["rsi_60"] = gain / (gain + loss)

    minute_of_day = data["datetime"].dt.hour * 60 + data["datetime"].dt.minute
    day_of_week = data["datetime"].dt.dayofweek
    features["hour_sin"] = np.sin(2 * np.pi * minute_of_day / 1440)
    features["hour_cos"] = np.cos(2 * np.pi * minute_of_day / 1440)
    features["weekday_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    features["weekday_cos"] = np.cos(2 * np.pi * day_of_week / 7)

    entry_price = close.shift(-1)
    exit_price = close.shift(-(horizon_bars + 1))
    label = exit_price / entry_price - 1
    valid = features[FEATURE_NAMES].notna().all(axis=1) & label.notna()
    return features.loc[valid, FEATURE_NAMES], label.loc[valid]


def regression_metrics(label: pd.Series, prediction: pd.Series) -> dict[str, float]:
    error = prediction - label
    return {
        "samples": int(len(label)),
        "mae": float(error.abs().mean()),
        "rmse": float(np.sqrt((error**2).mean())),
        "directional_accuracy": float((np.sign(prediction) == np.sign(label)).mean()),
        "label_mean": float(label.mean()),
        "prediction_mean": float(prediction.mean()),
    }


def average_true_range(data: pd.DataFrame, window: int) -> pd.Series:
    previous_close = data["close"].shift(1)
    true_range = pd.concat(
        [
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(window, min_periods=window).mean()


def run_backtest(
    data: pd.DataFrame,
    prediction: pd.Series,
    test_start: str,
    test_end: str,
    horizon_bars: int,
    rebalance_bars: int,
    threshold: float,
    exit_threshold: float,
    exit_confirm_bars: int,
    fee_rate: float,
    slippage_rate: float,
    atr_window: int,
    initial_stop_atr: float,
    trailing_stop_atr: float,
    breakout_failure_bars: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    test_start_ts = pd.Timestamp(test_start)
    test_end_ts = pd.Timestamp(test_end)
    signal_rows = data.index[(data["datetime"] >= test_start_ts) & (data["datetime"] <= test_end_ts)]
    if len(signal_rows) == 0:
        raise ValueError("No test data is available for the requested period")

    prediction_by_row = prediction.reindex(data.index)
    atr_by_row = average_true_range(data, atr_window)
    prior_high_20 = data["high"].rolling(20, min_periods=20).max().shift(1)
    first_signal = signal_rows[0]
    last_data_row = data.index[data["datetime"] <= test_end_ts][-1]
    signal_rows = range(first_signal, last_data_row, rebalance_bars)
    cash = 1.0
    units = 0.0
    current_trade = None
    trade_rows = []
    equity_rows = []
    signal_schedule = {}

    for signal_row in signal_rows:
        signal_time = data.at[signal_row, "datetime"]
        if signal_time > test_end_ts:
            break
        signal = prediction_by_row.at[signal_row]
        if pd.isna(signal):
            continue
        entry_row = signal_row + 1
        if entry_row > last_data_row:
            break
        breakout_level = prior_high_20.at[signal_row]
        signal_schedule[entry_row] = {
            "prediction": float(signal),
            "signal_time": signal_time,
            "atr": float(atr_by_row.at[signal_row]),
            "breakout_level": (
                float(breakout_level)
                if pd.notna(breakout_level) and data.at[signal_row, "close"] > breakout_level
                else np.nan
            ),
        }

    def close_long(row: int, reference_price: float, reason: str) -> None:
        nonlocal cash, units, current_trade, pending_exit_reason
        effective_exit = reference_price * (1 - slippage_rate)
        cash = units * effective_exit * (1 - fee_rate)
        units = 0.0
        pending_exit_reason = None
        current_trade["exit_time"] = data.at[row, "datetime"]
        current_trade["exit_reference_price"] = reference_price
        current_trade["exit_reason"] = reason
        current_trade["long"] = True
        current_trade["holding_bars"] = row - current_trade.pop("entry_row")
        current_trade["return"] = cash / current_trade.pop("entry_equity") - 1
        current_trade["equity"] = cash
        trade_rows.append(current_trade)
        current_trade = None

    long_position_rows = 0
    active_stop = np.nan
    breakout_failure_count = 0
    weak_prediction_count = 0
    pending_exit_reason = None
    for row in range(first_signal, last_data_row + 1):
        closed_this_row = False
        execution_price = float(data.at[row, "close"])

        if units > 0 and pd.notna(active_stop) and data.at[row, "low"] <= active_stop:
            stop_reference_price = min(float(data.at[row, "open"]), float(active_stop))
            stop_reason = (
                "trailing_stop"
                if active_stop > current_trade["initial_stop_price"]
                else "initial_stop"
            )
            close_long(row, stop_reference_price, stop_reason)
            closed_this_row = True

        if units > 0 and pending_exit_reason is not None:
            close_long(row, execution_price, pending_exit_reason)
            closed_this_row = True

        scheduled_signal = signal_schedule.get(row)
        if units > 0 and scheduled_signal is not None:
            signal = scheduled_signal["prediction"]
            weak_prediction_count = weak_prediction_count + 1 if signal <= exit_threshold else 0
            if weak_prediction_count >= exit_confirm_bars:
                close_long(row, execution_price, "prediction_exit")
                closed_this_row = True

        if units > 0 and row >= current_trade["expires_row"]:
            if scheduled_signal is not None and scheduled_signal["prediction"] > threshold:
                current_trade["expires_row"] = row + horizon_bars
                current_trade["renewals"] += 1
            else:
                close_long(row, execution_price, "horizon_expired")
                closed_this_row = True

        if units == 0 and not closed_this_row and scheduled_signal is not None:
            signal = scheduled_signal["prediction"]
            signal_atr = scheduled_signal["atr"]
            if signal > threshold and np.isfinite(signal_atr) and signal_atr > 0:
                effective_entry = execution_price * (1 + slippage_rate)
                entry_equity = cash
                units = cash / (effective_entry * (1 + fee_rate))
                cash = 0.0
                initial_stop_price = execution_price - initial_stop_atr * signal_atr
                current_trade = {
                    "signal_time": scheduled_signal["signal_time"],
                    "entry_time": data.at[row, "datetime"],
                    "prediction": signal,
                    "entry_reference_price": execution_price,
                    "initial_stop_price": initial_stop_price,
                    "breakout_level": scheduled_signal["breakout_level"],
                    "expires_row": row + horizon_bars,
                    "renewals": 0,
                    "entry_row": row,
                    "entry_equity": entry_equity,
                }
                active_stop = initial_stop_price
                breakout_failure_count = 0
                weak_prediction_count = 0

        if units > 0:
            breakout_level = current_trade["breakout_level"]
            if pd.notna(breakout_level):
                breakout_failure_count = (
                    breakout_failure_count + 1
                    if execution_price < breakout_level
                    else 0
                )
                if breakout_failure_count >= breakout_failure_bars:
                    pending_exit_reason = "breakout_failure"

            current_trade["highest_close"] = max(
                current_trade.get("highest_close", execution_price), execution_price
            )
            current_atr = atr_by_row.at[row]
            if pd.notna(current_atr) and current_atr > 0:
                trailing_stop = current_trade["highest_close"] - trailing_stop_atr * current_atr
                active_stop = max(current_trade["initial_stop_price"], trailing_stop)

        mark_price = float(data.at[row, "close"])
        equity_rows.append(
            {
                "datetime": data.at[row, "datetime"],
                "equity": cash + units * mark_price,
            }
        )
        long_position_rows += int(units > 0)

    final_price = float(data.at[last_data_row, "close"])
    if units > 0:
        close_long(last_data_row, final_price, "end_of_test")
        equity_rows[-1]["equity"] = cash

    trade_columns = [
        "signal_time",
        "entry_time",
        "prediction",
        "entry_reference_price",
        "initial_stop_price",
        "breakout_level",
        "exit_time",
        "exit_reference_price",
        "exit_reason",
        "long",
        "holding_bars",
        "renewals",
        "return",
        "equity",
    ]
    trades = pd.DataFrame(trade_rows, columns=trade_columns)
    equity_curve = pd.DataFrame(equity_rows)
    equity = equity_curve["equity"]
    drawdown = equity / equity.cummax() - 1
    active_trades = trades[trades["long"]] if not trades.empty else trades
    bar_returns = equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    bars_per_year = 365 * 24 * 60
    bar_std = bar_returns.std(ddof=0) if not bar_returns.empty else 0.0
    total_bars = max(len(equity_rows), 1)
    metrics = {
        "initial_equity": 1.0,
        "final_equity": float(cash),
        "total_return": float(cash - 1),
        "max_drawdown": float(drawdown.min()),
        "trades": int(len(active_trades)),
        "win_rate": float((active_trades["return"] > 0).mean()) if len(active_trades) else 0.0,
        "average_trade_return": float(active_trades["return"].mean()) if len(active_trades) else 0.0,
        "average_holding_bars": float(active_trades["holding_bars"].mean()) if len(active_trades) else 0.0,
        "median_holding_bars": float(active_trades["holding_bars"].median()) if len(active_trades) else 0.0,
        "p90_holding_bars": float(active_trades["holding_bars"].quantile(0.9)) if len(active_trades) else 0.0,
        "max_holding_bars": int(active_trades["holding_bars"].max()) if len(active_trades) else 0,
        "total_renewals": int(active_trades["renewals"].sum()) if len(active_trades) else 0,
        "average_renewals": float(active_trades["renewals"].mean()) if len(active_trades) else 0.0,
        "exit_reason_counts": {
            str(reason): int(count)
            for reason, count in active_trades["exit_reason"].value_counts().items()
        },
        "sharpe": float(bar_returns.mean() / bar_std * np.sqrt(bars_per_year))
        if bar_std and len(bar_returns) > 1
        else 0.0,
        "long_exposure": float(long_position_rows / total_bars),
    }
    return metrics, trades


def main() -> None:
    args = parse_args()
    positive_integer_args = {
        "horizon-bars": args.horizon_bars,
        "rebalance-bars": args.rebalance_bars,
        "exit-confirm-bars": args.exit_confirm_bars,
        "atr-window": args.atr_window,
        "breakout-failure-bars": args.breakout_failure_bars,
    }
    if any(value < 1 for value in positive_integer_args.values()):
        raise ValueError(
            f"These arguments must be positive: "
            f"{', '.join(name for name, value in positive_integer_args.items() if value < 1)}"
        )
    if args.initial_stop_atr <= 0 or args.trailing_stop_atr <= 0:
        raise ValueError("initial-stop-atr and trailing-stop-atr must be positive")
    if args.fee_rate < 0 or args.slippage_rate < 0 or args.min_edge < 0:
        raise ValueError("fee-rate, slippage-rate, and min-edge cannot be negative")

    break_even_return = round_trip_break_even_return(args.fee_rate, args.slippage_rate)
    entry_threshold = (
        args.threshold
        if args.threshold is not None
        else break_even_return + args.min_edge
    )

    data = load_market_data(args.provider_uri, args.train_start, args.test_end)
    actual_test_end = args.test_end or data["datetime"].max().strftime("%Y-%m-%d %H:%M:%S")
    features, labels = make_dataset(data, args.horizon_bars)
    features["datetime"] = data.loc[features.index, "datetime"]
    label_exit_datetime = label_exit_datetimes(data, args.horizon_bars).reindex(features.index)
    segment_masks, segment_audit = build_segment_masks(
        features,
        label_exit_datetime,
        args.train_start,
        args.train_end,
        args.valid_start,
        args.valid_end,
        args.test_start,
        actual_test_end,
    )
    train_mask = segment_masks["train"]
    valid_mask = segment_masks["valid"]
    test_mask = segment_masks["test"]
    feature_columns = FEATURE_NAMES
    if not train_mask.any() or not valid_mask.any() or not test_mask.any():
        raise ValueError("One or more train/valid/test segments are empty")

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=args.num_boost_round,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=10.0,
        n_jobs=args.n_jobs,
        verbosity=-1,
    )
    model.fit(
        features.loc[train_mask, feature_columns],
        labels.loc[train_mask],
        eval_set=[
            (features.loc[train_mask, feature_columns], labels.loc[train_mask]),
            (features.loc[valid_mask, feature_columns], labels.loc[valid_mask]),
        ],
        eval_names=["train", "valid"],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    valid_prediction = pd.Series(
        model.predict(features.loc[valid_mask, feature_columns]), index=features.index[valid_mask]
    )
    test_prediction = pd.Series(
        model.predict(features.loc[test_mask, feature_columns]), index=features.index[test_mask]
    )
    valid_metrics = regression_metrics(labels.loc[valid_mask], valid_prediction)
    test_metrics = regression_metrics(labels.loc[test_mask], test_prediction)

    prediction_by_row = pd.Series(index=data.index, dtype=float)
    prediction_by_row.loc[test_prediction.index] = test_prediction
    backtest_metrics, trades = run_backtest(
        data=data,
        prediction=prediction_by_row,
        test_start=args.test_start,
        test_end=actual_test_end,
        horizon_bars=args.horizon_bars,
        rebalance_bars=args.rebalance_bars,
        threshold=entry_threshold,
        exit_threshold=args.exit_threshold,
        exit_confirm_bars=args.exit_confirm_bars,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        atr_window=args.atr_window,
        initial_stop_atr=args.initial_stop_atr,
        trailing_stop_atr=args.trailing_stop_atr,
        breakout_failure_bars=args.breakout_failure_bars,
    )

    buy_hold_start = data.loc[data["datetime"] >= pd.Timestamp(args.test_start), "close"].iloc[0]
    buy_hold_end = data.loc[data["datetime"] <= pd.Timestamp(actual_test_end), "close"].iloc[-1]
    buy_hold_return = float(buy_hold_end / buy_hold_start - 1)
    result = {
        "provider_uri": str(args.provider_uri.resolve()),
        "instrument": "BTC_USDT",
        "frequency": "1min",
        "model": {
            "type": "lightgbm.LGBMRegressor",
            "best_iteration": int(getattr(model, "best_iteration_", 0) or 0),
            "fitted_estimators": int(getattr(model, "n_estimators_", args.num_boost_round)),
        },
        "dataset": {
            "feature_rows": int(len(features)),
            "label_horizon_bars": args.horizon_bars,
            "label_exit_offset_bars": args.horizon_bars + 1,
            "segments": segment_audit,
        },
        "train": {"start": args.train_start, "end": args.train_end},
        "valid": {"start": args.valid_start, "end": args.valid_end, **valid_metrics},
        "test": {"start": args.test_start, "end": actual_test_end, **test_metrics},
        "backtest": {
            "horizon_bars": args.horizon_bars,
            "rebalance_bars": args.rebalance_bars,
            "fee_rate_per_side": args.fee_rate,
            "slippage_rate_per_side": args.slippage_rate,
            "round_trip_break_even_return": break_even_return,
            "minimum_edge": args.min_edge,
            "entry_prediction_threshold": entry_threshold,
            "exit_prediction_threshold": args.exit_threshold,
            "exit_confirm_bars": args.exit_confirm_bars,
            "atr_window": args.atr_window,
            "initial_stop_atr": args.initial_stop_atr,
            "trailing_stop_atr": args.trailing_stop_atr,
            "breakout_failure_bars": args.breakout_failure_bars,
            "buy_and_hold_return": buy_hold_return,
            **backtest_metrics,
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output_dir / "model.joblib")
    trades.to_csv(args.output_dir / "trades.csv", index=False)
    (args.output_dir / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()