from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "examples" / "binance_btc_usdt_train_backtest.py"
SPEC = importlib.util.spec_from_file_location("binance_btc_usdt_train_backtest", SCRIPT_PATH)
BACKTEST = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BACKTEST)


class TestBinanceBtcUsdtRiskExits(unittest.TestCase):
    @staticmethod
    def run_case(
        close: np.ndarray,
        predictions: dict[int, float],
        *,
        high: np.ndarray | None = None,
        low: np.ndarray | None = None,
        horizon_bars: int = 15,
        initial_stop_atr: float = 2.0,
        trailing_stop_atr: float = 3.0,
    ) -> tuple[dict, pd.DataFrame]:
        high = close + 1 if high is None else high
        low = close - 1 if low is None else low
        data = pd.DataFrame(
            {
                "datetime": pd.date_range("2026-04-01", periods=len(close), freq="min"),
                "open": close,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.ones(len(close)),
            }
        )
        prediction = pd.Series(np.nan, index=data.index)
        for row, value in predictions.items():
            prediction.at[row] = value

        return BACKTEST.run_backtest(
            data=data,
            prediction=prediction,
            test_start=str(data["datetime"].iloc[0]),
            test_end=str(data["datetime"].iloc[-1]),
            horizon_bars=horizon_bars,
            rebalance_bars=1,
            threshold=0.001,
            exit_threshold=0.0,
            exit_confirm_bars=2,
            fee_rate=0.0004,
            slippage_rate=0.0001,
            atr_window=3,
            initial_stop_atr=initial_stop_atr,
            trailing_stop_atr=trailing_stop_atr,
            breakout_failure_bars=2,
        )

    def test_horizon_requires_strong_prediction_to_renew(self):
        close = np.full(30, 100.0)
        _, trades = self.run_case(
            close,
            {5: 0.002, 8: 0.002, 11: 0.0005},
            horizon_bars=3,
        )

        self.assertEqual(trades.loc[0, "exit_reason"], "horizon_expired")
        self.assertEqual(trades.loc[0, "holding_bars"], 6)
        self.assertEqual(trades.loc[0, "renewals"], 1)

    def test_horizon_expires_without_new_prediction(self):
        close = np.full(30, 100.0)
        _, trades = self.run_case(close, {5: 0.002}, horizon_bars=3)

        self.assertEqual(trades.loc[0, "exit_reason"], "horizon_expired")
        self.assertEqual(trades.loc[0, "holding_bars"], 3)

    def test_sixty_minute_horizon_expires_after_sixty_bars(self):
        close = np.full(100, 100.0)
        _, trades = self.run_case(close, {5: 0.002}, horizon_bars=60)

        self.assertEqual(trades.loc[0, "exit_reason"], "horizon_expired")
        self.assertEqual(trades.loc[0, "holding_bars"], 60)

    def test_two_weak_predictions_exit_position(self):
        close = np.full(30, 100.0)
        _, trades = self.run_case(close, {5: 0.002, 6: -0.0001, 7: -0.0001})

        self.assertEqual(trades.loc[0, "exit_reason"], "prediction_exit")
        self.assertEqual(trades.loc[0, "holding_bars"], 2)

    def test_initial_and_trailing_stops(self):
        close = np.full(30, 100.0)
        initial_stop_low = close - 1
        initial_stop_low[8] = 94.0
        _, initial_stop_trades = self.run_case(
            close,
            {5: 0.002},
            low=initial_stop_low,
        )
        self.assertEqual(initial_stop_trades.loc[0, "exit_reason"], "initial_stop")

        trailing_stop_low = close - 1
        trailing_stop_low[7] = 97.0
        _, trailing_stop_trades = self.run_case(
            close,
            {5: 0.002},
            low=trailing_stop_low,
            initial_stop_atr=10.0,
            trailing_stop_atr=1.0,
        )
        self.assertEqual(trailing_stop_trades.loc[0, "exit_reason"], "trailing_stop")

    def test_two_closes_below_breakout_exit_on_next_bar(self):
        close = np.r_[np.full(20, 100.0), 102.0, 101.0, 99.0, 99.0, 99.0, np.full(5, 99.0)]
        _, trades = self.run_case(close, {20: 0.002})

        self.assertEqual(trades.loc[0, "breakout_level"], 101.0)
        self.assertEqual(trades.loc[0, "exit_reason"], "breakout_failure")
        self.assertEqual(trades.loc[0, "holding_bars"], 3)

    def test_pending_breakout_exit_does_not_leak_to_next_trade(self):
        close = np.r_[
            np.full(20, 100.0),
            102.0,
            101.0,
            99.0,
            99.0,
            102.0,
            102.0,
            np.full(4, 102.0),
        ]
        _, trades = self.run_case(
            close,
            {20: 0.002, 23: 0.002, 24: 0.002},
            horizon_bars=15,
        )

        self.assertGreaterEqual(len(trades), 2)
        self.assertEqual(trades.loc[0, "exit_reason"], "breakout_failure")
        self.assertNotEqual(trades.loc[1, "exit_reason"], "breakout_failure")

    def test_cost_filter_can_produce_no_trades(self):
        close = np.full(30, 100.0)
        metrics, trades = self.run_case(close, {5: 0.0005})

        self.assertTrue(trades.empty)
        self.assertEqual(metrics["trades"], 0)
        self.assertEqual(metrics["long_exposure"], 0.0)

    def test_segment_masks_purge_labels_crossing_next_segment(self):
        data = pd.DataFrame(
            {"datetime": pd.date_range("2025-12-31 22:00:00", periods=500, freq="min")}
        )
        features = data.copy()
        label_exit_datetime = BACKTEST.label_exit_datetimes(data, horizon_bars=60)
        masks, audit = BACKTEST.build_segment_masks(
            features,
            label_exit_datetime,
            "2025-12-31 22:00:00",
            "2025-12-31 23:59:00",
            "2026-01-01 00:00:00",
            "2026-01-01 01:59:00",
            "2026-01-01 02:00:00",
            "2026-01-01 06:19:00",
        )

        self.assertEqual(int(masks["train"].sum()), 59)
        self.assertEqual(audit["train"]["purged_samples"], 61)
        self.assertEqual(audit["train"]["last_signal"], "2025-12-31 22:58:00")
        self.assertEqual(audit["train"]["last_label_exit"], "2025-12-31 23:59:00")
        self.assertEqual(audit["valid"]["last_signal"], "2026-01-01 00:58:00")


if __name__ == "__main__":
    unittest.main()