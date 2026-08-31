"""震荡市均值回归策略完整流程：因子/标签 -> 训练 -> 信号分析 -> 回测。

用法：
    python examples/mean_reversion/workflow_mr.py
"""

import sys
from pathlib import Path

import qlib
from qlib.constant import REG_CN
from qlib.tests.config import CSI300_BENCH
from qlib.utils import flatten_dict, init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import PortAnaRecord, SigAnaRecord, SignalRecord

sys.path.append(str(Path(__file__).resolve().parent))

PROVIDER_URI = "F:/Python/AiQuantization/qlib/.qlib/qlib_data/cn_data"
MARKET = "csi300"
BENCHMARK = CSI300_BENCH

TRAIN = ("2008-01-01", "2014-12-31")
VALID = ("2015-01-01", "2016-12-31")
TEST = ("2017-01-01", "2020-08-01")

# 均值回归是短线：2 日持仓期，标签与策略的时间止损须匹配
HOLDING_DAYS = 2

TASK = {
    "model": {
        "class": "LGBModel",
        "module_path": "qlib.contrib.model.gbdt",
        "kwargs": {
            "loss": "mse",
            "learning_rate": 0.02,
            "num_leaves": 64,
            "max_depth": 6,
            "colsample_bytree": 0.8,
            "subsample": 0.8,
            "lambda_l1": 10,
            "lambda_l2": 50,
            "num_threads": 8,
        },
    },
    "dataset": {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "MeanReversionHandler",
                "module_path": "handler",
                "kwargs": {
                    "start_time": TRAIN[0],
                    "end_time": TEST[1],
                    "fit_start_time": TRAIN[0],
                    "fit_end_time": TRAIN[1],
                    "instruments": MARKET,
                    "benchmark": BENCHMARK,
                    "label_type": "risk_adj_ret",
                    "holding_days": HOLDING_DAYS,
                },
            },
            "segments": {"train": TRAIN, "valid": VALID, "test": TEST},
        },
    },
}

PORT_ANALYSIS_CONFIG = {
    "executor": {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
    },
    "strategy": {
        "class": "MeanReversionStrategy",
        "module_path": "strategy",
        "kwargs": {
            "signal": "<PRED>",
            "buy_thresh": 0.5,
            "exit_thresh": 0.1,
            "max_n": 20,
            "min_n": 3,
            "max_weight": 0.08,
            "target_vol": 0.015,
            "vol_window": 20,
            "max_hold_days": 5,
            "market": MARKET,
            "benchmark": BENCHMARK,
            "regime_er_thresh": 0.35,
            "crash_thresh": -0.06,
            "risk_degree": 0.95,
        },
    },
    "backtest": {
        "start_time": TEST[0],
        "end_time": TEST[1],
        "account": 100000000,
        "benchmark": BENCHMARK,
        "exchange_kwargs": {
            "freq": "day",
            "limit_threshold": 0.095,
            "deal_price": "close",
            "open_cost": 0.0005,
            "close_cost": 0.0015,
            "min_cost": 5,
        },
    },
}


if __name__ == "__main__":
    qlib.init(provider_uri=PROVIDER_URI, region=REG_CN)

    model = init_instance_by_config(TASK["model"])
    dataset = init_instance_by_config(TASK["dataset"])

    print(dataset.prepare("train").head())

    with R.start(experiment_name="mean_reversion"):
        R.log_params(**flatten_dict(TASK))
        model.fit(dataset)
        R.save_objects(**{"params.pkl": model})

        recorder = R.get_recorder()
        SignalRecord(model, dataset, recorder).generate()
        SigAnaRecord(recorder).generate()
        PortAnaRecord(recorder, PORT_ANALYSIS_CONFIG, "day").generate()
