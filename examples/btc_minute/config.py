"""BTC/USDT 分钟级单标的策略配置。

所有时间一律 UTC；7x24 市场无交易日概念，滚动窗口按自然分钟计算。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"


@dataclass(frozen=True)
class DataConfig:
    symbol: str = "BTC/USDT"
    bar_freq: str = "1min"
    trades_table: str = "trades"
    orderbook_table: str = "orderbook"
    # 订单簿聚合成分钟末快照时，先按该频率采样再取 last/mean/std
    book_sample_freq: str = "1s"
    book_top_n: int = 10
    raw_dir: Path = DATA_DIR / "raw"
    bar_path: Path = DATA_DIR / "bars_1min.parquet"


@dataclass(frozen=True)
class LabelConfig:
    # 预测未来 horizon 根 K 线的对数收益（以 vwap 成交近似真实执行价）
    # 需与 TradeConfig.min_hold_minutes 保持一致，否则信号周期与持仓周期错配
    horizon: int = 30
    price_col: str = "vwap"
    # t 收盘出信号、t+lag 根 bar 才成交，标签必须从执行价起算
    execution_lag: int = 1
    # 三分类阈值以滚动波动率为单位，None 表示做回归
    vol_window: int = 240
    threshold_in_vol: float | None = 0.5


@dataclass(frozen=True)
class SplitConfig:
    """Purged walk-forward：训练段与验证段之间留 embargo 隔离标签泄漏。"""

    train_minutes: int = 60 * 24 * 90
    valid_minutes: int = 60 * 24 * 14
    test_minutes: int = 60 * 24 * 14
    step_minutes: int = 60 * 24 * 14
    embargo_minutes: int = 60 * 6


@dataclass(frozen=True)
class CostConfig:
    taker_fee: float = 4.5e-4
    maker_fee: float = 2.0e-4
    # 单边滑点，按 mid 的相对值；无盘口数据时的兜底估计
    slippage: float = 1.0e-4
    # 冲击成本系数：impact = coef * (order_notional / bar_notional) ** 0.5
    impact_coef: float = 1.0e-3


@dataclass(frozen=True)
class TradeConfig:
    max_position: float = 1.0
    # 信号绝对值低于该分位数时不开仓；taker 成本下必须取高以压换手
    entry_quantile: float = 0.97
    exit_quantile: float = 0.6
    # 信号平滑窗口，降低预测噪声引起的无效翻仓
    signal_smooth_minutes: int = 5
    # 最短持仓与冷却，抑制分钟级高频抖动
    min_hold_minutes: int = 30
    cooldown_minutes: int = 10
    # 单笔止损/止盈（对数收益）
    stop_loss: float = 3.0e-3
    take_profit: float = 6.0e-3
    # 日内回撤熔断，触发后停止开新仓 halt_minutes
    daily_drawdown_halt: float = 0.02
    halt_minutes: int = 60 * 4


@dataclass(frozen=True)
class ModelConfig:
    name: str = "lightgbm"
    params: dict = field(
        default_factory=lambda: {
            "objective": "regression",
            "metric": "l2",
            "learning_rate": 0.02,
            "num_leaves": 63,
            "min_data_in_leaf": 500,
            "feature_fraction": 0.7,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "lambda_l2": 10.0,
            "verbosity": -1,
        }
    )
    num_boost_round: int = 2000
    early_stopping_rounds: int = 100


@dataclass(frozen=True)
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    label: LabelConfig = field(default_factory=LabelConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    trade: TradeConfig = field(default_factory=TradeConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    seed: int = 42


CONFIG = Config()
