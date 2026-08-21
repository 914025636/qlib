"""趋势跟踪策略：阈值入场 / 迟滞出场 / 波动率目标定权 / 允许空仓。

与 TopkDropoutStrategy 的区别在于不再固定持有前 K 只：
- 只有预测值超过 buy_thresh 的股票才建仓，信号不足时自动降低仓位甚至空仓；
- 已持仓股票只要预测值仍高于 sell_thresh 就继续持有，形成迟滞区间，避免边界反复换手；
- 每只股票的权重与其波动率成反比（波动率目标法），使组合风险稳定而非等权。
"""

import numpy as np
import pandas as pd

from qlib.contrib.strategy.signal_strategy import WeightStrategyBase
from qlib.data import D
from qlib.log import get_module_logger


class TrendFollowingStrategy(WeightStrategyBase):
    def __init__(
        self,
        *,
        buy_thresh: float = 0.3,
        sell_thresh: float = 0.0,
        max_n: int = 50,
        min_n: int = 5,
        max_weight: float = 0.05,
        target_vol: float = 0.02,
        vol_window: int = 20,
        market: str = "csi300",
        equal_weight: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        buy_thresh : float
            入场阈值。标签为 risk_adj_ret 时其单位是"未来持仓期能赚几个 sigma"，
            0.3 表示只买预测收益约 0.3 倍波动以上的股票。
        sell_thresh : float
            出场阈值，须小于 buy_thresh 以形成迟滞区间。
        max_n : int
            最大持仓只数，用于分散风险。
        min_n : int
            合格信号少于该数量时视为无趋势行情，直接空仓。
        max_weight : float
            单只股票权重上限。
        target_vol : float
            单只股票的目标日波动贡献，权重 = target_vol / 个股波动率。
        vol_window : int
            估计个股波动率的窗口长度。
        market : str
            用于预取波动率数据的股票池，需与 handler 的 instruments 一致。
        equal_weight : bool
            为 True 时退化为合格信号等权，便于对照波动率定权是否有效。
        """
        super().__init__(**kwargs)
        if sell_thresh > buy_thresh:
            raise ValueError("sell_thresh must not exceed buy_thresh")
        self.buy_thresh = buy_thresh
        self.sell_thresh = sell_thresh
        self.max_n = max_n
        self.min_n = min_n
        self.max_weight = max_weight
        self.target_vol = target_vol
        self.vol_window = vol_window
        self.market = market
        self.equal_weight = equal_weight
        self.logger = get_module_logger("TrendFollowingStrategy")
        self._vol_panel = None

    def _get_vol(self, date: pd.Timestamp) -> pd.Series:
        """取 date 当日各股票的历史波动率，首次调用时一次性载入整段回测区间。"""
        if self._vol_panel is None:
            start_time, end_time = self.trade_calendar.get_all_time()
            field = f"Std($close/Ref($close,1)-1,{self.vol_window})"
            df = D.features(D.instruments(self.market), [field], start_time=start_time, end_time=end_time)
            self._vol_panel = df[field].unstack(level="instrument").sort_index()
        idx = self._vol_panel.index.searchsorted(date, side="right") - 1
        if idx < 0:
            return pd.Series(dtype=float)
        return self._vol_panel.iloc[idx].dropna()

    def generate_target_weight_position(self, score, current, trade_start_time, trade_end_time):
        if isinstance(score, pd.DataFrame):
            score = score.iloc[:, 0]
        score = score.dropna()
        if score.empty:
            return {}

        held = set(current.get_stock_list())
        # 新股票要过 buy_thresh，持仓股票只需过 sell_thresh，中间地带保持原状不动
        keep = score[(score >= self.sell_thresh) & score.index.isin(held)]
        entry = score[(score >= self.buy_thresh) & ~score.index.isin(held)]

        candidates = pd.concat([keep, entry]).sort_values(ascending=False)
        # 持仓优先保留，剩余名额才给新信号，避免因新信号更强而无谓换手
        if len(candidates) > self.max_n:
            keep_part = keep.sort_values(ascending=False).iloc[: self.max_n]
            room = self.max_n - len(keep_part)
            candidates = pd.concat([keep_part, entry.sort_values(ascending=False).iloc[:room]])

        if len(candidates) < self.min_n:
            self.logger.info(f"{trade_start_time:%Y-%m-%d}: only {len(candidates)} signals, go to cash")
            return {}

        if self.equal_weight:
            weights = pd.Series(1.0 / len(candidates), index=candidates.index)
        else:
            vol = self._get_vol(trade_start_time).reindex(candidates.index)
            vol = vol.fillna(vol.median() if vol.notna().any() else self.target_vol)
            weights = (self.target_vol / vol.clip(lower=1e-4)).clip(upper=self.max_weight)

        # 权重之和不足 1 时余下部分留作现金，这正是趋势跟踪的择时来源
        total = weights.sum()
        if total > 1.0:
            weights = weights / total
        return weights[weights > 1e-6].to_dict()
