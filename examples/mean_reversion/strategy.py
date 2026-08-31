"""震荡市均值回归策略：市场状态门控 / 限时持仓 / 目标止盈 / 波动率定权。

与趋势跟踪的迟滞持有相反，均值回归的持仓逻辑是"快进快出"：
- regime 门控：只有指数处于震荡状态(低效率比且未崩盘)时才开新仓，趋势市自动停工；
- 限时持仓：回归要么在几天内发生要么不发生，超过 max_hold_days 无条件离场（时间止损）；
- 目标止盈：预测值跌回 exit_thresh 以下（偏离已收敛）立即卖出，不恋战；
- 崩盘熔断：指数短期跌幅超过 crash_thresh 时清仓，避免"接飞刀"死于系统性下跌。
"""

import pandas as pd

from qlib.contrib.strategy.signal_strategy import WeightStrategyBase
from qlib.data import D
from qlib.log import get_module_logger


class MeanReversionStrategy(WeightStrategyBase):
    def __init__(
        self,
        *,
        buy_thresh: float = 0.5,
        exit_thresh: float = 0.1,
        max_n: int = 20,
        min_n: int = 3,
        max_weight: float = 0.08,
        target_vol: float = 0.015,
        vol_window: int = 20,
        max_hold_days: int = 5,
        market: str = "csi300",
        benchmark: str = "SH000300",
        regime_er_thresh: float = 0.35,
        crash_thresh: float = -0.06,
        equal_weight: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        buy_thresh : float
            入场阈值。均值回归是逆势接单，须比趋势跟踪(0.3)更苛刻，只做深度偏离。
        exit_thresh : float
            止盈阈值：预测值回落至此以下说明偏离已基本收敛，立即离场。
        max_hold_days : int
            时间止损：持有超过该天数仍未回归则无条件卖出。
        regime_er_thresh : float
            指数 20 日 Kaufman 效率比高于该值视为趋势市，停止开新仓。
        crash_thresh : float
            指数 5 日收益低于该值视为崩盘，清仓所有持仓（熔断）。
        其余参数含义与 TrendFollowingStrategy 相同。
        """
        super().__init__(**kwargs)
        if exit_thresh > buy_thresh:
            raise ValueError("exit_thresh must not exceed buy_thresh")
        self.buy_thresh = buy_thresh
        self.exit_thresh = exit_thresh
        self.max_n = max_n
        self.min_n = min_n
        self.max_weight = max_weight
        self.target_vol = target_vol
        self.vol_window = vol_window
        self.max_hold_days = max_hold_days
        self.market = market
        self.benchmark = benchmark
        self.regime_er_thresh = regime_er_thresh
        self.crash_thresh = crash_thresh
        self.equal_weight = equal_weight
        self.logger = get_module_logger("MeanReversionStrategy")
        self._vol_panel = None
        self._regime_panel = None
        self._entry_dates: dict = {}  # stock_id -> 建仓日，用于时间止损

    def _load_panels(self):
        start_time, end_time = self.trade_calendar.get_all_time()
        vol_field = f"Std($close/Ref($close,1)-1,{self.vol_window})"
        df = D.features(D.instruments(self.market), [vol_field], start_time=start_time, end_time=end_time)
        self._vol_panel = df[vol_field].unstack(level="instrument").sort_index()

        er_field = "Abs($close/Ref($close,20)-1)/(Sum(Abs($close/Ref($close,1)-1),20)+1e-12)"
        ret5_field = "$close/Ref($close,5)-1"
        mkt = D.features([self.benchmark], [er_field, ret5_field], start_time=start_time, end_time=end_time)
        self._regime_panel = mkt.droplevel("instrument").sort_index().rename(
            columns={er_field: "er", ret5_field: "ret5"}
        )

    def _asof(self, panel: pd.DataFrame, date: pd.Timestamp):
        idx = panel.index.searchsorted(date, side="right") - 1
        return None if idx < 0 else panel.iloc[idx]

    def generate_target_weight_position(self, score, current, trade_start_time, trade_end_time):
        if self._vol_panel is None:
            self._load_panels()
        if isinstance(score, pd.DataFrame):
            score = score.iloc[:, 0]
        score = score.dropna()

        held = set(current.get_stock_list())
        self._entry_dates = {k: v for k, v in self._entry_dates.items() if k in held}

        # 崩盘熔断：系统性下跌时均值回归的"均值"本身在坍塌，无条件清仓
        regime = self._asof(self._regime_panel, trade_start_time)
        if regime is not None and regime["ret5"] <= self.crash_thresh:
            self.logger.info(f"{trade_start_time:%Y-%m-%d}: market crash ({regime['ret5']:.2%}), flatten all")
            self._entry_dates.clear()
            return {}

        # 出场：止盈(预测值收敛) 或 时间止损(超期未回归)
        keep = {}
        for code in held:
            entry = self._entry_dates.get(code, trade_start_time)
            held_days = max(0, self._vol_panel.index.searchsorted(trade_start_time) - self._vol_panel.index.searchsorted(entry))
            if held_days >= self.max_hold_days:
                continue
            if score.get(code, -1.0) < self.exit_thresh:
                continue
            keep[code] = score.get(code)
        keep = pd.Series(keep, dtype=float)

        # 入场门控：趋势市(高 ER)不开新仓，只管理存量持仓
        trending = regime is not None and regime["er"] >= self.regime_er_thresh
        if trending:
            entry = pd.Series(dtype=float)
        else:
            entry = score[(score >= self.buy_thresh) & ~score.index.isin(held)]

        candidates = pd.concat([keep, entry]).sort_values(ascending=False)
        if len(candidates) > self.max_n:
            keep_part = keep.sort_values(ascending=False).iloc[: self.max_n]
            room = self.max_n - len(keep_part)
            candidates = pd.concat([keep_part, entry.sort_values(ascending=False).iloc[:room]])

        if len(candidates) < self.min_n:
            self._entry_dates = {k: v for k, v in self._entry_dates.items() if k in candidates.index}
            return {}

        for code in candidates.index:
            self._entry_dates.setdefault(code, trade_start_time)

        if self.equal_weight:
            weights = pd.Series(1.0 / len(candidates), index=candidates.index)
        else:
            vol_row = self._asof(self._vol_panel, trade_start_time)
            vol = (vol_row.reindex(candidates.index) if vol_row is not None else pd.Series(index=candidates.index)).astype(float)
            vol = vol.fillna(vol.median() if vol.notna().any() else self.target_vol)
            weights = (self.target_vol / vol.clip(lower=1e-4)).clip(upper=self.max_weight)

        total = weights.sum()
        if total > 1.0:
            weights = weights / total
        return weights[weights > 1e-6].to_dict()
