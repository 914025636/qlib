"""事件驱动式分钟级回测：含手续费、滑点、冲击成本与风控。

时序约定：t 时刻收盘拿到信号，t+1 根 bar 以 vwap 成交，避免用当根收盘价成交。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CostConfig, TradeConfig

MINUTES_PER_YEAR = 365 * 24 * 60


def signal_to_position(pred: pd.Series, cfg: TradeConfig, window: int = 60 * 24 * 7) -> pd.Series:
    """滚动分位数门限 + 最短持仓 + 冷却，把连续预测转为目标仓位。"""
    if cfg.signal_smooth_minutes > 1:
        pred = pred.rolling(cfg.signal_smooth_minutes, min_periods=1).mean()
    strength = pred.abs()
    entry_thr = strength.rolling(window, min_periods=window // 4).quantile(cfg.entry_quantile).shift(1)
    exit_thr = strength.rolling(window, min_periods=window // 4).quantile(cfg.exit_quantile).shift(1)

    pos = np.zeros(len(pred))
    hold = 0
    cooldown = 0
    values = pred.to_numpy()
    e_in = entry_thr.to_numpy()
    e_out = exit_thr.to_numpy()

    for i in range(len(values)):
        prev = pos[i - 1] if i else 0.0
        v, thr_in, thr_out = values[i], e_in[i], e_out[i]
        if np.isnan(v) or np.isnan(thr_in):
            pos[i] = prev
            continue
        if cooldown > 0:
            cooldown -= 1
            pos[i] = 0.0
            continue
        if prev == 0.0:
            if abs(v) >= thr_in:
                pos[i] = np.sign(v) * cfg.max_position
                hold = 1
            else:
                pos[i] = 0.0
        else:
            hold += 1
            reverse = np.sign(v) != np.sign(prev) and abs(v) >= thr_in
            weak = abs(v) < thr_out
            if hold < cfg.min_hold_minutes and not reverse:
                pos[i] = prev
            elif reverse:
                pos[i] = np.sign(v) * cfg.max_position
                hold = 1
            elif weak:
                pos[i] = 0.0
                hold = 0
                cooldown = cfg.cooldown_minutes
            else:
                pos[i] = prev
    return pd.Series(pos, index=pred.index, name="target_pos")


def run_backtest(
    bars: pd.DataFrame,
    pred: pd.Series,
    cost: CostConfig,
    trade: TradeConfig,
    capital: float = 1.0,
) -> tuple[pd.DataFrame, dict]:
    idx = pred.index.intersection(bars.index)
    bars = bars.loc[idx]
    target = signal_to_position(pred.loc[idx], trade)

    exec_price = bars["vwap"].shift(-1).ffill()
    ret = np.log(exec_price.shift(-1) / exec_price)  # 持仓期收益，与执行价对齐
    bar_notional = bars["amount"].rolling(60, min_periods=10).mean().shift(1)

    pos = np.zeros(len(idx))
    entry_price = np.nan
    hold = 0
    halt_until = -1
    equity = np.ones(len(idx)) * capital
    turnover = np.zeros(len(idx))
    cost_series = np.zeros(len(idx))
    pnl = np.zeros(len(idx))

    tgt = target.to_numpy()
    px = exec_price.to_numpy()
    r = ret.to_numpy()
    notional = bar_notional.to_numpy()
    day = idx.floor("D")
    day_peak = capital
    cur_day = day[0] if len(day) else None

    for i in range(len(idx)):
        prev_pos = pos[i - 1] if i else 0.0
        prev_eq = equity[i - 1] if i else capital

        if day[i] != cur_day:
            cur_day = day[i]
            day_peak = prev_eq

        desired = tgt[i]
        if i <= halt_until:
            desired = 0.0

        # 止损止盈：按当前浮动收益判断
        if prev_pos != 0.0 and not np.isnan(entry_price) and px[i] > 0:
            unreal = np.log(px[i] / entry_price) * np.sign(prev_pos)
            if unreal <= -trade.stop_loss or unreal >= trade.take_profit:
                desired = 0.0

        if prev_pos != 0.0 and hold < trade.min_hold_minutes and desired == 0.0 and i > halt_until:
            desired = prev_pos  # 未到最短持仓且非风控触发，维持

        delta = desired - prev_pos
        trade_cost = 0.0
        if abs(delta) > 1e-9:
            order_notional = abs(delta) * prev_eq
            impact = 0.0
            if notional[i] and notional[i] > 0:
                impact = cost.impact_coef * np.sqrt(order_notional / notional[i])
            trade_cost = abs(delta) * (cost.taker_fee + cost.slippage + impact)
            entry_price = px[i] if desired != 0.0 else np.nan
            hold = 0
        hold += 1

        bar_ret = 0.0 if np.isnan(r[i]) else desired * r[i]
        step = bar_ret - trade_cost
        equity[i] = prev_eq * float(np.exp(step)) if abs(step) < 1 else prev_eq
        pos[i] = desired
        turnover[i] = abs(delta)
        cost_series[i] = trade_cost
        pnl[i] = step

        day_peak = max(day_peak, equity[i])
        if day_peak > 0 and equity[i] / day_peak - 1 <= -trade.daily_drawdown_halt:
            halt_until = i + trade.halt_minutes
            pos[i] = 0.0

    result = pd.DataFrame(
        {
            "pred": pred.loc[idx],
            "target_pos": target,
            "position": pos,
            "bar_ret": pnl,
            "cost": cost_series,
            "turnover": turnover,
            "equity": equity,
        },
        index=idx,
    )
    return result, performance(result)


def performance(result: pd.DataFrame) -> dict:
    ret = result["bar_ret"].fillna(0.0)
    equity = result["equity"]
    n = len(ret)
    if n == 0:
        return {}
    total = float(equity.iloc[-1] / equity.iloc[0] - 1)
    years = n / MINUTES_PER_YEAR
    ann_ret = float((1 + total) ** (1 / years) - 1) if years > 0 and total > -1 else np.nan
    ann_vol = float(ret.std() * np.sqrt(MINUTES_PER_YEAR))
    downside = ret[ret < 0].std() * np.sqrt(MINUTES_PER_YEAR)
    dd = equity / equity.cummax() - 1
    trades = int((result["position"].diff().fillna(result["position"]) != 0).sum())
    gross = ret + result["cost"]
    return {
        "total_return": total,
        "annual_return": ann_ret,
        "annual_vol": ann_vol,
        "sharpe": float(ret.mean() / (ret.std() + 1e-12) * np.sqrt(MINUTES_PER_YEAR)),
        "sortino": float(ret.mean() * MINUTES_PER_YEAR / (downside + 1e-12)),
        "max_drawdown": float(dd.min()),
        "calmar": float(ann_ret / abs(dd.min())) if dd.min() < 0 else np.nan,
        "win_rate": float((ret[result["position"].shift().fillna(0) != 0] > 0).mean()),
        "trade_count": trades,
        "turnover_daily": float(result["turnover"].sum() / (n / 1440)),
        "total_cost": float(result["cost"].sum()),
        "cost_to_gross": float(result["cost"].sum() / (abs(gross.sum()) + 1e-12)),
        "exposure": float((result["position"] != 0).mean()),
    }


def cost_sensitivity(
    bars: pd.DataFrame,
    pred: pd.Series,
    cost: CostConfig,
    trade: TradeConfig,
    multipliers: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0),
) -> pd.DataFrame:
    """成本压力测试：分钟级策略必做，验证 alpha 是否被成本吞掉。"""
    from dataclasses import replace

    rows = []
    for m in multipliers:
        scaled = replace(
            cost,
            taker_fee=cost.taker_fee * m,
            slippage=cost.slippage * m,
            impact_coef=cost.impact_coef * m,
        )
        _, perf = run_backtest(bars, pred, scaled, trade)
        rows.append({"cost_multiplier": m, **perf})
    return pd.DataFrame(rows)
