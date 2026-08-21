"""趋势跟踪专用的数据处理器（因子 + 标签）。

与 Alpha158 的关键区别：
1. 标签保留绝对量纲（不做截面标准化），预测值才能用于阈值判断和择时；
2. 特征用时序 RobustZScoreNorm 而非截面标准化，保留"市场整体处于什么水平"的信息；
3. 因子聚焦趋势的存在性、一致性、健康度与确认度，而非短期反转类信号。
"""

from math import sqrt

from qlib.contrib.data.handler import check_transform_proc
from qlib.data.dataset.handler import DataHandlerLP

# 特征：先处理 inf，再按训练期统计做稳健 z-score（保留截面间与时间上的绝对水平），最后填 0
_TREND_INFER_PROCESSORS = [
    {"class": "ProcessInf", "kwargs": {}},
    {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
    {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
]

# 标签只丢缺失，不做任何标准化，否则预测值失去"未来能赚多少"的绝对含义
_TREND_LEARN_PROCESSORS = [
    {"class": "DropnaLabel"},
]


def get_trend_feature_config(benchmark="SH000300"):
    """返回 (fields, names) 形式的趋势因子集合。"""
    fields, names = [], []

    # 1) 多周期动量：趋势的方向与强度
    for w in [5, 10, 20, 60, 120]:
        fields.append(f"$close/Ref($close,{w})-1")
        names.append(f"ROC{w}")

    # 2) 趋势斜率与一致性。RSQR 越高说明价格越贴近一条直线，是趋势跟踪最核心的质量指标
    for w in [10, 20, 60]:
        fields.append(f"Slope($close,{w})/$close")
        names.append(f"SLOPE{w}")
        fields.append(f"Rsquare($close,{w})")
        names.append(f"RSQR{w}")
        fields.append(f"Resi($close,{w})/$close")
        names.append(f"RESI{w}")

    # 3) 均线排列：短中长期均线是否多头发散
    for w in [5, 10, 20, 60, 120]:
        fields.append(f"$close/Mean($close,{w})-1")
        names.append(f"MA{w}BIAS")
    fields += ["Mean($close,5)/Mean($close,20)-1", "Mean($close,20)/Mean($close,60)-1"]
    names += ["MA5_20", "MA20_60"]

    # 4) 通道位置与突破：价格在 N 日区间中的相对位置、距新高的距离
    for w in [20, 60, 120]:
        fields.append(f"($close-Min($low,{w}))/(Max($high,{w})-Min($low,{w})+1e-12)")
        names.append(f"CHANPOS{w}")
        fields.append(f"$close/Max($high,{w})-1")
        names.append(f"HIGHDIST{w}")
        fields.append(f"IdxMax($close,{w})/{w}")
        names.append(f"IDXMAX{w}")

    # 5) 波动与回撤：趋势的"平顺度"，同时供策略侧做波动率目标定权
    for w in [20, 60]:
        fields.append(f"Std($close/Ref($close,1)-1,{w})")
        names.append(f"VOL{w}")
        fields.append(f"Mean($high-$low,{w})/$close")
        names.append(f"ATR{w}")
        fields.append(f"$close/Max($close,{w})-1")
        names.append(f"DD{w}")

    # 6) 上涨日占比：趋势是靠连续小涨还是单日跳空堆出来的
    for w in [20, 60]:
        fields.append(f"Mean($close>Ref($close,1),{w})")
        names.append(f"UPRATIO{w}")

    # 7) 量能确认：放量上涨才是健康趋势
    for w in [20, 60]:
        fields.append(f"$volume/(Mean($volume,{w})+1e-12)")
        names.append(f"VSTD{w}")
        fields.append(f"Corr($close,Log($volume+1),{w})")
        names.append(f"CORR{w}")

    # 8) 市场状态：指数自身的趋势，决定整体该不该上仓位
    for w in [20, 60, 120]:
        fields.append(f"ChangeInstrument('{benchmark}',$close/Ref($close,{w})-1)")
        names.append(f"MKTROC{w}")
    fields.append(f"ChangeInstrument('{benchmark}',$close/Mean($close,120)-1)")
    names.append("MKTMA120BIAS")

    # 9) 相对强度：剔除市场后的个股趋势
    for w in [20, 60]:
        fields.append(f"($close/Ref($close,{w})-1)-ChangeInstrument('{benchmark}',$close/Ref($close,{w})-1)")
        names.append(f"RS{w}")

    return fields, names


def get_trend_label_config(label_type="risk_adj_ret", holding_days=5):
    """趋势跟踪的标签。持仓期 holding_days，均假设 t 日决策、t+1 日收盘建仓。"""
    h = holding_days
    if label_type == "risk_adj_ret":
        # 未来 h 日收益 / 事前波动，量纲为"几个 sigma"，阈值 0 天然可解释
        scale = sqrt(h)
        return (
            [f"(Ref($close,-{h + 1})/Ref($close,-1)-1)/(Std($close/Ref($close,1)-1,20)*{scale:.6f}+1e-4)"],
            ["LABEL0"],
        )
    if label_type == "fwd_ret":
        return [f"Ref($close,-{h + 1})/Ref($close,-1)-1"], ["LABEL0"]
    if label_type == "trend_quality":
        # MFE - MAE：奖励一路向上的平滑趋势，惩罚先大幅回撤再涨的伪趋势
        mfe = f"Ref(Max($high,{h}),-{h})/Ref($close,-1)-1"
        mae = f"1-Ref(Min($low,{h}),-{h})/Ref($close,-1)"
        return [f"({mfe})-({mae})"], ["LABEL0"]
    raise ValueError(f"unknown label_type: {label_type}")


class TrendHandler(DataHandlerLP):
    def __init__(
        self,
        instruments="csi300",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=_TREND_INFER_PROCESSORS,
        learn_processors=_TREND_LEARN_PROCESSORS,
        fit_start_time=None,
        fit_end_time=None,
        process_type=DataHandlerLP.PTYPE_A,
        filter_pipe=None,
        inst_processors=None,
        benchmark="SH000300",
        label_type="risk_adj_ret",
        holding_days=5,
        **kwargs,
    ):
        self.benchmark = benchmark
        self.label_type = label_type
        self.holding_days = holding_days

        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": get_trend_feature_config(benchmark),
                    "label": kwargs.pop("label", get_trend_label_config(label_type, holding_days)),
                },
                "filter_pipe": filter_pipe,
                "freq": freq,
                "inst_processors": inst_processors,
            },
        }
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            process_type=process_type,
            **kwargs,
        )
