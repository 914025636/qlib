"""震荡市均值回归套利的数据处理器（因子 + 标签）。

与趋势跟踪(trend_following)互补：
- 趋势跟踪赚"价格离开均值"的钱，标签是 5 日风险调整收益，偏好高 R^2 的单边行情；
- 本方案赚"价格回到均值"的钱，标签是 2 日短线风险调整收益，
  因子聚焦超卖深度、震荡市识别（低效率比/低 R^2）与恐慌性放量，
  只有在"个股显著偏离 + 市场处于震荡状态"时信号才会走高。
标签同样保留绝对量纲（不做截面标准化），使阈值入场与空仓择时成立。
"""

from math import sqrt

from qlib.contrib.data.handler import check_transform_proc
from qlib.data.dataset.handler import DataHandlerLP

_MR_INFER_PROCESSORS = [
    {"class": "ProcessInf", "kwargs": {}},
    {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
    {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
]

# 不加 CSZScoreNorm：预测值须保留"未来能赚几个 sigma"的绝对含义
_MR_LEARN_PROCESSORS = [
    {"class": "DropnaLabel"},
]


def get_mr_feature_config(benchmark="SH000300"):
    """返回 (fields, names)。共 9 组，围绕：偏离多深 / 是否震荡市 / 是否恐慌尾声。"""
    fields, names = [], []

    # 1) 布林位置（价格偏离均值几个 sigma）：均值回归最核心的入场依据
    for w in [5, 10, 20]:
        fields.append(f"($close-Mean($close,{w}))/(Std($close,{w})+1e-12)")
        names.append(f"BOLL{w}")

    # 2) 短期反转动量：近几日跌幅，负得越深回归空间越大
    for w in [1, 2, 3, 5]:
        fields.append(f"$close/Ref($close,{w})-1")
        names.append(f"ROC{w}")

    # 3) RSI 类强弱：上涨动能占比，低位超卖
    for w in [6, 14]:
        fields.append(
            f"Mean(Greater($close-Ref($close,1),0),{w})/(Mean(Abs($close-Ref($close,1)),{w})+1e-12)"
        )
        names.append(f"RSI{w}")

    # 4) 通道位置：贴近下轨(0)为超卖，贴近上轨(1)为超买
    for w in [10, 20]:
        fields.append(f"($close-Min($low,{w}))/(Max($high,{w})-Min($low,{w})+1e-12)")
        names.append(f"CHANPOS{w}")

    # 5) 震荡市识别：Kaufman 效率比(净位移/路径长度)与趋势 R^2，都低才是震荡市
    for w in [10, 20]:
        fields.append(f"Abs($close/Ref($close,{w})-1)/(Sum(Abs($close/Ref($close,1)-1),{w})+1e-12)")
        names.append(f"ER{w}")
        fields.append(f"Rsquare($close,{w})")
        names.append(f"RSQR{w}")

    # 6) 均值锚点稳定性：20 日均线自身的斜率，均线走平时回归目标才可靠
    fields.append("Slope(Mean($close,20),10)/$close")
    names.append("MASLOPE")
    fields.append("$close/Mean($close,20)-1")
    names.append("MA20BIAS")

    # 7) 恐慌确认：下跌日的放量（capitulation）与当日振幅
    fields.append("If($close<Ref($close,1),$volume/(Mean($volume,20)+1e-12),0)")
    names.append("PANICVOL")
    fields.append("($high-$low)/(Ref($close,1)+1e-12)")
    names.append("AMPLITUDE")
    fields.append("($close-$low)/($high-$low+1e-12)")
    names.append("CLOSEPOS")  # 当日收盘在日内区间的位置，低开高走=1 有企稳迹象

    # 8) 波动率：供策略侧定权，也让模型区分"正常回调"与"波动率爆炸"
    for w in [10, 20]:
        fields.append(f"Std($close/Ref($close,1)-1,{w})")
        names.append(f"VOL{w}")
    fields.append("Std($close/Ref($close,1)-1,5)/(Std($close/Ref($close,1)-1,60)+1e-12)")
    names.append("VOLRATIO")  # 短期波动/长期波动，>1 表示波动正在放大

    # 9) 市场状态：指数是否震荡（低 ER）、是否在崩盘（深负 ROC），决定策略该不该开工
    fields.append(f"ChangeInstrument('{benchmark}',Abs($close/Ref($close,20)-1)/(Sum(Abs($close/Ref($close,1)-1),20)+1e-12))")
    names.append("MKTER20")
    fields.append(f"ChangeInstrument('{benchmark}',Rsquare($close,20))")
    names.append("MKTRSQR20")
    for w in [5, 20]:
        fields.append(f"ChangeInstrument('{benchmark}',$close/Ref($close,{w})-1)")
        names.append(f"MKTROC{w}")
    fields.append(f"ChangeInstrument('{benchmark}',Std($close/Ref($close,1)-1,20))")
    names.append("MKTVOL20")

    return fields, names


def get_mr_label_config(label_type="risk_adj_ret", holding_days=2):
    """短持仓期标签，t 日决策、t+1 日收盘建仓。"""
    h = holding_days
    if label_type == "risk_adj_ret":
        scale = sqrt(h)
        return (
            [f"(Ref($close,-{h + 1})/Ref($close,-1)-1)/(Std($close/Ref($close,1)-1,20)*{scale:.6f}+1e-4)"],
            ["LABEL0"],
        )
    if label_type == "fwd_ret":
        return [f"Ref($close,-{h + 1})/Ref($close,-1)-1"], ["LABEL0"]
    if label_type == "revert_to_mean":
        # 未来 h 日内价格向 20 日均线的收敛程度：当前偏离 - 未来偏离，专注"回归"本身
        cur_dev = "$close/Mean($close,20)-1"
        fut_dev = f"Ref($close,-{h + 1})/Ref(Mean($close,20),-{h + 1})-1"
        return [f"Abs({cur_dev})-Abs({fut_dev})"], ["LABEL0"]
    raise ValueError(f"unknown label_type: {label_type}")


class MeanReversionHandler(DataHandlerLP):
    def __init__(
        self,
        instruments="csi300",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=_MR_INFER_PROCESSORS,
        learn_processors=_MR_LEARN_PROCESSORS,
        fit_start_time=None,
        fit_end_time=None,
        process_type=DataHandlerLP.PTYPE_A,
        filter_pipe=None,
        inst_processors=None,
        benchmark="SH000300",
        label_type="risk_adj_ret",
        holding_days=2,
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
                    "feature": get_mr_feature_config(benchmark),
                    "label": kwargs.pop("label", get_mr_label_config(label_type, holding_days)),
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
