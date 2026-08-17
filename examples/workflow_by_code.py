#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.
"""
Qlib provides two kinds of interfaces.
(1) Users could define the Quant research workflow by a simple configuration.
(2) Qlib is designed in a modularized way and supports creating research workflow by code just like building blocks.

The interface of (1) is `qrun XXX.yaml`.  The interface of (2) is script like this, which nearly does the same thing as `qrun XXX.yaml`

中文说明：
Qlib 提供两种使用方式：
(1) 配置驱动：写一个 YAML 文件，用命令 `qrun XXX.yaml` 一键跑完整个量化研究流程。
(2) 代码驱动：像本脚本这样，用积木式的模块化 API 逐步搭建流程，效果与 (1) 基本等价，
    但更灵活，便于调试、插入自定义逻辑。

本脚本完整演示了一条最小可用的量化研究流水线：
    数据准备 -> 初始化 -> 构建模型/数据集 -> 训练 -> 生成预测信号 -> 信号分析 -> 组合回测
"""

import qlib

# REG_CN：区域常量，表示 A 股（中国市场）。它会决定交易日历、涨跌停规则、
# 最小交易单位（100 股/手）、交易成本默认值等一系列市场相关配置。
from qlib.constant import REG_CN

# init_instance_by_config：Qlib 的核心工具函数，把 {"class":..., "module_path":..., "kwargs":...}
#                          形式的字典“反射”成真正的 Python 对象实例，是配置驱动的基础。
# flatten_dict：把嵌套字典拍平成单层（如 {"a": {"b": 1}} -> {"a.b": 1}），
#               便于作为实验参数记录到 MLflow（MLflow 的参数不支持嵌套结构）。
from qlib.utils import init_instance_by_config, flatten_dict

# R：Qlib 的全局实验记录器（Recorder / QlibRecorder 单例），底层基于 MLflow。
#    负责实验管理、参数记录、模型与中间结果的持久化。
from qlib.workflow import R

# 三个“记录模板”（Record Template），负责在实验中自动产出并保存分析结果：
#   SignalRecord ：用训练好的模型在测试集上做预测，产出并保存原始预测信号 pred.pkl、label.pkl
#   SigAnaRecord ：对预测信号做质量分析，产出 IC、Rank IC、ICIR、多空组合收益等指标
#   PortAnaRecord：把信号送入策略并执行回测，产出年化收益、信息比率、最大回撤等组合绩效指标
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord

# GetData：数据下载辅助类，可从官方源自动拉取并解压预处理好的 qlib 二进制格式行情数据
from qlib.tests.data import GetData

# CSI300_BENCH   ：沪深300 指数代码（"SH000300"），作为回测基准
# CSI300_GBDT_TASK：官方预置的示例任务配置字典，包含 model（LightGBM 超参）
#                   和 dataset（数据处理器、特征表达式、训练/验证/测试时间段划分）两部分
from qlib.tests.config import CSI300_BENCH, CSI300_GBDT_TASK

# 只有直接运行本文件时才执行下面的流程；被其他模块 import 时不会触发。
# 注意：Qlib 内部使用多进程（joblib/loky）加速表达式计算，
#       在 Windows 上必须有 __main__ 保护，否则会因子进程重复导入而死循环。
if __name__ == "__main__":
    # ------------------------------------------------------------------
    # 第 1 步：准备数据
    # ------------------------------------------------------------------
    # provider_uri：本地数据仓库根目录。Qlib 不直接读 CSV，而是使用自研的
    #               列式二进制格式（.bin），按 features/calendars/instruments 三类目录组织，
    #               以获得极高的读取性能。
    provider_uri = "F:/Python/AiQuantization/qlib/.qlib/qlib_data/cn_data"  # target_dir

    # 下载 A 股日频数据到上述目录。
    # exists_skip=True 表示如果目标目录已存在数据则跳过下载，避免每次运行都重复拉取。
    GetData().qlib_data(target_dir=provider_uri, region=REG_CN, exists_skip=True)

    # 初始化 Qlib 运行环境（全局唯一，必须在使用任何其他 Qlib 功能之前调用）。
    # 它会完成：注册数据提供者、加载交易日历与股票池、设置缓存策略、
    #           按 region 应用 A 股特有的市场规则等。
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    # ------------------------------------------------------------------
    # 第 2 步：根据配置构建模型与数据集对象
    # ------------------------------------------------------------------
    # 通过配置字典实例化 LightGBM 模型（qlib.contrib.model.gbdt.LGBModel），
    # 其中包含 learning_rate、num_leaves、max_depth、subsample 等超参数。
    model = init_instance_by_config(CSI300_GBDT_TASK["model"])

    # 实例化数据集（DatasetH，H 表示 Handler）。内部包含：
    #   - handler：Alpha158 特征工程器，自动计算 158 个技术因子（KMID、ROC、MA、STD…），
    #              并完成缺失值处理、横截面标准化（CSZScoreNorm）、去极值等预处理；
    #   - segments：时间切分，train=2008-01-01~2014-12-31，
    #               valid=2015-01-01~2016-12-31，test=2017-01-01~2020-08-01。
    #     这种严格按时间先后的切分可避免未来函数（look-ahead bias）。
    dataset = init_instance_by_config(CSI300_GBDT_TASK["dataset"])

    # ------------------------------------------------------------------
    # 第 3 步：定义组合回测配置
    # ------------------------------------------------------------------
    # 该字典描述了「怎么执行交易」「用什么策略」「在什么市场环境下回测」三件事，
    # 后续会被 PortAnaRecord 读取并用 init_instance_by_config 逐层实例化。
    port_analysis_config = {
        # ---------- 执行器：负责按时间步推进并撮合订单 ----------
        "executor": {
            "class": "SimulatorExecutor",  # 模拟撮合执行器（非嵌套、非真实下单）
            "module_path": "qlib.backtest.executor",
            "kwargs": {
                # 回测的时间粒度：每步推进一个交易日。
                # 若做日内高频，可改为 "30min"、"5min" 等。
                "time_per_step": "day",
                # 是否逐日生成组合指标（净值、收益、换手率、持仓明细等）。
                # 必须为 True，否则后续无法计算年化收益、最大回撤等绩效指标。
                "generate_portfolio_metrics": True,
            },
        },
        # ---------- 策略：负责把预测信号转换成买卖决策 ----------
        "strategy": {
            # TopkDropoutStrategy：Qlib 最经典的基准策略。
            # 每个调仓日按预测分数从高到低排序，持有前 topk 只股票；
            # 调仓时卖出当前持仓中排名最靠后的 n_drop 只，再买入排名最靠前的等量新股票。
            # 相比每期完全换成 Top-K，这种“部分替换”能显著降低换手率和交易成本。
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy.signal_strategy",
            "kwargs": {
                # 信号来源：直接传入 (模型, 数据集) 二元组，
                # 策略内部会调用 model.predict(dataset) 实时生成打分；
                # 也可以传入一个已算好的 pandas DataFrame/Series 作为信号。
                "signal": (model, dataset),
                "topk": 50,  # 目标持仓股票数量（同时持有 50 只，等权分配资金）
                "n_drop": 5,  # 每次调仓最多换掉 5 只，控制换手率
            },
        },
        # ---------- 回测环境：时间区间、初始资金、基准与交易所规则 ----------
        "backtest": {
            # 回测区间必须落在 dataset 的 test 段内，确保用的是模型从未见过的样本外数据
            "start_time": "2017-01-01",
            "end_time": "2020-08-01",
            "account": 100000000,  # 初始资金 1 亿元。资金量足够大可减小整手取整带来的误差
            "benchmark": CSI300_BENCH,  # 业绩基准：沪深300 指数，用于计算超额收益与信息比率
            "exchange_kwargs": {  # 模拟交易所的撮合规则
                "freq": "day",  # 行情频率，需与 executor 的 time_per_step 保持一致
                # 涨跌停限制：单日涨跌幅超过 9.5% 视为触及涨跌停，该股当日不可成交。
                # 这里取 0.095 而非 0.1，是为对齐真实的四舍五入价位并留出安全边界。
                "limit_threshold": 0.095,
                "deal_price": "close",  # 成交价采用当日收盘价（也可用 "open"/"vwap"）
                "open_cost": 0.0005,  # 买入成本 0.05%（券商佣金等）
                "close_cost": 0.0015,  # 卖出成本 0.15%（佣金 + 印花税 0.1%）
                "min_cost": 5,  # 单笔最低手续费 5 元，符合 A 股券商惯例
            },
        },
    }

    # ------------------------------------------------------------------
    # 第 4 步（可选）：预览训练集数据
    # ------------------------------------------------------------------
    # NOTE: This line is optional
    # It demonstrates that the dataset can be used standalone.
    # 这一段仅用于演示 dataset 可以脱离整个工作流单独使用。
    # prepare("train") 会触发特征计算与预处理，返回一个 MultiIndex(datetime, instrument) 的 DataFrame，
    # 列为二级索引：feature（158 个因子）与 label（未来收益率标签）。
    # 首次运行较慢（需计算全部因子表达式），之后会命中磁盘缓存。
    example_df = dataset.prepare("train")
    print(example_df.head())

    # ------------------------------------------------------------------
    # 第 5 步：在实验记录上下文中完成 训练 -> 预测 -> 分析 -> 回测
    # ------------------------------------------------------------------
    # start exp
    # R.start 会创建（或复用）名为 "workflow" 的 MLflow 实验，并新建一次 run。
    # 使用 with 上下文可确保 run 在结束或异常时被正确关闭，产物写入 ./mlruns 目录。
    # 运行结束后可执行 `mlflow ui` 在浏览器中查看结果。
    with R.start(experiment_name="workflow"):
        # 记录本次实验的全部超参数。flatten_dict 把嵌套配置拍平，
        # 便于在 MLflow 界面中横向对比不同实验的参数差异。
        R.log_params(**flatten_dict(CSI300_GBDT_TASK))

        # 训练模型：内部会取出 dataset 的 train 段拟合，用 valid 段做早停（early stopping）。
        model.fit(dataset)

        # 把训练好的模型序列化保存为 params.pkl 作为实验产物（artifact），便于复现与线上部署。
        R.save_objects(**{"params.pkl": model})

        # prediction
        # 获取当前活动的 recorder 对象，后续所有分析结果都会写入它对应的目录。
        recorder = R.get_recorder()

        # 生成预测信号：在 test 段上执行 model.predict(dataset)，
        # 保存 pred.pkl（预测分数）与 label.pkl（真实标签），供后续分析使用。
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()

        # Signal Analysis
        # 信号质量分析：读取上一步的 pred/label，计算并保存
        # IC（信息系数）、ICIR、Rank IC、Rank ICIR，以及多空组合的累计收益曲线。
        # 经验参考：IC 均值 > 0.03 通常认为该因子/模型具备一定预测能力。
        sar = SigAnaRecord(recorder)
        sar.generate()

        # backtest. If users want to use backtest based on their own prediction,
        # please refer to https://qlib.readthedocs.io/en/latest/component/recorder.html#record-template.
        # 组合回测：按 port_analysis_config 组装策略与执行器并逐日模拟交易，
        # 输出含成本/不含成本两套绩效指标（年化收益、信息比率 IR、最大回撤 MDD、超额收益等）。
        # 第三个参数 "day" 指定分析频率，需与 executor 的 time_per_step 一致。
        # 若想用自己算好的预测结果回测（而非模型实时预测），可参考上面链接中的 Record Template 文档。
        par = PortAnaRecord(recorder, port_analysis_config, "day")
        par.generate()
