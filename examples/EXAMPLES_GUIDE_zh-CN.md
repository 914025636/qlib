# Qlib `examples` 案例详解

本文是 `qlib/examples` 的中文导航。这里的案例不是一套必须顺序执行的教程，而是围绕量化研究生命周期拆分的实验：数据准备、特征与数据集、模型训练、信号分析、组合构建、回测、在线服务和强化学习执行。

## 1. 先理解 Qlib 的标准流程

大多数日频选股案例都可以归纳为下面的链路：

```text
Qlib 二进制数据
    -> DataHandler / Dataset
    -> Model 训练
    -> 预测信号（Signal）
    -> Signal 分析（IC、Rank IC 等）
    -> Strategy 生成目标持仓
    -> Executor 模拟成交
    -> Portfolio 分析（收益、IR、回撤）
```

仓库提供两种写法：

1. **配置驱动**：在 YAML 中声明 `model`、`dataset`、`strategy` 和 `backtest`，使用 `qrun workflow_config_xxx.yaml` 执行。适合复现实验、批量比较和保存配置。
2. **代码驱动**：在 Python 中逐步组装对象。入口是 [workflow_by_code.py](workflow_by_code.py)，适合调试、插入自定义逻辑和理解模块之间的关系。

建议第一次运行从 `workflow_by_code.py` 开始，然后再运行一个 `benchmarks` 下的 LightGBM 配置。

## 2. 环境和数据准备

### 最小要求

`workflow_by_code.py` 的原始说明要求约 16 GB 内存和 5 GB 可用磁盘。深度学习基准、滚动训练和强化学习通常需要更多内存、磁盘和运行时间。

Qlib 初始化时必须指向已经准备好的数据目录，例如：

```python
qlib.init(provider_uri="C:/path/to/qlib_data/cn_data", region=REG_CN)
```

仓库中的部分脚本仍包含 Linux 路径、`wget`、`sed`、Conda `bin` 路径或 MongoDB 本地连接假设。Windows 下应优先使用 PowerShell、手动修改 YAML 路径，并检查脚本是否把路径写死；不要直接照抄所有 Bash 命令。

### 结果如何解读

- **IC / Rank IC**：预测分数与未来收益的横截面相关性，主要看信号是否有排序能力。
- **ICIR / Rank ICIR**：IC 的均值除以波动，反映信号稳定性。
- **Annualized Return**：回测年化收益，必须结合成本和回撤看。
- **Information Ratio**：相对基准的风险调整表现。
- **Max Drawdown**：最大回撤，负值越小越危险。

示例中的历史结果受数据版本、成本、随机种子、操作系统和 Qlib 版本影响。它们用于理解流程和相对比较，不代表未来收益，也不应直接当作实盘策略。

## 3. 入门和通用工作流

### `workflow_by_code.py`

这是最适合读源码的完整日频例子。它会准备数据、初始化 Qlib，创建 CSI300 的 GBDT 模型和数据集，训练模型，生成预测信号，运行信号分析，并使用 `TopkDropoutStrategy` 做组合回测。代码还展示了如何通过 `R.start`、`SignalRecord`、`SigAnaRecord` 和 `PortAnaRecord` 保存实验记录。

运行前先确认脚本中的 `provider_uri` 指向本机数据：

```powershell
python workflow_by_code.py
```

适合回答：数据集如何独立 `prepare`、模型如何训练、预测如何进入回测、实验结果保存在哪里。

### `plot_quick_result.py`

用于把已有实验记录生成快速报告或图表。它不是训练入口，因此应先完成一个工作流并产生 recorder，再运行：

```powershell
python plot_quick_result.py
```

### `workflow_by_code.ipynb`

Notebook 版工作流，适合逐单元查看数据、模型和回测结果。它与 Python 脚本解决相同问题，但执行状态保存在 Kernel 中，重复运行单元时要注意变量和实验记录残留。

## 4. 模型基准与动态市场

### `benchmarks/`

这是最大的案例集合。每个模型目录通常包含：

- `requirements.txt`：模型额外依赖；
- `README.md`：论文或模型说明；
- `workflow_config_<model>_<dataset>.yaml`：可由 `qrun` 读取的完整任务。

模型大致分为三组：

- **树模型和线性基线**：`Linear`、`LightGBM`、`XGBoost`、`CatBoost`、`DoubleEnsemble`，适合先建立可靠基线；
- **序列模型**：`LSTM`、`GRU`、`ALSTM`、`TCN`、`Transformer`、`Localformer`、`TFT`、`SFM`，适合研究时间依赖；
- **关系、注意力和非平稳建模**：`GATs`、`TRA`、`HIST`、`IGMTF`、`TCTS`、`AdaRNN`、`ADD`、`KRNN`、`Sandwich`、`TabNet`、`MLP`、`GeneralPtNN`，适合模型研究和论文复现。

单个模型的典型流程：

```powershell
cd benchmarks/LightGBM
pip install -r requirements.txt
qrun workflow_config_lightgbm_Alpha158.yaml
```

其中：

- **Alpha158** 是人工设计特征较多的表格数据集；
- **Alpha360** 更接近原始价格和成交量序列，时间维度关系更明显；
- `run_all_model.py` 可以批量运行模型、收集 recorder 并生成均值和标准差表，但原脚本含有 Linux/Conda 假设，Windows 下建议先单模型验证。

### `benchmarks_dynamic/`

用于研究市场分布变化导致的模型退化，以及滚动训练、动态适应等方法。`DDG-DA` 和 `baseline` 是该目录的主要实验入口。它比普通基准更关注时间窗口和非平稳性，运行前必须确认对应数据版本、标签周期和滚动区间。

## 5. 数据模块和数据集生命周期

### `data_demo/`

展示数据相关模块的常见用法：

- `data_cache_demo.py`：数据缓存，减少重复读取和计算；
- `data_mem_resuse_demo.py`：复用内存中的数据，适合观察大数据处理时的内存占用。

这两个脚本适合在修改 DataHandler、Processor 或数据读取逻辑后做快速回归，不需要先训练复杂模型。

### `highfreq/`

包含高频数据集和高频价格趋势预测两个方向。`workflow.py` 提供 `get_data`、`dump_and_load_dataset` 等命令：

```powershell
python workflow.py get_data
python workflow.py dump_and_load_dataset
```

重点不是某个模型，而是展示 `DatasetH` 的序列化、从磁盘恢复以及重新初始化 `instruments`、时间范围和 segments。高频预测部分则演示用模型信号判断多空方向，并计算多空精度与收益。

### `rolling_process_data/`

展示滚动训练时如何处理会随窗口变化的可学习状态，例如均值和标准差。流程使用 DataHandler-based DataLoader 读取原始特征，再在不同滚动窗口中重新执行 Processor，避免每个窗口都重新生成全部原始数据：

```powershell
python workflow.py rolling_process
```

这是理解 rolling dataset、数据泄漏和处理器状态隔离的关键案例。

### `model_rolling/`

展示模型本身的滚动训练和任务管理。`task_manager_rolling.py` 负责按时间窗口安排任务，适合研究定期重新训练、预测和结果汇总。运行前先阅读 `requirements.txt`，并确认本机任务调度和输出目录配置。

## 6. 回测、组合和交易执行

### `portfolio/`

展示从“预测收益”进一步走向“风险约束组合”的方法。`prepare_riskdata.py` 准备风险数据，`config_enhanced_indexing.yaml` 使用 `EnhancedIndexingStrategy`，在追求收益的同时控制相对基准的跟踪误差：

```powershell
python prepare_riskdata.py
qrun config_enhanced_indexing.yaml
```

示例需要 CSI300 权重和风险模型数据。文档默认使用统计风险模型；实际研究中应根据数据质量和业务要求评估基本面风险模型或深度风险模型。

### `nested_decision_execution/`

展示嵌套决策执行：较低频的策略先生成组合，较高频的策略再负责把目标持仓拆成订单。入口包括：

```powershell
python workflow.py collect_data
python workflow.py backtest
python workflow.py backtest_highfreq
```

它分别覆盖周频组合加日频执行，以及日频组合加分钟级执行。这个案例适合研究“选什么”和“怎么成交”分离后的回测结构。

### `orderbook_data/`

用于没有固定共享频率的数据，例如订单可能在任意时间到达。案例使用 Arctic/MongoDB 后端导入和查询订单簿数据，再通过 `example.py` 计算高频特征。大致步骤是安装 MongoDB 和依赖、下载示例压缩包、执行 `create_dataset.py initialize_library` 与 `import_data`，最后运行：

```powershell
pytest -s --disable-warnings example.py
```

当前限制是暂不支持不同频率之间的表达式计算；脚本默认连接本机 MongoDB，端口和认证配置需要按环境调整。

### `online_srv/`

包含在线管理和预测更新的模拟案例：

- `online_management_simulate.py`：模拟在线组合管理；
- `rolling_online_management.py`：模拟滚动在线管理；
- `update_online_pred.py`：更新在线预测结果。

这类案例关注模型训练完成后的服务化流程、预测刷新和在线状态，而不是单次离线回测。运行前应先检查脚本中的实验记录、模型路径和时间配置。

## 7. 超参数、解释性和其他研究工具

### `hyperparameter/LightGBM/`

提供 Alpha158 和 Alpha360 的 LightGBM 超参数搜索示例。它适合在固定数据集、固定标签和固定回测设置下比较参数，而不是替代完整的模型基准。搜索时要控制试验数量，并使用独立验证区间，避免把测试集当成调参集。

### `model_interpreter/`

`feature.py` 展示模型特征解释的入口，用于分析哪些输入特征对预测更重要。解释结果应结合训练区间、特征定义和数据处理方式理解，不能把特征重要性直接当作因果关系。

### `rl/`

包含强化学习的最小 Notebook 示例 `simple_example.ipynb`，适合了解 Qlib RL 组件的基本结构。它和监督学习的“特征 -> 标签 -> 预测”不同，核心是环境、状态、动作、奖励和策略更新。

### `rl_order_execution/`

这是完整的强化学习订单执行案例，覆盖数据生成、PPO/OPDS 训练和回测。建议按以下顺序执行：

```powershell
python scripts/gen_pickle_data.py -c scripts/pickle_data_config.yml
python scripts/gen_training_orders.py
python scripts/merge_orders.py
python -m qlib.rl.contrib.train_onpolicy --config_path exp_configs/train_opds.yml --run_backtest
python -m qlib.rl.contrib.backtest --config_path exp_configs/backtest_opds.yml
```

训练阶段的简化模拟器与回测阶段的真实约束模拟器不同，因此两者结果不必完全一致。回测时必须填写已训练 checkpoint，否则随机初始化模型的结果没有解释价值；`TWAP` 配置可作为规则基线。

## 8. 推荐学习路线

1. 先运行 `workflow_by_code.py`，理解数据集、模型、信号、实验记录和回测的连接方式。
2. 再运行 `benchmarks/LightGBM`，熟悉 YAML、`qrun`、Alpha158 和结果指标。
3. 用 `data_demo`、`highfreq` 和 `rolling_process_data` 理解数据缓存、序列化、高频和滚动窗口。
4. 用 `portfolio` 和 `nested_decision_execution` 学习风险约束、组合生成与订单执行分层。
5. 最后进入 `benchmarks_dynamic`、`model_rolling`、`online_srv` 和 `rl_order_execution`，研究非平稳市场、在线更新和强化学习。

## 9. 常见问题排查

### 找不到数据

确认 `qlib.init(provider_uri=...)`、环境变量和配置文件中的数据目录一致。相对路径应以当前工作目录为准，不一定以脚本所在目录为准。

### 找不到额外依赖

先进入对应案例目录安装其 `requirements.txt`。不同模型可能依赖不同版本的 PyTorch、LightGBM、XGBoost 或其他库，建议为实验建立独立虚拟环境。

### 结果和 README 不一致

检查 Qlib 数据版本、市场区域、基准、交易成本、时间区间、随机种子和 Python/依赖版本。基准表通常是多次运行的均值和标准差，单次运行不应期待完全相同。

### Windows 运行失败

优先把路径改成 `pathlib.Path` 或正确的 Windows 路径；将 `wget`、`sed`、`cp`、`unzip` 替换为 PowerShell 等价操作；对 `run_all_model.py`、部分高频和 MongoDB 案例尤其要检查平台假设。

### 如何增加自己的模型

参考 `benchmarks` 下任一模型目录，提供依赖文件、模型说明和可被 `qrun` 读取的 YAML；将模型实现放入合适的 `qlib.contrib.model` 模块，再用固定数据、固定回测设置和多个随机种子进行比较。

## 10. 文件速查表

| 目标 | 推荐入口 |
|---|---|
| 第一次理解完整流程 | `workflow_by_code.py` |
| 用 YAML 跑一个基线 | `benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml` |
| 批量比较模型 | `run_all_model.py` |
| 查看快速报告 | `plot_quick_result.py` |
| 学习数据缓存 | `data_demo/` |
| 学习高频数据集 | `highfreq/` |
| 学习滚动数据处理 | `rolling_process_data/` |
| 学习滚动训练 | `model_rolling/` |
| 学习风险约束组合 | `portfolio/` |
| 学习多频率订单执行 | `nested_decision_execution/` |
| 学习非固定频率订单簿 | `orderbook_data/` |
| 学习在线预测更新 | `online_srv/` |
| 学习超参数搜索 | `hyperparameter/LightGBM/` |
| 学习模型解释 | `model_interpreter/` |
| 学习强化学习执行 | `rl_order_execution/` |
| 学习动态市场适应 | `benchmarks_dynamic/` |