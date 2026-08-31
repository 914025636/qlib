
- [Download Qlib Data](#Download-Qlib-Data)
  - [Download CN Data](#Download-CN-Data)
  - [Download US Data](#Download-US-Data)
  - [Download CN Simple Data](#Download-CN-Simple-Data)
  - [Help](#Help)
- [Using in Qlib](#Using-in-Qlib)
  - [US data](#US-data)
  - [CN data](#CN-data)


## Download Qlib Data


### Download CN Data

```bash
# daily data
python get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn

# 1min  data (Optional for running non-high-frequency strategies)
python get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data_1min --region cn --interval 1min
```

### Download US Data


```bash
python get_data.py qlib_data --target_dir ~/.qlib/qlib_data/us_data --region us
```

### Download CN Simple Data

```bash
python get_data.py qlib_data --name qlib_data_simple --target_dir ~/.qlib/qlib_data/cn_data --region cn
```

### Help

```bash
python get_data.py qlib_data --help
```

## Using in Qlib
> For more information: https://qlib.readthedocs.io/en/latest/start/initialization.html


### US data

> Need to download data first: [Download US Data](#Download-US-Data)

```python
import qlib
from qlib.config import REG_US
provider_uri = "~/.qlib/qlib_data/us_data"  # target_dir
qlib.init(provider_uri=provider_uri, region=REG_US)
```

### CN data

> Need to download data first: [Download CN Data](#Download-CN-Data)

```python
import qlib
from qlib.constant import REG_CN

provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
qlib.init(provider_uri=provider_uri, region=REG_CN)
```

## Use Crowd Sourced Data
The is also a [crowd sourced version of qlib data](data_collector/crowd_source/README.md): https://github.com/chenditc/investment_data/releases
```bash
wget https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz
tar -zxvf qlib_bin.tar.gz -C ~/.qlib/qlib_data/cn_data --strip-components=2
```

## QuestDB 高频数据

安装 QuestDB 的可选客户端依赖：

```bash
pip install "pyqlib[questdb]"
```

`questdb_to_qlib.py` 支持从 QuestDB 查询 `trades` 或 orderbook 增量数据，并按
`1s`、`5s` 等 pandas 频率聚合为每个 symbol 一个 CSV 文件。字段名通过参数配置，
因此不要求 QuestDB 使用固定 schema。

trades 示例：

```bash
python scripts/questdb_to_qlib.py \
  --table btc_trades --date 2024-01-02 --freq 1s --data-type trades \
  --output-dir ./questdb_csv --timestamp-column ts --symbol-column instrument \
  --price-column trade_price --size-column trade_size
```

orderbook 增量示例：

```bash
python scripts/questdb_to_qlib.py \
  --table btc_orderbook --date 2024-01-02 --freq 5s --data-type orderbook \
  --output-dir ./questdb_csv --timestamp-column ts --symbol-column instrument \
  --side-column side --price-column price --size-column quantity \
  --book-size-mode absolute --top-n 5
```

默认 orderbook 数量是该价位的绝对数量；若数据记录的是数量变化量，使用
`--book-size-mode delta`。追加 `--dump-qlib ./qlib_data` 可以在生成 CSV 后直接调用
Qlib 的 dump 逻辑生成 `calendars`、`features` 和 `instruments` 目录。

## Binance BTC/USDT 1 分钟数据训练与回测

已转换的 BTC/USDT 现货 1 分钟数据位于：
`.qlib/qlib_data/binance_btc_usdt_1m`。可以使用示例脚本执行一个
LightGBM 未来 15 分钟收益预测、信号生成和单标的现货回测：

```bash
python examples/binance_btc_usdt_train_backtest.py \
  --provider-uri .qlib/qlib_data/binance_btc_usdt_1m \
  --output-dir .qlib/experiments/binance_btc_usdt_1m
```

默认时间切分为：

- 训练集：2024-01-01 至 2025-12-31
- 验证集：2026-01-01 至 2026-03-31
- 测试集/回测：2026-04-01 至数据末尾

策略为只做多或空仓，预测在下一根 K 线成交，并逐分钟评估信号和风险。默认按每边
0.04% 手续费和 0.01% 滑点计算；只有预测收益超过精确双边成本并额外留出 0.02%
安全边际时才开仓。持仓默认 15 分钟到期，只有新的强预测才能续持，并使用以下退出规则：

- 连续两次预测不大于 0
- 连续两根 K 线收回入场突破位下方
- 2 倍 60 分钟 ATR 初始止损
- 3 倍 60 分钟 ATR 移动止损

- `model.joblib`：训练后的 LightGBM 模型
- `trades.csv`：逐笔交易明细
- `report.json`：验证集、测试集和回测指标

这只是基线实验，不构成投资建议。当前模型的测试集预测幅度不足以覆盖约 0.10%
完整交易成本与安全边际，因此默认成本过滤会产生零笔交易。这比强制持仓更准确地反映
模型尚未证明具有可交易的样本外优势。

### 60 分钟对齐持仓实验

如需让持仓周期更长，应同步延长预测标签窗口，而不是用 15 分钟预测直接授权长时间持仓。
下面的命令使用未来 60 分钟收益作为标签和首次持仓授权周期，仍然每分钟检查信号、止损和
突破失败；`--min-edge 0` 表示开仓门槛只覆盖精确的双边手续费与滑点成本：

```bash
python examples/binance_btc_usdt_train_backtest.py \
  --horizon-bars 60 \
  --min-edge 0 \
  --provider-uri .qlib/qlib_data/binance_btc_usdt_1m \
  --output-dir .qlib/experiments/binance_btc_usdt_1m_h60_be_min_edge0_20260830
```

该实验不会覆盖 15 分钟基线。报告中的 `dataset.segments` 会记录因未来标签跨越下一段而
清除的样本；60 分钟标签在训练集和验证集末端分别清除 61 个信号位置。测试集只评估能够
完整计算未来 60 分钟标签的样本。

生成对应交易价格报告：

```bash
python examples/plot_binance_trades.py \
  --provider-uri .qlib/qlib_data/binance_btc_usdt_1m \
  --trades-path .qlib/experiments/binance_btc_usdt_1m_h60_be_min_edge0_20260830/trades.csv \
  --output-dir .qlib/experiments/binance_btc_usdt_1m_h60_be_min_edge0_20260830
```

生成 15/60 分钟模型和因子实验的对比总览：

```bash
python examples/compare_binance_experiments.py \
  --output-dir .qlib/experiments/binance_experiment_comparison
```

输出包含净值与回撤曲线、收益/交易数/多头暴露对比、模型预测幅度与成本门槛，
以及四组实验的明细表：

- `binance_experiment_comparison.html`：交互式对比页面
- `binance_experiment_comparison.json`：页面使用的结构化指标

本次实际 h60 运行结果：训练集 `1,051,139` 条、验证集 `129,539` 条、测试集
`217,777` 条；训练和验证末端各 purge `61` 个跨界标签。LightGBM 最佳迭代为 `1`，
测试集方向准确率约 `50.11%`，预测均值约 `0.511 bp`，仍低于约 `10.005 bp` 的完整
交易成本，因此模型回测为 `0` 笔交易、`0%` 暴露。这说明延长预测窗口没有自动创造
可交易优势，下一步应改进标签或模型，而不是继续降低成本门槛。
