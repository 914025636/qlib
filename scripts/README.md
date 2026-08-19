
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
