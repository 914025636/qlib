# 波动收缩后放量突破因子

本文把“震荡幅度逐渐缩紧，随后放量向上突破”的 K 线形态拆成可计算、可训练、可回测的量化因子。当前实现位于 [`../binance_btc_usdt_train_backtest.py`](../binance_btc_usdt_train_backtest.py)，使用 BTC/USDT 1 分钟 OHLCV 数据。

## 1. 形态拆解

该形态包含四个部分：

1. **波动收缩**：短周期波动率低于长周期波动率。
2. **区间上沿**：当前收盘价突破此前震荡区间的最高价。
3. **成交量确认**：突破柱成交量显著高于此前正常水平。
4. **收盘确认**：突破柱收盘靠近本根 K 线高点，而不是冲高回落。

只使用第 $t$ 根及其之前的数据生成信号，并在第 $t+1$ 根 K 线成交，避免前视偏差。

## 2. 因子定义

### 2.1 前置震荡区间

设当前 K 线为 $t$，使用此前 20 根 K 线计算区间上沿和下沿：

$$
H_t=\max(High_{t-20},\ldots,High_{t-1})
$$

$$
L_t=\min(Low_{t-20},\ldots,Low_{t-1})
$$

归一化区间宽度为：

$$
RW_t=\frac{H_t-L_t}{Close_{t-1}+\epsilon}
$$

计算区间时必须排除当前突破柱，否则突破柱会参与计算自己的压力位。

### 2.2 波动压缩

对数收益率为：

$$
r_t=\ln\left(\frac{Close_t}{Close_{t-1}}\right)
$$

短长周期波动率压缩比为：

$$
CR_t=
\frac{\operatorname{Std}(r_{t-20:t-1})}
{\operatorname{Std}(r_{t-60:t-1})+\epsilon}
$$

$CR_t$ 越小，短期波动相对长期波动越紧。当前二值信号使用 $CR_t\leq0.7$。

### 2.3 相对成交量

当前成交量与此前 60 根成交量中位数比较：

$$
VR_t=
\frac{Volume_t}
{\operatorname{Median}(Volume_{t-60:t-1})+\epsilon}
$$

基准窗口同样排除当前柱，防止放量尖峰抬高自己的比较基准。当前二值信号使用 $VR_t\geq1.5$。

### 2.4 突破强度

突破幅度相对于此前震荡区间宽度进行归一化：

$$
BS_t=\frac{Close_t-H_t}{H_t-L_t+\epsilon}
$$

$BS_t>0$ 表示收盘价已经站上此前区间上沿。

### 2.5 收盘确认

突破柱的收盘位置为：

$$
CS_t=\frac{Close_t-Low_t}{High_t-Low_t+\epsilon}
$$

$CS_t$ 越接近 1，收盘越靠近本柱高点。当前二值信号使用 $CS_t\geq0.6$。

## 3. 连续因子与事件信号

### 3.1 连续评分

连续评分用于 LightGBM 等模型，让模型自行学习不同强度组合与未来收益的关系：

$$
F_t=
\operatorname{clip}(1-CR_t,0,1)
\cdot\tanh(\max(\ln VR_t,0))
\cdot\tanh(\max(BS_t,0))
\cdot\operatorname{clip}(CS_t,0,1)
$$

评分范围为 $[0,1]$。使用 $\tanh$ 限制极端放量和极窄区间造成的异常值。

### 3.2 二值事件

当前事件信号定义为：

$$
Signal_t=\mathbf{1}\left[
\begin{aligned}
&CR_t\leq0.7\\
&VR_t\geq1.5\\
&BS_t>0\\
&CS_t\geq0.6\\
&Close_{t-1}\leq H_{t-1},\quad Close_t>H_t
\end{aligned}
\right]
$$

最后一个条件只保留首次收盘上穿，避免同一轮突破在后续 K 线上重复触发。

## 4. 当前特征列

[`../binance_btc_usdt_train_backtest.py`](../binance_btc_usdt_train_backtest.py) 已加入以下特征：

| 特征 | 含义 | 用途 |
| --- | --- | --- |
| `range_width_20` | 前 20 根 K 线归一化区间宽度 | 描述震荡空间 |
| `volatility_compression_20_60` | 20/60 根收益波动率比 | 衡量波动收缩 |
| `volume_ratio_60_prev` | 当前量/此前 60 根成交量中位数 | 衡量放量程度 |
| `breakout_strength_20` | 收盘突破前高的归一化幅度 | 衡量突破强度 |
| `squeeze_breakout_score` | 有界连续组合评分 | 作为模型特征 |
| `squeeze_breakout_signal` | 首次上穿的二值事件 | 事件研究或规则过滤 |

核心 Pandas 实现如下：

```python
prior_high_20 = data["high"].rolling(20).max().shift(1)
prior_low_20 = data["low"].rolling(20).min().shift(1)
prior_close = close.shift(1)
prior_channel_width = prior_high_20 - prior_low_20

prior_short_vol = one_bar_return.rolling(20).std().shift(1)
prior_long_vol = one_bar_return.rolling(60).std().shift(1)
compression_ratio = prior_short_vol / (prior_long_vol + 1e-12)

prior_volume_median = data["volume"].rolling(60).median().shift(1)
volume_ratio_prev = data["volume"] / (prior_volume_median + 1e-12)

breakout_strength = (close - prior_high_20) / (prior_channel_width + 1e-12)
first_breakout = (close > prior_high_20) & (prior_close <= prior_high_20.shift(1))
```

## 5. Qlib 表达式参考

如果在 Qlib Handler 中直接构造表达式，可以使用以下组件：

```python
fields = [
    "(Ref(Max($high,20),1)-Ref(Min($low,20),1))/(Ref($close,1)+1e-12)",
    "Ref(Std($close/Ref($close,1)-1,20),1)/(Ref(Std($close/Ref($close,1)-1,60),1)+1e-12)",
    "$volume/(Ref(Med($volume,60),1)+1e-12)",
    "($close-Ref(Max($high,20),1))/(Ref(Max($high,20),1)-Ref(Min($low,20),1)+1e-12)",
    "($close-$low)/($high-$low+1e-12)",
]
```

Qlib 中 `Ref(X, 1)` 表示上一根数据；`Ref(X, -1)` 表示未来数据，不能用于交易因子。Qlib Rolling 默认 `min_periods=1`，而当前 Pandas 实现要求完整窗口，改写时需要注意预热期行为差异。

## 6. 标签与成交时序

当前脚本使用第 $t$ 根 K 线生成特征，在第 $t+1$ 根收盘价入场，并在持有 $h$ 根后退出：

$$
Label_t=\frac{Close_{t+h+1}}{Close_{t+1}}-1
$$

这使标签和回测成交时点保持一致。不能用当前柱收盘价同时生成信号并假设以同一个收盘价成交。

## 7. BTC/USDT 1 分钟数据实证

数据范围为 2024-01-01 至 2026-08-30，共约 140 万根 1 分钟 K 线。使用上述最终事件定义：

- 有效特征行数：1,398,622
- 首次突破事件数：3,535
- 信号发生率：约 0.253%
- 未来 15 分钟平均收益：约 -0.605 bp
- 未来 15 分钟胜率：约 46.31%
- 未来 60 分钟平均收益：约 +0.139 bp
- 未来 60 分钟胜率：约 48.64%

分时段结果如下。收益单位为 bp，$1\text{ bp}=0.01\%$：

| 时段 | 事件数 | 15 分钟均值 | 60 分钟均值 | 240 分钟均值 |
| --- | ---: | ---: | ---: | ---: |
| 训练集 2024-2025 | 2,506 | -0.648 | +0.335 | +2.217 |
| 验证集 2026-01 至 2026-03 | 357 | +0.051 | +0.778 | -5.353 |
| 测试集 2026-04 至 2026-08 | 约 670 | -0.789 | -0.934 | -2.118 |

当前配置每边手续费为 0.04%，每边滑点为 0.01%，完整开平仓成本约为 10 bp。因子的分钟级原始收益远低于交易成本，且训练、验证、测试时段方向不稳定。因此：

> 该形态已经被量化，但当前固定阈值尚未证明具有可交易的样本外优势。

它更适合作为模型输入或状态过滤器，不应直接被视为无条件买入信号。

## 8. 推荐实验流程

1. 在训练集生成连续因子和事件信号。
2. 分别统计 15、60、240 分钟未来收益，而不是先固定持有周期。
3. 只在训练集搜索窗口和阈值，例如 20/60、0.6/0.7、1.5/2.0。
4. 在验证集选择配置并冻结参数。
5. 最后只在测试集评估一次，避免反复调参污染样本外结果。
6. 收益必须扣除手续费、滑点和资金费率。
7. 对相邻事件设置冷却期，或者使用不重叠事件，避免显著性被重复样本夸大。
8. 除平均收益和胜率外，同时检查中位数、尾部亏损、最大回撤、换手率和不同市场状态下的稳定性。

### 当前持仓与退出规则

默认回测逐分钟评估信号和风险。模型预测未来 15 分钟收益，开仓后 15 分钟到期；只有新的预测仍高于交易成本与安全边际时才续持。清仓条件包括连续两次预测不大于 0、连续两根 K 线收回入场突破位下方、2 倍 60 分钟 ATR 初始止损，以及 3 倍 60 分钟 ATR 移动止损。

默认安全边际为 2 bp，因此开仓阈值约为 12 bp。当前模型的预测幅度不足以达到该阈值，默认结果为零笔交易；这表示模型没有识别出足以覆盖成本的机会，而不是回测失败。

### 60 分钟对齐持仓实验

若希望持仓更久，应把预测标签和持仓授权周期一起改为 60 分钟，并继续用 1 分钟数据逐分钟检查风险，而不是用 15 分钟预测直接延长持仓：

```powershell
Set-Location D:\Git\AIQuant\qlib
& D:\Git\AIQuant\.venv\Scripts\python.exe `
  examples\binance_btc_usdt_train_backtest.py `
  --horizon-bars 60 `
  --min-edge 0 `
  --provider-uri .qlib\qlib_data\binance_btc_usdt_1m `
  --output-dir .qlib\experiments\binance_btc_usdt_1m_h60_be_min_edge0_20260830
```

该版本保留 15 分钟基线，使用精确双边成本作为开仓阈值，不额外增加安全边际。训练和验证末端会按未来标签结束时间做 purge；报告的 `dataset.segments` 记录候选数、清除数、保留数以及最后信号和标签结束时间。测试集只使用标签完整的 60 分钟样本。

本次实际 h60 模型实验仍未产生交易：LightGBM 最佳迭代为 `1`，测试集方向准确率约 `50.11%`，预测均值约 `0.511 bp`，低于约 `10.005 bp` 的完整成本。对应模型报告位于 `.qlib/experiments/binance_btc_usdt_1m_h60_be_min_edge0_20260830/report.json`。

另有一条仅针对二值突破事件的固定持有对照回测，使用相同的 1 分钟数据和 60 分钟持有期，共执行 `500` 笔交易，总收益约 `-42.60%`，同期买入持有约 `14.70%`，最大回撤约 `-45.995%`。这说明“持仓更久”本身不是改进，应优先改进信号质量和退出逻辑。对照报告位于 `.qlib/experiments/squeeze_breakout_factor_h60/squeeze_breakout_backtest.html`。

## 9. 可继续扩展的因子

### 区间宽度相对压缩

$$
RC_t=\frac{H_t^{20}-L_t^{20}}{H_t^{60}-L_t^{60}+\epsilon}
$$

这比只看收益波动率更接近“价格箱体正在变窄”的视觉定义。

### 收缩持续性

统计最近若干根 K 线中满足 $CR_t<0.7$ 的比例，排除只在单个时点偶然收缩的情况。

### 突破回踩确认

首次突破后不立即追价，等待价格回踩原区间上沿且未跌破，再生成第二阶段信号。

### 趋势与市场状态过滤

加入高周期均线方向、趋势斜率、资金费率、波动率状态或大盘趋势，检验该形态是否只在特定市场状态下有效。

### 风险调整标签

可将普通未来收益改为事前波动率调整收益：

$$
Y_t=\frac{Return_{t+1:t+h}}{\sigma_{t,60}\sqrt{h}+\epsilon}
$$

这样模型学习的是单位事前风险对应的收益，而不是单纯偏好高波动行情。

## 10. 运行示例

### 因子事件回测可视化

[`plot_backtest.py`](plot_backtest.py) 专门检验二值形态事件：信号在第 $t$ 根 K 线收盘后确认，第 $t+1$ 根收盘买入，默认持有 15 分钟后卖出；持仓期间出现的其他信号会被忽略。每边手续费为 0.04%，每边滑点为 0.01%。

```powershell
Set-Location D:\Git\AIQuant\qlib
& D:\Git\AIQuant\.venv\Scripts\python.exe `
  "examples\波动收缩放量突破因子\plot_backtest.py"
```

也可以修改测试区间、持有期或成本：

```powershell
& D:\Git\AIQuant\.venv\Scripts\python.exe `
  "examples\波动收缩放量突破因子\plot_backtest.py" `
  --start "2026-04-01 00:00:00" `
  --end "2026-08-30 06:37:00" `
  --horizon-bars 60 `
  --fee-rate 0.0004 `
  --slippage-rate 0.0001
```

默认输出到 `.qlib\experiments\squeeze_breakout_factor`：

| 文件 | 内容 |
| --- | --- |
| `squeeze_breakout_backtest.html` | 交互式回测报告 |
| `squeeze_breakout_trades.csv` | 逐笔交易及因子值 |
| `squeeze_breakout_equity.csv` | 分钟级策略、回撤和买入持有曲线 |
| `squeeze_breakout_metrics.json` | 汇总指标与回测参数 |

HTML 报告包含策略与买入持有净值、回撤、价格与买卖点、逐笔收益分布、评分和收益散点、月度复合收益、评分分位收益、代表交易 K 线及最近 200 笔交易表。

当前 2026-04-01 至 2026-08-30 测试集、15 分钟持有期的结果为：

- 候选信号 672 个，去除持仓重叠后执行 572 笔；
- 计入成本后总收益 -46.73%，最大回撤 -47.33%；
- 不计成本的逐笔复合收益约 -5.62%；
- 买入持有收益 +14.70%，交易胜率 18.01%；
- 成本造成的终值差约 41.11 个百分点。

结果说明该事件在当前参数和分钟级成本下不可直接交易。可视化报告的用途是定位问题来自原始事件收益、过高换手、成本侵蚀还是特定评分区间，而不是把形态图解包装成有效策略。

### LightGBM 模型回测

训练和回测：

```powershell
Set-Location D:\Git\AIQuant\qlib
& D:\Git\AIQuant\.venv\Scripts\python.exe examples\binance_btc_usdt_train_backtest.py `
  --provider-uri .qlib\qlib_data\binance_btc_usdt_1m `
  --output-dir .qlib\experiments\binance_btc_usdt_1m
```

生成模型交易前后价格报告：

```powershell
& D:\Git\AIQuant\.venv\Scripts\python.exe examples\plot_binance_trades.py `
  --provider-uri .qlib\qlib_data\binance_btc_usdt_1m `
  --trades-path .qlib\experiments\binance_btc_usdt_1m\trades.csv `
  --output-dir .qlib\experiments\binance_btc_usdt_1m `
  --window-bars 30
```

本文仅说明因子研究方法，不构成投资建议。