# 趋势跟踪策略：因子与标签设计

本目录实现了一套**趋势跟踪**（Trend Following）策略，用于替代 `examples/workflow_by_code.py` 中"固定买入前 50 只股票"的截面选股方案。

| 文件 | 作用 |
|---|---|
| `handler.py` | 趋势因子集（48 个）+ 3 种可选标签，继承 `DataHandlerLP` |
| `strategy.py` | 阈值入场 / 迟滞出场 / 波动率目标定权的组合策略，继承 `WeightStrategyBase` |
| `workflow_trend.py` | 端到端流程：训练 → 信号分析 → 回测 |

运行：

```powershell
cd f:\Python\AiQuantization\qlib
python examples/trend_following/workflow_trend.py
```

---

## 一、为什么不能直接改策略参数

很多人以为"把 `topk=50` 改小、或者加个止损"就变成趋势跟踪了，这是不成立的。原始方案的三个组件是**互相锁死**的：

```
Alpha158 标签  →  CSZScoreNorm(截面标准化)  →  预测值只有相对排名含义
                                            ↓
                              只能"选最强的 K 只"，无法回答"该不该持仓"
                                            ↓
                                   TopkDropoutStrategy 永远满仓
```

`qlib/contrib/data/handler.py` 中的默认配置是：

```python
_DEFAULT_LEARN_PROCESSORS = [
    {"class": "DropnaLabel"},
    {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},  # ← 问题所在
]
```

`CSZScoreNorm` 按 `datetime` 分组做 z-score，**每一天的标签均值被强制归零**。这意味着：

- 大盘暴跌 20% 的那天，跌 10% 的股票标签仍然是正数（因为它跑赢了当天均值）；
- 模型学到的是"谁比谁强"，而不是"能不能赚钱"；
- 预测值 0.8 在牛市和熊市含义完全相同 → **任何基于阈值的择时都失效**。

所以趋势跟踪必须从标签层重构，策略层的改动只是结果而非原因。

### 三层对照

| 维度 | 截面选股（原方案） | 趋势跟踪（本方案） |
|---|---|---|
| 标签 | 1 日收益 + 截面标准化 | 多日风险调整收益，**保留绝对量纲** |
| 特征归一化 | `CSZScoreNorm`（抹掉市场整体水平） | `RobustZScoreNorm`（保留市场水平，才能择时） |
| 持仓 | 固定 topk 等权，永远满仓 | 阈值筛选 + 波动率定权，**允许空仓** |
| 收益来源 | 选股 alpha | 选股 alpha + 择时 beta |
| 换手率 | 由 `n_drop` 固定 | 由迟滞区间自适应，趋势延续时接近 0 |

---

## 二、标签设计

代码位置：`handler.py` 的 `get_trend_label_config()`。

所有标签都假设 **t 日收盘决策、t+1 日收盘建仓**，因此分母统一是 `Ref($close,-1)` 而不是 `$close`，避免未来函数。

### 1. `risk_adj_ret`（默认）

$$\text{label}_t=\frac{P_{t+h+1}/P_{t+1}-1}{\sigma_{20}\cdot\sqrt{h}}$$

```python
(Ref($close,-6)/Ref($close,-1)-1)/(Std($close/Ref($close,1)-1,20)*2.236068+1e-4)
```

**为什么除以波动率**：如果直接用原始收益，模型会偏好高波动小盘股——它们的收益方差大，MSE 损失下更"值得预测"。除以事前波动率后，标签变成**风险调整后的收益**，单位是"未来 $h$ 日能赚几个 $\sigma$"。

这个量纲的好处是阈值天然可解释：

- `label > 0` → 预期上涨
- `label > 0.3` → 预期涨幅约 0.3 倍波动，是有意义的趋势
- 高波动股票需要更大的绝对涨幅才能达到同一标签值，风险自动被惩罚

CSI300 上 2017H1 的实测分布（见验证记录）：均值 0.072、标准差 1.261、中位数 0.025。所以 `buy_thresh=0.3` 大致对应前 40% 分位，属于合理量级。

$\sqrt{h}$ 是波动率的时间缩放（随机游走假设下 $h$ 日波动 $=\sigma_1\sqrt{h}$），确保不同持仓期的标签量纲可比。

### 2. `fwd_ret`

```python
Ref($close,-6)/Ref($close,-1)-1
```

原始 $h$ 日前瞻收益。适合已有独立风控模块、不希望标签里混入波动率信息的场景。缺点是模型会系统性偏向高波动股。

### 3. `trend_quality`

$$\text{label}=\underbrace{\frac{\max(H_{t+1..t+h})}{P_{t+1}}-1}_{\text{MFE 最大有利偏移}}-\underbrace{\left(1-\frac{\min(L_{t+1..t+h})}{P_{t+1}}\right)}_{\text{MAE 最大不利偏移}}$$

```python
(Ref(Max($high,5),-5)/Ref($close,-1)-1)-(1-Ref(Min($low,5),-5)/Ref($close,-1))
```

衡量趋势是否**平滑**。两只股票同样 5 日涨 10%，一路小阳线的那只 MAE 接近 0、标签高；先跌 8% 再涨 18% 的那只 MAE 很大、标签低。

这对趋势跟踪特别重要——后者在实盘中会触发止损，或让持有者在回撤中提前离场，纸面收益吃不到。**如果你打算加止损，用这个标签。**

> `Ref(Max($high,5),-5)` 的含义：先算 5 日滚动最高价，再整体前移 5 天，得到 $[t+1, t+5]$ 区间的最高价。

### 标签处理器：关键改动

```python
_TREND_LEARN_PROCESSORS = [
    {"class": "DropnaLabel"},
    # 刻意不加 CSZScoreNorm
]
```

这一行的缺席是整套方案成立的前提。

---

## 三、因子设计

代码位置：`handler.py` 的 `get_trend_feature_config()`，共 48 个因子分 9 组。

设计原则：**只回答三个问题** —— 趋势是否存在？是否健康？是否被确认？

不包含短期反转类因子（如 Alpha158 里的 `RSV`、部分 `KBAR`），因为它们与趋势跟踪的逻辑直接冲突。

### 1. 多周期动量（5 个）— 趋势是否存在

```python
$close/Ref($close,w)-1   # w ∈ {5,10,20,60,120}
```

最朴素的趋势度量。多个周期并列，让模型自己学"短期动量与长期动量方向一致时才可靠"。

### 2. 趋势斜率与一致性（9 个）— 趋势是否健康 ★核心

```python
Slope($close,w)/$close   # 斜率，除以价格做无量纲化
Rsquare($close,w)        # 线性拟合优度
Resi($close,w)/$close    # 回归残差
```

**`Rsquare` 是本方案最重要的因子。** 同样是 20 日涨 10%：

- $R^2=0.9$ → 价格贴着一条直线稳步上行，是真趋势
- $R^2=0.2$ → 剧烈震荡后碰巧收在高位，是噪声

单看 `ROC20` 无法区分二者，`Slope` × `Rsquare` 才是完整的趋势描述：**方向 × 可信度**。

> `Rsquare` 在 `qlib/data/ops.py` 中对近似常数序列（`std ≈ 0`）返回 `NaN`，由 `Fillna` 兜底。

### 3. 均线排列（7 个）

```python
$close/Mean($close,w)-1        # 价格对各周期均线的乖离
Mean($close,5)/Mean($close,20)-1
Mean($close,20)/Mean($close,60)-1
```

后两个是**均线多头排列**的量化表达。经典趋势跟踪的"金叉"是这两个值由负转正的时刻，这里保留连续值让模型判断强度。

### 4. 通道位置与突破（9 个）

```python
($close-Min($low,w))/(Max($high,w)-Min($low,w)+1e-12)  # 唐奇安通道内的相对位置，∈[0,1]
$close/Max($high,w)-1                                   # 距 w 日新高的距离，≥0 即创新高
IdxMax($close,w)/w                                      # 最高点出现在窗口的什么位置
```

`IDXMAX` 值接近 1 表示**最高价就在最近几天**（趋势仍在推进）；接近 0 表示高点在很久以前（趋势可能已衰竭）。这是纯截面因子给不出的时序信息。

### 5. 波动与回撤（6 个）

```python
Std($close/Ref($close,1)-1,w)   # 历史波动率
Mean($high-$low,w)/$close       # ATR 的简化形式
$close/Max($close,w)-1          # 从 w 日高点算起的当前回撤
```

双重用途：既是模型特征，`VOL20` 的计算式也被 `strategy.py` 复用于仓位定权（见下文）。

### 6. 上涨日占比（2 个）

```python
Mean($close>Ref($close,1),w)
```

区分"连续 60 天小涨 0.3%"和"59 天横盘 + 1 天涨停"。前者是可持续趋势，后者是事件驱动。

> qlib 的比较算子返回布尔值，`Mean` 会自动按 0/1 求均值。写法与 `qlib/contrib/data/loader.py:231` 一致。

### 7. 量能确认（4 个）

```python
$volume/(Mean($volume,w)+1e-12)        # 量比
Corr($close,Log($volume+1),w)          # 价量相关性
```

**放量上涨**是趋势被资金确认的标志；缩量上涨往往难以持续。`Log` 用于压缩成交量的长尾分布。

### 8. 市场状态（4 个）— 择时的关键

```python
ChangeInstrument('SH000300',$close/Ref($close,w)-1)
ChangeInstrument('SH000300',$close/Mean($close,120)-1)
```

`ChangeInstrument` 算子（`qlib/data/ops.py:64`）让个股的特征行里也能取到**指数**的数据。

这组因子是"整体该不该上仓位"的信息源。熊市中所有 `MKTROC` 为负，模型学到的映射是"此时多数股票的未来风险调整收益为负"，预测值整体下移 → 通过 `buy_thresh` 过滤后合格标的自然减少 → 仓位自动下降。**择时不是硬编码规则，而是标签量纲 + 市场因子共同涌现的结果。**

> 注意：这类因子在截面上对同一天的所有股票取值相同。在 `CSZScoreNorm` 下会被完全抹成 0（这正是原方案无法择时的另一原因），在 `RobustZScoreNorm` 下则被保留。

### 9. 相对强度（2 个）

```python
($close/Ref($close,w)-1)-ChangeInstrument('SH000300',$close/Ref($close,w)-1)
```

剔除市场 beta 后的个股趋势，用于在"市场该上仓位"的前提下挑选**领涨股**。

### 特征处理器：第二个关键改动

```python
_TREND_INFER_PROCESSORS = [
    {"class": "ProcessInf"},
    {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
    {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
]
```

用**时序**稳健标准化（基于训练期的中位数与 MAD）而非截面标准化。差别在于：

- `CSZScoreNorm`：每天减去当天截面均值 → 市场整体涨跌被抹除
- `RobustZScoreNorm`：减去训练期固定的中位数 → **"今天全市场都在涨"这个信息被保留**

`clip_outlier=True` 将结果截断到 $[-3,3]$，防止极端行情主导训练。统计量只在 `fit_start_time ~ fit_end_time`（训练段）上估计，不泄露测试期信息。

---

## 四、策略设计

代码位置：`strategy.py`，继承 `WeightStrategyBase`（输出目标权重字典）而非 `TopkDropoutStrategy`（输出固定股票数）。

### 阈值入场 + 迟滞出场

```python
keep  = score[(score >= sell_thresh) & score.index.isin(held)]    # 已持仓：宽松
entry = score[(score >= buy_thresh)  & ~score.index.isin(held)]   # 新建仓：严格
```

`sell_thresh=0.0 < buy_thresh=0.3` 构成**迟滞区间**（hysteresis）。落在 $[0, 0.3)$ 的股票：

- 已持有 → 继续持有
- 未持有 → 不买入

这是趋势跟踪"让利润奔跑"的机制，也是降低换手的主要手段。相比之下 `TopkDropoutStrategy` 的 `n_drop=5` 意味着**每个调仓日强制换掉 5 只**，无论趋势是否延续。

验证结果：

```
score = {A:0.9, B:0.5, C:0.15, D:-0.2, E:0.35, F:0.31}, max_n=4

held=[C,D] -> {C, A, B, E}   # C(0.15) 在迟滞区间被保留；D(-0.2) 跌破 sell_thresh 清仓
held=[]    -> {A, B, E, F}   # 空仓起步时 C 不满足 buy_thresh，不买
无合格信号  -> {}             # 直接空仓
```

注意 `held=[C,D]` 时结果包含 `C` 而不含 `F`(0.31)——**持仓优先占名额**，避免因新信号略强而无谓换手。

### 波动率目标定权

```python
weights = (target_vol / vol.clip(lower=1e-4)).clip(upper=max_weight)
```

$$w_i=\min\left(\frac{\sigma_{\text{target}}}{\sigma_i},\ w_{\max}\right)$$

每只股票贡献大致相同的风险而非相同的资金。低波动股票拿更多仓位，高波动股票自动缩量。`max_weight=0.05` 防止极低波动标的占据过大比例。

`_get_vol()` 在首次调用时一次性载入整段回测区间的波动率面板并缓存，避免逐日查询 IO。

### 允许空仓

```python
total = weights.sum()
if total > 1.0:
    weights = weights / total
return weights[weights > 1e-6].to_dict()
```

**只在权重和超过 1 时才归一化。** 若合格信号只有 10 只、每只权重 0.04，则总仓位 40%，剩余 60% 自动留作现金。

这与 `TopkDropoutStrategy` 的行为形成根本差异——后者的 `value = cash * risk_degree / len(buy)` 永远把资金分光。本策略中 `risk_degree` 退化为**资金上限**，实际仓位由信号数量决定。

另有 `min_n` 兜底：合格标的少于 5 只时判定为无趋势行情，返回 `{}` 全部清仓。

---

## 五、参数调节

`workflow_trend.py` 中的可调项：

| 参数 | 默认 | 作用 | 调节方向 |
|---|---|---|---|
| `buy_thresh` | 0.3 | 入场门槛 | ↑ 更保守、空仓更多、换手更低 |
| `sell_thresh` | 0.0 | 出场门槛 | ↓ 持有更久，迟滞区间更宽 |
| `max_n` | 50 | 最大持仓数 | ↑ 更分散 |
| `min_n` | 5 | 空仓触发线 | ↑ 更容易空仓 |
| `target_vol` | 0.02 | 单股风险预算 | ↑ 整体仓位提高（等效加杠杆） |
| `max_weight` | 0.05 | 单股权重上限 | ↓ 强制分散 |
| `HOLDING_DAYS` | 5 | 标签前瞻窗口 | 需与实际持仓周期匹配 |

`equal_weight=True` 可退化为等权，用于对照波动率定权是否真的有效。

---

## 六、两个必须注意的坑

### 1. 持仓期必须与标签前瞻窗口一致

`HOLDING_DAYS=5` 同时决定标签的 `Ref($close,-6)` 和策略的实际持有时长。只改一头会导致**"模型学的"和"策略做的"不是同一件事**：标签预测 5 日收益，但策略每日换手 → 实际吃到的是 1 日收益，IC 再高也无法转化为盈利。

如果调整 `HOLDING_DAYS`，应同步放宽 `sell_thresh` 让平均持仓期匹配。

### 2. IC 会显著低于 Alpha158 基线，这是正常的

`SigAnaRecord` 输出的 IC 会明显下降。原因：IC 是**截面**相关系数，而本方案的标签刻意保留了时序上的绝对量纲，两者的评价口径不匹配。

**不要用 IC 判断本策略的好坏**，应关注：

- 回测的年化收益 / 最大回撤 / Calmar 比率
- **换手率**（迟滞机制是否真的生效）
- **仓位随时间的变化曲线**——这是趋势跟踪的核心检验：2018 年熊市仓位是否显著下降？

第三项最关键。可从 `PortAnaRecord` 保存的 `positions_normal.pkl` 中提取每日持仓市值占比来绘制。如果仓位常年贴近 100%，说明标签或市场因子没起作用，需回头检查 `learn_processors` 是否误加了 `CSZScoreNorm`。

---

## 七、验证记录

| 项目 | 结果 |
|---|---|
| 48 个因子 + 3 种标签的表达式解析 | 全部通过 |
| CSI300 / 2017H1 真实数据计算 | 35700 × 49，无异常 |
| `risk_adj_ret` 标签分布 | 均值 0.072，标准差 1.261，中位数 0.025 |
| 策略迟滞 / 名额分配 / 空仓逻辑 | 三种情形均符合预期 |
