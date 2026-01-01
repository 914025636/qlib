[![Python Versions](https://img.shields.io/pypi/pyversions/pyqlib.svg?logo=python&logoColor=white)](https://pypi.org/project/pyqlib/#files)
[![Platform](https://img.shields.io/badge/platform-linux%20%7C%20windows%20%7C%20macos-lightgrey)](https://pypi.org/project/pyqlib/#files)
[![PypI Versions](https://img.shields.io/pypi/v/pyqlib)](https://pypi.org/project/pyqlib/#history)
[![Upload Python Package](https://github.com/microsoft/qlib/workflows/Upload%20Python%20Package/badge.svg)](https://pypi.org/project/pyqlib/)
[![Github Actions Test Status](https://github.com/microsoft/qlib/workflows/Test/badge.svg?branch=main)](https://github.com/microsoft/qlib/actions)
[![Documentation Status](https://readthedocs.org/projects/qlib/badge/?version=latest)](https://qlib.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/pypi/l/pyqlib)](LICENSE)
[![Join the chat at https://gitter.im/Microsoft/qlib](https://badges.gitter.im/Microsoft/qlib.svg)](https://gitter.im/Microsoft/qlib?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

## :newspaper: **最新动态!** &nbsp;   :sparkling_heart: 

最近发布的功能

### 介绍 <a href="https://github.com/microsoft/RD-Agent"><img src="docs/_static/img/rdagent_logo.png" alt="RD_Agent" style="height: 2em"></a>: 基于大语言模型的工业级数据驱动研发自主进化智能体

我们很高兴地宣布发布 **RD-Agent**📢，这是一个强大的工具，支持量化投资研发中的自动化因子挖掘和模型优化。

RD-Agent 现已在 [GitHub](https://github.com/microsoft/RD-Agent) 上可用，欢迎您的 star🌟！

要了解更多信息，请访问我们的 [♾️演示页面](https://rdagent.azurewebsites.net/)。在这里，您可以找到中英文的演示视频，帮助您更好地理解 RD-Agent 的场景和用法。

我们为您准备了几个演示视频：
| 场景 | 演示视频 (英文) | 演示视频 (中文) |
| --                      | ------    | ------    |
| 量化因子挖掘 | [链接](https://rdagent.azurewebsites.net/factor_loop?lang=en) | [链接](https://rdagent.azurewebsites.net/factor_loop?lang=zh) |
| 基于报告的量化因子挖掘 | [链接](https://rdagent.azurewebsites.net/report_factor?lang=en) | [链接](https://rdagent.azurewebsites.net/report_factor?lang=zh) |
| 量化模型优化 | [链接](https://rdagent.azurewebsites.net/model_loop?lang=en) | [链接](https://rdagent.azurewebsites.net/model_loop?lang=zh) |

- 📃**论文**: [R&D-Agent-Quant: A Multi-Agent Framework for Data-Centric Factors and Model Joint Optimization](https://arxiv.org/abs/2505.15155)
- 👾**代码**: https://github.com/microsoft/RD-Agent/
```BibTeX
@misc{li2025rdagentquant,
    title={R\&D-Agent-Quant: A Multi-Agent Framework for Data-Centric Factors and Model Joint Optimization},
    author={Yuante Li and Xu Yang and Xiao Yang and Minrui Xu and Xisen Wang and Weiqing Liu and Jiang Bian},
    year={2025},
    eprint={2505.15155},
    archivePrefix={arXiv},
    primaryClass={cs.AI}
}
```
![image](https://github.com/user-attachments/assets/3198bc10-47ba-4ee0-8a8e-46d5ce44f45d)

***

| 功能 | 状态 |
| --                      | ------    |
| [R&D-Agent-Quant](https://arxiv.org/abs/2505.15155) 发表 | 将 R&D-Agent 应用于 Qlib 进行量化交易 | 
| 端到端学习的 BPQP | 📈即将推出!([审核中](https://github.com/microsoft/qlib/pull/1863)) |
| 🔥LLM 驱动的自动量化工厂🔥 | 🚀 于 2024 年 8 月 8 日在 [♾️RD-Agent](https://github.com/microsoft/RD-Agent) 中发布 |
| KRNN 和 Sandwich 模型 | :chart_with_upwards_trend: [已发布](https://github.com/microsoft/qlib/pull/1414/) 于 2023 年 5 月 26 日 |
| 发布 Qlib v0.9.0 | :octocat: [已发布](https://github.com/microsoft/qlib/releases/tag/v0.9.0) 于 2022 年 12 月 9 日 |
| RL 学习框架 | :hammer: :chart_with_upwards_trend: 发布于 2022 年 11 月 10 日。 [#1332](https://github.com/microsoft/qlib/pull/1332), [#1322](https://github.com/microsoft/qlib/pull/1322), [#1316](https://github.com/microsoft/qlib/pull/1316),[#1299](https://github.com/microsoft/qlib/pull/1299),[#1263](https://github.com/microsoft/qlib/pull/1263), [#1244](https://github.com/microsoft/qlib/pull/1244), [#1169](https://github.com/microsoft/qlib/pull/1169), [#1125](https://github.com/microsoft/qlib/pull/1125), [#1076](https://github.com/microsoft/qlib/pull/1076)|
| HIST 和 IGMTF 模型 | :chart_with_upwards_trend: [已发布](https://github.com/microsoft/qlib/pull/1040) 于 2022 年 4 月 10 日 |
| Qlib [notebook 教程](https://github.com/microsoft/qlib/tree/main/examples/tutorial) | 📖 [已发布](https://github.com/microsoft/qlib/pull/1037) 于 2022 年 4 月 7 日 | 
| Ibovespa 指数数据 | :rice: [已发布](https://github.com/microsoft/qlib/pull/990) 于 2022 年 4 月 6 日 |
| Point-in-Time 数据库 | :hammer: [已发布](https://github.com/microsoft/qlib/pull/343) 于 2022 年 3 月 10 日 |
| Arctic 提供商后端和订单簿数据示例 | :hammer: [已发布](https://github.com/microsoft/qlib/pull/744) 于 2022 年 1 月 17 日 |
| 基于元学习的框架和 DDG-DA  | :chart_with_upwards_trend:  :hammer: [已发布](https://github.com/microsoft/qlib/pull/743) 于 2022 年 1 月 10 日 | 
| 基于规划的投资组合优化 | :hammer: [已发布](https://github.com/microsoft/qlib/pull/754) 于 2021 年 12 月 28 日 | 
| 发布 Qlib v0.8.0 | :octocat: [已发布](https://github.com/microsoft/qlib/releases/tag/v0.8.0) 于 2021 年 12 月 8 日 |
| ADD 模型 | :chart_with_upwards_trend: [已发布](https://github.com/microsoft/qlib/pull/704) 于 2021 年 11 月 22 日 |
| ADARNN 模型 | :chart_with_upwards_trend: [已发布](https://github.com/microsoft/qlib/pull/689) 于 2021 年 11 月 14 日 |
| TCN 模型 | :chart_with_upwards_trend: [已发布](https://github.com/microsoft/qlib/pull/668) 于 2021 年 11 月 4 日 |
| 嵌套决策框架 | :hammer: [已发布](https://github.com/microsoft/qlib/pull/438) 于 2021 年 10 月 1 日。 [示例](https://github.com/microsoft/qlib/blob/main/examples/nested_decision_execution/workflow.py) 和 [文档](https://qlib.readthedocs.io/en/latest/component/highfreq.html) |
| 时间路由适配器 (TRA) | :chart_with_upwards_trend: [已发布](https://github.com/microsoft/qlib/pull/531) 于 2021 年 7 月 30 日 |
| Transformer 和 Localformer | :chart_with_upwards_trend: [已发布](https://github.com/microsoft/qlib/pull/508) 于 2021 年 7 月 22 日 |
| 发布 Qlib v0.7.0 | :octocat: [已发布](https://github.com/microsoft/qlib/releases/tag/v0.7.0) 于 2021 年 7 月 12 日 |
| TCTS 模型 | :chart_with_upwards_trend: [已发布](https://github.com/microsoft/qlib/pull/491) 于 2021 年 7 月 1 日 |
| 在线服务和自动模型滚动 | :hammer:  [已发布](https://github.com/microsoft/qlib/pull/290) 于 2021 年 5 月 17 日 | 
| DoubleEnsemble 模型 | :chart_with_upwards_trend: [已发布](https://github.com/microsoft/qlib/pull/286) 于 2021 年 3 月 2 日 | 
| 高频数据处理示例 | :hammer: [已发布](https://github.com/microsoft/qlib/pull/257) 于 2021 年 2 月 5 日  |
| 高频交易示例 | :chart_with_upwards_trend: [部分代码已发布](https://github.com/microsoft/qlib/pull/227) 于 2021 年 1 月 28 日  | 
| 高频数据(1min) | :rice: [已发布](https://github.com/microsoft/qlib/pull/221) 于 2021 年 1 月 27 日 |
| Tabnet 模型 | :chart_with_upwards_trend: [已发布](https://github.com/microsoft/qlib/pull/205) 于 2021 年 1 月 22 日 |

2021 年之前发布的功能未在此列出。

<p align="center">
  <img src="docs/_static/img/logo/1.png" />
</p>

Qlib 是一个开源的、面向 AI 的量化投资平台，旨在利用 AI 技术在量化投资中实现潜力、赋能研究并创造价值，从探索想法到实现生产。Qlib 支持多种机器学习建模范式，包括监督学习、市场动态建模和强化学习。

越来越多的不同范式的 SOTA 量化研究工作/论文在 Qlib 中发布，以协同解决量化投资中的关键挑战。例如，1) 使用监督学习从丰富和异构的金融数据中挖掘市场复杂的非线性模式，2) 使用自适应概念漂移技术对金融市场的动态性质进行建模，以及 3) 使用强化学习对连续投资决策进行建模，并协助投资者优化其交易策略。

它包含数据处理、模型训练、回测的完整 ML 管道；并涵盖了量化投资的整个链条：alpha 寻找、风险建模、投资组合优化和订单执行。
有关更多详细信息，请参阅我们的论文 ["Qlib: An AI-oriented Quantitative Investment Platform"](https://arxiv.org/abs/2009.11189)。


<table>
  <tbody>
    <tr>
      <th>框架、教程、数据和 DevOps</th>
      <th>量化研究中的主要挑战和解决方案</th>
    </tr>
    <tr>
      <td>
        <li><a href="#plans"><strong>计划</strong></a></li>
        <li><a href="#framework-of-qlib">Qlib 框架</a></li>
        <li><a href="#quick-start">快速开始</a></li>
          <ul dir="auto">
            <li type="circle"><a href="#installation">安装</a> </li>
            <li type="circle"><a href="#data-preparation">数据准备</a></li>
            <li type="circle"><a href="#auto-quant-research-workflow">自动量化研究工作流</a></li>
            <li type="circle"><a href="#building-customized-quant-research-workflow-by-code">通过代码构建自定义量化研究工作流</a></li></ul>
        <li><a href="#quant-dataset-zoo"><strong>量化数据集 Zoo</strong></a></li>
        <li><a href="#learning-framework">学习框架</a></li>
        <li><a href="#more-about-qlib">更多关于 Qlib</a></li>
        <li><a href="#offline-mode-and-online-mode">离线模式和在线模式</a>
        <ul>
          <li type="circle"><a href="#performance-of-qlib-data-server">Qlib 数据服务器的性能</a></li></ul>
        <li><a href="#related-reports">相关报告</a></li>
        <li><a href="#contact-us">联系我们</a></li>
        <li><a href="#contributing">贡献</a></li>
      </td>
      <td valign="baseline">
        <li><a href="#main-challenges--solutions-in-quant-research">量化研究中的主要挑战和解决方案</a>
          <ul>
            <li type="circle"><a href="#forecasting-finding-valuable-signalspatterns">预测：寻找有价值的信号/模式</a>
              <ul>
                <li type="disc"><a href="#quant-model-paper-zoo"><strong>量化模型 (论文) Zoo</strong></a>
                  <ul>
                    <li type="circle"><a href="#run-a-single-model">运行单个模型</a></li>
                    <li type="circle"><a href="#run-multiple-models">运行多个模型</a></li>
                  </ul>
                </li>
              </ul>
            </li>
          <li type="circle"><a href="#adapting-to-market-dynamics">适应市场动态</a></li>
          <li type="circle"><a href="#reinforcement-learning-modeling-continuous-decisions">强化学习：建模连续决策</a></li>
          </ul>
        </li>
      </td>
    </tr>
  </tbody>
</table>

# 计划
正在开发的新功能（按预计发布时间排序）。
您对功能的反馈非常重要。
<!-- | Feature                        | Status      | -->
<!-- | --                      | ------    | -->

# Qlib 框架

<div style="align: center">
<img src="docs/_static/img/framework-abstract.jpg" />
</div>

Qlib 的高级框架如上所示（用户可以在深入了解细节时找到 Qlib 设计的 [详细框架](https://qlib.readthedocs.io/en/latest/introduction/introduction.html#framework)）。
组件被设计为松散耦合的模块，每个组件都可以独立使用。

Qlib 提供了强大的基础设施来支持量化研究。[数据](https://qlib.readthedocs.io/en/latest/component/data.html) 始终是重要的一部分。
设计了一个强大的学习框架来支持不同的学习范式（例如 [强化学习](https://qlib.readthedocs.io/en/latest/component/rl.html)，[监督学习](https://qlib.readthedocs.io/en/latest/component/workflow.html#model-section)）和不同层级的模式（例如 [市场动态建模](https://qlib.readthedocs.io/en/latest/component/meta.html)）。
通过对市场建模，[交易策略](https://qlib.readthedocs.io/en/latest/component/strategy.html) 将生成将被执行的交易决策。不同层级或粒度的多个交易策略和执行器可以 [嵌套在一起进行优化和运行](https://qlib.readthedocs.io/en/latest/component/highfreq.html)。
最后，将提供全面的 [分析](https://qlib.readthedocs.io/en/latest/component/report.html)，并且模型可以以低成本 [在线服务](https://qlib.readthedocs.io/en/latest/component/online.html)。


# 快速开始

本快速入门指南旨在演示
1. 使用 _Qlib_ 构建完整的量化研究工作流并尝试您的想法非常容易。
2. 即使使用 *公开数据* 和 *简单模型*，机器学习技术在实际量化投资中也 **非常有效**。

这是一个快速 **[演示](https://terminalizer.com/view/3f24561a4470)**，展示了如何安装 ``Qlib``，并使用 ``qrun`` 运行 LightGBM。**但是**，请确保您已按照 [说明](#data-preparation) 准备好数据。


## 安装

此表演示了 `Qlib` 支持的 Python 版本：
|               | install with pip      | install from source  |        plot        |
| ------------- |:---------------------:|:--------------------:|:------------------:|
| Python 3.8    | :heavy_check_mark:    | :heavy_check_mark:   | :heavy_check_mark: |
| Python 3.9    | :heavy_check_mark:    | :heavy_check_mark:   | :heavy_check_mark: |
| Python 3.10   | :heavy_check_mark:    | :heavy_check_mark:   | :heavy_check_mark: |
| Python 3.11   | :heavy_check_mark:    | :heavy_check_mark:   | :heavy_check_mark: |
| Python 3.12   | :heavy_check_mark:    | :heavy_check_mark:   | :heavy_check_mark: |

**注意**: 
1. 建议使用 **Conda** 管理您的 Python 环境。在某些情况下，在 `conda` 环境之外使用 Python 可能会导致缺少头文件，从而导致某些包安装失败。
2. 请注意，在 Python 3.6 中安装 cython 会在从源代码安装 ``Qlib`` 时引发一些错误。如果用户在其机器上使用 Python 3.6，建议将 Python *升级* 到 3.8 或更高版本，或使用 `conda` 的 Python 从源代码安装 ``Qlib``。

### 使用 pip 安装
用户可以根据以下命令通过 pip 轻松安装 ``Qlib``。

```bash
  pip install pyqlib
```

**注意**: pip 将安装最新的稳定版 qlib。但是，qlib 的主分支正在积极开发中。如果您想测试主分支中的最新脚本或功能。请使用以下方法安装 qlib。

### 从源码安装
此外，用户可以根据以下步骤通过源代码安装最新的开发版本 ``Qlib``：

* 在从源码安装 ``Qlib`` 之前，用户需要安装一些依赖项：

  ```bash
  pip install numpy
  pip install --upgrade cython
  ```

* 克隆仓库并按如下方式安装 ``Qlib``。
    ```bash
    git clone https://github.com/microsoft/qlib.git && cd qlib
    pip install .  # 推荐使用 `pip install -e .[dev]` 进行开发。查看 docs/developer/code_standard_and_dev_guide.rst 中的详细信息
    ```

**提示**: 如果您无法在您的环境中安装 `Qlib` 或运行示例，将您的步骤与 [CI 工作流](.github/workflows/test_qlib_from_source.yml) 进行比较可能会帮助您找到问题。

**Mac 提示**: 如果您使用的是配备 M1 的 Mac，您可能会在构建 LightGBM 的 wheel 时遇到问题，这是由于缺少 OpenMP 的依赖项。要解决此问题，请先使用 ``brew install libomp`` 安装 openmp，然后运行 ``pip install .`` 以成功构建它。

## 数据准备
❗ 由于更严格的数据安全政策。官方数据集暂时禁用。您可以尝试社区贡献的 [此数据源](https://github.com/chenditc/investment_data/releases)。
这是一个下载最新数据的示例。
```bash
wget https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz
mkdir -p ~/.qlib/qlib_data/cn_data
tar -zxvf qlib_bin.tar.gz -C ~/.qlib/qlib_data/cn_data --strip-components=1
rm -f qlib_bin.tar.gz
```

下面的官方数据集将在不久的将来恢复。


----

通过运行以下代码加载和准备数据：

### 使用模块获取
  ```bash
  # 获取 1d 数据
  python -m qlib.cli.data qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn

  # 获取 1min 数据
  python -m qlib.cli.data qlib_data --target_dir ~/.qlib/qlib_data/cn_data_1min --region cn --interval 1min

  ```

### 从源获取

  ```bash
  # 获取 1d 数据
  python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn

  # 获取 1min 数据
  python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data_1min --region cn --interval 1min

  ```

该数据集由 [爬虫脚本](scripts/data_collector/) 收集的公开数据创建，这些脚本已在同一仓库中发布。
用户可以使用它创建相同的数据集。[数据集描述](https://github.com/microsoft/qlib/tree/main/scripts/data_collector#description-of-dataset)

*请 **注意**，数据是从 [Yahoo Finance](https://finance.yahoo.com/lookup) 收集的，数据可能并不完美。
如果用户拥有高质量的数据集，我们建议用户准备自己的数据。有关更多信息，用户可以参考 [相关文档](https://qlib.readthedocs.io/en/latest/component/data.html#converting-csv-format-into-qlib-format)*。

### 每日频率数据的自动更新（来自 yahoo finance）
  > 如果用户只想在历史数据上尝试他们的模型和策略，则此步骤是 *可选的*。
  > 
  > 建议用户手动更新一次数据 (--trading_date 2021-05-25)，然后将其设置为自动更新。
  >
  > **注意**: 用户不能基于 Qlib 提供的离线数据增量更新数据（为了减小数据大小，删除了一些字段）。用户应使用 [yahoo collector](https://github.com/microsoft/qlib/tree/main/scripts/data_collector/yahoo#automatic-update-of-daily-frequency-datafrom-yahoo-finance) 从头开始下载 Yahoo 数据，然后增量更新它。
  > 
  > 有关更多信息，请参考：[yahoo collector](https://github.com/microsoft/qlib/tree/main/scripts/data_collector/yahoo#automatic-update-of-daily-frequency-datafrom-yahoo-finance)

  * 每个交易日自动更新数据到 "qlib" 目录 (Linux)
      * 使用 *crontab*: `crontab -e`
      * 设置定时任务：

        ```
        * * * * 1-5 python <script path> update_data_to_bin --qlib_data_1d_dir <user data dir>
        ```
        * **script path**: *scripts/data_collector/yahoo/collector.py*

  * 手动更新数据
      ```
      python scripts/data_collector/yahoo/collector.py update_data_to_bin --qlib_data_1d_dir <user data dir> --trading_date <start date> --end_date <end date>
      ```
      * *trading_date*: 交易日开始
      * *end_date*: 交易日结束（不包含）

### 检查数据的健康状况
  * 我们提供了一个脚本来检查数据的健康状况，您可以运行以下命令来检查数据是否健康。
    ```
    python scripts/check_data_health.py check_data --qlib_dir ~/.qlib/qlib_data/cn_data
    ```
  * 当然，您也可以添加一些参数来调整测试结果，例如这样。
    ```
    python scripts/check_data_health.py check_data --qlib_dir ~/.qlib/qlib_data/cn_data --missing_data_num 30055 --large_step_threshold_volume 94485 --large_step_threshold_price 20
    ```
  * 如果您想了解有关 `check_data_health` 的更多信息，请参考 [文档](https://qlib.readthedocs.io/en/latest/component/data.html#checking-the-health-of-the-data)。

<!-- 
- Run the initialization code and get stock data:

  ```python
  import qlib
  from qlib.data import D
  from qlib.constant import REG_CN

  # Initialization
  mount_path = "~/.qlib/qlib_data/cn_data"  # target_dir
  qlib.init(mount_path=mount_path, region=REG_CN)

  # Get stock data by Qlib
  # Load trading calendar with the given time range and frequency
  print(D.calendar(start_time='2010-01-01', end_time='2017-12-31', freq='day')[:2])

  # Parse a given market name into a stockpool config
  instruments = D.instruments('csi500')
  print(D.list_instruments(instruments=instruments, start_time='2010-01-01', end_time='2017-12-31', as_list=True)[:6])

  # Load features of certain instruments in given time range
  instruments = ['SH600000']
  fields = ['$close', '$volume', 'Ref($close, 1)', 'Mean($close, 3)', '$high-$low']
  print(D.features(instruments, fields, start_time='2010-01-01', end_time='2017-12-31', freq='day').head())
  ```
 -->

## Docker 镜像
1. 从 docker hub 仓库拉取 docker 镜像
    ```bash
    docker pull pyqlib/qlib_image_stable:stable
    ```
2. 启动一个新的 Docker 容器
    ```bash
    docker run -it --name <container name> -v <Mounted local directory>:/app pyqlib/qlib_image_stable:stable
    ```
3. 此时您已进入 docker 环境，可以运行 qlib 脚本。示例：
    ```bash
    >>> python scripts/get_data.py qlib_data --name qlib_data_simple --target_dir ~/.qlib/qlib_data/cn_data --interval 1d --region cn
    >>> python qlib/cli/run.py examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml
    ```
4. 退出容器
    ```bash
    >>> exit
    ```
5. 重启容器
    ```bash
    docker start -i -a <container name>
    ```
6. 停止容器
    ```bash
    docker stop <container name>
    ```
7. 删除容器
    ```bash
    docker rm <container name>
    ```
8. 如果您想了解更多信息，请参考 [文档](https://qlib.readthedocs.io/en/latest/developer/how_to_build_image.html)。

## 自动量化研究工作流
Qlib 提供了一个名为 `qrun` 的工具来自动运行整个工作流（包括构建数据集、训练模型、回测和评估）。您可以按照以下步骤启动自动量化研究工作流并进行图形化报告分析：

1. 量化研究工作流：使用 lightgbm 工作流配置 ([workflow_config_lightgbm_Alpha158.yaml](examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml) 运行 `qrun`，如下所示。
    ```bash
      cd examples  # 避免在包含 `qlib` 的目录下运行程序
      qrun benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml
    ```
    如果用户想在调试模式下使用 `qrun`，请使用以下命令：
    ```bash
    python -m pdb qlib/cli/run.py examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml
    ```
    `qrun` 的结果如下，请参考 [文档](https://qlib.readthedocs.io/en/latest/component/strategy.html#result) 了解有关结果的更多解释。

    ```bash

    'The following are analysis results of the excess return without cost.'
                           risk
    mean               0.000708
    std                0.005626
    annualized_return  0.178316
    information_ratio  1.996555
    max_drawdown      -0.081806
    'The following are analysis results of the excess return with cost.'
                           risk
    mean               0.000512
    std                0.005626
    annualized_return  0.128982
    information_ratio  1.444287
    max_drawdown      -0.091078
    ```
    这里是 `qrun` 和 [工作流](https://qlib.readthedocs.io/en/latest/component/workflow.html) 的详细文档。

2. 图形化报告分析：首先，运行 `python -m pip install .[analysis]` 安装所需的依赖项。然后使用 `jupyter notebook` 运行 `examples/workflow_by_code.ipynb` 以获取图形化报告。
    - 预测信号（模型预测）分析
      - 组的累积收益
      ![Cumulative Return](https://github.com/microsoft/qlib/blob/main/docs/_static/img/analysis/analysis_model_cumulative_return.png)
      - 收益分布
      ![long_short](https://github.com/microsoft/qlib/blob/main/docs/_static/img/analysis/analysis_model_long_short.png)
      - 信息系数 (IC)
      ![Information Coefficient](https://github.com/microsoft/qlib/blob/main/docs/_static/img/analysis/analysis_model_IC.png)
      ![Monthly IC](https://github.com/microsoft/qlib/blob/main/docs/_static/img/analysis/analysis_model_monthly_IC.png)
      ![IC](https://github.com/microsoft/qlib/blob/main/docs/_static/img/analysis/analysis_model_NDQ.png)
      - 预测信号（模型预测）的自相关
      ![Auto Correlation](https://github.com/microsoft/qlib/blob/main/docs/_static/img/analysis/analysis_model_auto_correlation.png)

    - 投资组合分析
      - 回测收益
      ![Report](https://github.com/microsoft/qlib/blob/main/docs/_static/img/analysis/report.png)
      <!-- 
      - Score IC
      ![Score IC](docs/_static/img/score_ic.png)
      - Cumulative Return
      ![Cumulative Return](docs/_static/img/cumulative_return.png)
      - Risk Analysis
      ![Risk Analysis](docs/_static/img/risk_analysis.png)
      - Rank Label
      ![Rank Label](docs/_static/img/rank_label.png)
      -->
   - 上述结果的 [解释](https://qlib.readthedocs.io/en/latest/component/report.html)

## 通过代码构建自定义量化研究工作流
自动工作流可能并不适合所有量化研究人员的研究工作流。为了支持灵活的量化研究工作流，Qlib 还提供了一个模块化接口，允许研究人员通过代码构建自己的工作流。[这里](examples/workflow_by_code.ipynb) 是一个通过代码自定义量化研究工作流的演示。

# 量化研究中的主要挑战和解决方案
量化投资是一个非常独特的场景，有许多关键挑战需要解决。
目前，Qlib 为其中的一些挑战提供了一些解决方案。

## 预测：寻找有价值的信号/模式
准确预测股票价格趋势是构建盈利投资组合的重要组成部分。
然而，金融市场中海量且格式多样的数据使得构建预测模型具有挑战性。

越来越多的 SOTA 量化研究工作/论文专注于构建预测模型以挖掘复杂金融数据中有价值的信号/模式，并在 `Qlib` 中发布。


### [量化模型 (论文) Zoo](examples/benchmarks)

这里是基于 `Qlib` 构建的模型列表。
- [基于 XGBoost 的 GBDT (Tianqi Chen, et al. KDD 2016)](examples/benchmarks/XGBoost/)
- [基于 LightGBM 的 GBDT (Guolin Ke, et al. NIPS 2017)](examples/benchmarks/LightGBM/)
- [基于 Catboost 的 GBDT (Liudmila Prokhorenkova, et al. NIPS 2018)](examples/benchmarks/CatBoost/)
- [基于 pytorch 的 MLP](examples/benchmarks/MLP/)
- [基于 pytorch 的 LSTM (Sepp Hochreiter, et al. Neural computation 1997)](examples/benchmarks/LSTM/)
- [基于 pytorch 的 GRU (Kyunghyun Cho, et al. 2014)](examples/benchmarks/GRU/)
- [基于 pytorch 的 ALSTM (Yao Qin, et al. IJCAI 2017)](examples/benchmarks/ALSTM)
- [基于 pytorch 的 GATs (Petar Velickovic, et al. 2017)](examples/benchmarks/GATs/)
- [基于 pytorch 的 SFM (Liheng Zhang, et al. KDD 2017)](examples/benchmarks/SFM/)
- [基于 tensorflow 的 TFT (Bryan Lim, et al. International Journal of Forecasting 2019)](examples/benchmarks/TFT/)
- [基于 pytorch 的 TabNet (Sercan O. Arik, et al. AAAI 2019)](examples/benchmarks/TabNet/)
- [基于 LightGBM 的 DoubleEnsemble (Chuheng Zhang, et al. ICDM 2020)](examples/benchmarks/DoubleEnsemble/)
- [基于 pytorch 的 TCTS (Xueqing Wu, et al. ICML 2021)](examples/benchmarks/TCTS/)
- [基于 pytorch 的 Transformer (Ashish Vaswani, et al. NeurIPS 2017)](examples/benchmarks/Transformer/)
- [基于 pytorch 的 Localformer (Juyong Jiang, et al.)](examples/benchmarks/Localformer/)
- [基于 pytorch 的 TRA (Hengxu, Dong, et al. KDD 2021)](examples/benchmarks/TRA/)
- [基于 pytorch 的 TCN (Shaojie Bai, et al. 2018)](examples/benchmarks/TCN/)
- [基于 pytorch 的 ADARNN (YunTao Du, et al. 2021)](examples/benchmarks/ADARNN/)
- [基于 pytorch 的 ADD (Hongshun Tang, et al.2020)](examples/benchmarks/ADD/)
- [基于 pytorch 的 IGMTF (Wentao Xu, et al.2021)](examples/benchmarks/IGMTF/)
- [基于 pytorch 的 HIST (Wentao Xu, et al.2021)](examples/benchmarks/HIST/)
- [基于 pytorch 的 KRNN](examples/benchmarks/KRNN/)
- [基于 pytorch 的 Sandwich](examples/benchmarks/Sandwich/)

非常欢迎您提交新量化模型的 PR。

每个模型在 `Alpha158` 和 `Alpha360` 数据集上的性能可以在 [这里](examples/benchmarks/README.md) 找到。

### 运行单个模型
上面列出的所有模型都可以用 ``Qlib`` 运行。用户可以通过 [benchmarks](examples/benchmarks) 文件夹找到我们提供的配置文件和有关模型的一些详细信息。更多信息可以在上面列出的模型文件中检索。

`Qlib` 提供了三种不同的方式来运行单个模型，用户可以选择最适合他们情况的一种：
- 用户可以使用上面提到的工具 `qrun` 基于配置文件运行模型的工作流。
- 用户可以基于 `examples` 文件夹中列出的 [脚本](examples/workflow_by_code.py) 创建 `workflow_by_code` python 脚本。

- 用户可以使用 `examples` 文件夹中列出的脚本 [`run_all_model.py`](examples/run_all_model.py) 来运行模型。这是要使用的特定 shell 命令的示例：`python run_all_model.py run --models=lightgbm`，其中 `--models` 参数可以接受上面列出的任意数量的模型（可用模型可以在 [benchmarks](examples/benchmarks/) 中找到）。有关更多用例，请参考文件的 [文档字符串](examples/run_all_model.py)。
    - **注意**: 每个基线都有不同的环境依赖项，请确保您的 python 版本符合要求（例如，由于 `tensorflow==1.15.0` 的限制，TFT 仅支持 Python 3.6~3.7）

### 运行多个模型
`Qlib` 还提供了一个脚本 [`run_all_model.py`](examples/run_all_model.py)，它可以运行多个模型进行多次迭代。（**注意**: 该脚本目前仅支持 *Linux*。未来将支持其他操作系统。此外，它也不支持多次并行运行同一个模型，这也将在未来的开发中修复。）

该脚本将为每个模型创建一个唯一的虚拟环境，并在训练后删除环境。因此，只会生成并存储实验结果，如 `IC` 和 `backtest` 结果。

这是一个运行所有模型 10 次迭代的示例：
```python
python run_all_model.py run 10
```

它还提供了立即运行特定模型的 API。有关更多用例，请参考文件的 [文档字符串](examples/run_all_model.py)。

### 破坏性变更
在 `pandas` 中，`group_key` 是 `groupby` 方法的参数之一。从 `pandas` 的 1.5 版本到 2.0 版本，`group_key` 的默认值已从 `no default` 更改为 `True`，这将导致 qlib 在运行期间报错。所以我们设置 `group_key=False`，但这并不能保证某些程序能正确运行，包括：
* qlib\examples\rl_order_execution\scripts\gen_training_orders.py
* qlib\examples\benchmarks\TRA\src\dataset.MTSDatasetH.py
* qlib\examples\benchmarks\TFT\tft.py



## [适应市场动态](examples/benchmarks_dynamic)

由于金融市场环境的非平稳性，数据分布可能会在不同时期发生变化，这使得基于训练数据构建的模型在未来的测试数据中性能下降。
因此，使预测模型/策略适应市场动态对于模型/策略的性能非常重要。

这里是基于 `Qlib` 构建的解决方案列表。
- [滚动再训练](examples/benchmarks_dynamic/baseline/)
- [基于 pytorch 的 DDG-DA (Wendi, et al. AAAI 2022)](examples/benchmarks_dynamic/DDG-DA/)

##  强化学习：建模连续决策
Qlib 现在支持强化学习，这是一项旨在对连续投资决策进行建模的功能。此功能通过从与环境的交互中学习以最大化某种累积奖励的概念，协助投资者优化其交易策略。

这里是基于 `Qlib` 构建的按场景分类的解决方案列表。

### [订单执行的 RL](examples/rl_order_execution)
[这里](https://qlib.readthedocs.io/en/latest/component/rl/overall.html#order-execution) 是此场景的介绍。下面所有的方法都在 [这里](examples/rl_order_execution) 进行了比较。
- [TWAP](examples/rl_order_execution/exp_configs/backtest_twap.yml)
- [PPO: "An End-to-End Optimal Trade Execution Framework based on Proximal Policy Optimization", IJCAL 2020](examples/rl_order_execution/exp_configs/backtest_ppo.yml)
- [OPDS: "Universal Trading for Order Execution with Oracle Policy Distillation", AAAI 2021](examples/rl_order_execution/exp_configs/backtest_opds.yml)

# 量化数据集 Zoo
数据集在量化中起着非常重要的作用。这里是基于 `Qlib` 构建的数据集列表：

| 数据集                                    | 美国市场 | 中国市场 |
| --                                         | --        | --           |
| [Alpha360](./qlib/contrib/data/handler.py) |  √        |  √           |
| [Alpha158](./qlib/contrib/data/handler.py) |  √        |  √           |

[这里](https://qlib.readthedocs.io/en/latest/advanced/alpha.html) 是使用 `Qlib` 构建数据集的教程。
非常欢迎您提交构建新量化数据集的 PR。


# 学习框架
Qlib 是高度可定制的，其许多组件都是可学习的。
可学习的组件是 `Forecast Model` 和 `Trading Agent` 的实例。它们基于 `Learning Framework` 层进行学习，然后应用于 `Workflow` 层中的多个场景。
学习框架也利用了 `Workflow` 层（例如共享 `Information Extractor`，基于 `Execution Env` 创建环境）。

根据学习范式，它们可以分为强化学习和监督学习。
- 对于监督学习，详细文档可以在 [这里](https://qlib.readthedocs.io/en/latest/component/model.html) 找到。
- 对于强化学习，详细文档可以在 [这里](https://qlib.readthedocs.io/en/latest/component/rl.html) 找到。Qlib 的 RL 学习框架利用 `Workflow` 层中的 `Execution Env` 来创建环境。值得注意的是，也支持 `NestedExecutor`。这使用户能够一起优化不同层级的策略/模型/代理（例如，为特定的投资组合管理策略优化订单执行策略）。


# 更多关于 Qlib
如果您想快速浏览 qlib 最常用的组件，您可以尝试 [这里](examples/tutorial/) 的 notebooks。

详细文档组织在 [docs](docs/) 中。
需要 [Sphinx](http://www.sphinx-doc.org) 和 readthedocs 主题来构建 html 格式的文档。
```bash
cd docs/
conda install sphinx sphinx_rtd_theme -y
# 否则，您可以使用 pip 安装它们
# pip install sphinx sphinx_rtd_theme
make html
```
您也可以直接在线查看 [最新文档](http://qlib.readthedocs.io/)。

Qlib 正在积极和持续开发中。我们的计划在路线图中，该路线图作为一个 [github 项目](https://github.com/microsoft/qlib/projects/1) 进行管理。



# 离线模式和在线模式
Qlib 的数据服务器可以部署为 `Offline` 模式或 `Online` 模式。默认模式是离线模式。

在 `Offline` 模式下，数据将部署在本地。

在 `Online` 模式下，数据将部署为共享数据服务。数据及其缓存将由所有客户端共享。由于缓存命中率更高，预计数据检索性能将得到提高。它也会消耗更少的磁盘空间。在线模式的文档可以在 [Qlib-Server](https://qlib-server.readthedocs.io/) 中找到。在线模式可以使用 [基于 Azure CLI 的脚本](https://qlib-server.readthedocs.io/en/latest/build.html#one-click-deployment-in-azure) 自动部署。在线数据服务器的源代码可以在 [Qlib-Server 仓库](https://github.com/microsoft/qlib-server) 中找到。

## Qlib 数据服务器的性能
数据处理的性能对于像 AI 技术这样的数据驱动方法非常重要。作为一个面向 AI 的平台，Qlib 提供了数据存储和数据处理的解决方案。为了演示 Qlib 数据服务器的性能，我们
将其与其他几种数据存储解决方案进行了比较。

我们通过完成相同的任务来评估几种存储解决方案的性能，
该任务从股票市场的基本 OHLCV 日数据（2007 年至 2020 年每天 800 只股票）创建一个数据集（14 个特征/因子）。该任务涉及数据查询和处理。

|                         | HDF5      | MySQL     | MongoDB   | InfluxDB  | Qlib -E -D  | Qlib +E -D   | Qlib +E +D  |
| --                      | ------    | ------    | --------  | --------- | ----------- | ------------ | ----------- |
| Total (1CPU) (seconds)  | 184.4±3.7 | 365.3±7.5 | 253.6±6.7 | 368.2±3.6 | 147.0±8.8   | 47.6±1.0     | **7.4±0.3** |
| Total (64CPU) (seconds) |           |           |           |           | 8.8±0.6     | **4.2±0.2**  |             |
* `+(-)E` 表示有（无） `ExpressionCache`
* `+(-)D` 表示有（无） `DatasetCache`

大多数通用数据库加载数据花费的时间太长。在深入研究底层实现后，我们发现数据在通用数据库解决方案中经过了太多层接口和不必要的格式转换。
这些开销极大地减慢了数据加载过程。
Qlib 数据以紧凑的格式存储，可以有效地组合成数组进行科学计算。

# 相关报告
- [Guide To Qlib: Microsoft’s AI Investment Platform](https://analyticsindiamag.com/qlib/)
- [微软也搞AI量化平台？还是开源的！](https://mp.weixin.qq.com/s/47bP5YwxfTp2uTHjUBzJQQ)
- [微矿Qlib：业内首个AI量化投资开源平台](https://mp.weixin.qq.com/s/vsJv7lsgjEi-ALYUz4CvtQ)

# 联系我们
- 如果您有任何问题，请在 [这里](https://github.com/microsoft/qlib/issues/new/choose) 创建 issue 或在 [gitter](https://gitter.im/Microsoft/qlib) 中发送消息。
- 如果您想为 `Qlib` 做贡献，请 [创建 pull requests](https://github.com/microsoft/qlib/compare)。
- 出于其他原因，欢迎通过电子邮件联系我们 ([qlib@microsoft.com](mailto:qlib@microsoft.com))。
  - 我们正在招聘新成员（全职员工和实习生），欢迎您的简历！

加入 IM 讨论组：
|[Gitter](https://gitter.im/Microsoft/qlib)|
|----|
|![image](https://github.com/microsoft/qlib/blob/main/docs/_static/img/qrcode/gitter_qr.png)|

# 贡献
我们感谢所有的贡献并感谢所有的贡献者！
<a href="https://github.com/microsoft/qlib/graphs/contributors"><img src="https://contrib.rocks/image?repo=microsoft/qlib" /></a>

在 2020 年 9 月我们将 Qlib 作为开源项目在 Github 上发布之前，Qlib 是我们组内的一个内部项目。不幸的是，内部提交历史没有保留。我们组的许多成员也为 Qlib 做出了很多贡献，包括 Ruihua Wang, Yinda Zhang, Haisu Yu, Shuyu Wang, Bochen Pang, 和 [Dong Zhou](https://github.com/evanzd/evanzd)。特别感谢 [Dong Zhou](https://github.com/evanzd/evanzd)，因为他是 Qlib 的初始版本作者。

## 指南

本项目欢迎贡献和建议。
**这里有一些提交 pull request 的
[代码标准和开发指南](docs/developer/code_standard_and_dev_guide.rst)。**

做贡献并不是一件难事。解决一个 issue（也许只是回答 [issues 列表](https://github.com/microsoft/qlib/issues) 或 [gitter](https://gitter.im/Microsoft/qlib) 中提出的问题），修复/发布一个 bug，改进文档甚至修复一个拼写错误都是对 Qlib 的重要贡献。

例如，如果您想为 Qlib 的文档/代码做贡献，您可以按照下图中的步骤操作。
<p align="center">
  <img src="https://github.com/demon143/qlib/blob/main/docs/_static/img/change%20doc.gif" />
</p>

如果您不知道如何开始贡献，可以参考以下示例。
| 类型 | 示例 |
| -- | -- |
| 解决 issues | [回答问题](https://github.com/microsoft/qlib/issues/749);  [发布](https://github.com/microsoft/qlib/issues/765) 或 [修复](https://github.com/microsoft/qlib/pull/792) bug |
| 文档 | [提高文档质量](https://github.com/microsoft/qlib/pull/797/files) ;  [修复拼写错误](https://github.com/microsoft/qlib/pull/774) | 
| 功能 |  实现 [请求的功能](https://github.com/microsoft/qlib/projects) 像 [这样](https://github.com/microsoft/qlib/pull/754); [重构接口](https://github.com/microsoft/qlib/pull/539/files) |
| 数据集 | [添加数据集](https://github.com/microsoft/qlib/pull/733) | 
| 模型 |  [实现新模型](https://github.com/microsoft/qlib/pull/689), [一些贡献模型的说明](https://github.com/microsoft/qlib/tree/main/examples/benchmarks#contributing) |

[Good first issues](https://github.com/microsoft/qlib/labels/good%20first%20issue) 被标记为表明它们很容易开始您的贡献。

您可以通过 `rg 'TODO|FIXME' qlib` 在 Qlib 中找到一些不完美的实现。

如果您想成为 Qlib 的维护者以做出更多贡献（例如帮助合并 PR，分类 issues），请通过电子邮件联系我们 ([qlib@microsoft.com](mailto:qlib@microsoft.com))。我们很高兴帮助升级您的权限。

## 许可证
大多数贡献要求您同意
贡献者许可协议 (CLA)，声明您有权并且确实授予我们
使用您的贡献的权利。有关详细信息，请访问 https://cla.opensource.microsoft.com。

当您提交 pull request 时，CLA 机器人会自动确定您是否需要提供
CLA 并适当地装饰 PR（例如，状态检查，评论）。只需按照机器人提供的说明操作即可。您只需在使用我们 CLA 的所有存储库中执行一次此操作。

本项目采用了 [Microsoft 开源行为准则](https://opensource.microsoft.com/codeofconduct/)。
有关更多信息，请参阅 [行为准则常见问题解答](https://opensource.microsoft.com/codeofconduct/faq/) 或
如有任何其他问题或意见，请联系 [opencode@microsoft.com](mailto:opencode@microsoft.com)。
