# 版本实验记录

按版本归档每次跑通后的结果、诊断与改动依据。新版本追加文件，**不覆盖旧文件**，便于纵向对比。

| 版本 | 文件 | 数据 | 核心结论 |
|---|---|---|---|
| v1 | [v1_baseline.md](v1_baseline.md) | 合成数据（demo） | 链路跑通，成本安全边际不足；结果不构成策略有效性证据 |

## 命名与写法约定

- 文件名 `v{n}_{关键改动}.md`，例如 `v2_maker_fill.md`
- 每篇固定包含：运行配置 / 结果数据 / 诊断结论 / 与上一版的 diff / 下一版待办
- 结果表直接引自 `artifacts/` 下的 csv 与 json，**跑新版本前先把旧 artifacts 归档**，否则会被覆盖：

```powershell
Copy-Item -Recurse qlib/examples/btc_minute/artifacts qlib/examples/btc_minute/reports/artifacts_v1
```
