# LongMemEval-V2：Qwen 直连与多模态 CRU-Harness 对照

本轮在本地 Mac 完成：Harness **8/20（40%）**，可用的原生首尾直连 **1/20（5%）**，提高 **35 个百分点**。完整直连 20 题均在生成前因容量不足被本地准入拒绝，语义准确率不适用。仅调用已有 Qwen 推理服务，没有修改远端部署文件。

这是预先固定的 **20 题 small 层级试评**：web 与 enterprise 各 10 题，五类能力各 4 题，4 题带题图。各领域共享一份 100 轨迹历史，共 200 条轨迹、5,095 张历史截图。不是 20 份独立历史，也不是完整 451 题或官方排行榜成绩。

| 路径 | 正确／总题数 | 返回 | UNKNOWN | 平均输入＋输出 tokens | 平均端到端秒 |
|---|---:|---:|---:|---:|---:|
| direct_full | — (20 capacity denials) | 0 | 0 | 0 | 0.00 |
| direct_headtail_native | 1/20 (5%) | 20 | 8 | 252,478 | 23.95 |
| harness | 8/20 (40%) | 20 | 0 | 32,792 | 20.10 |

Harness 累计输入＋输出 **655,835 tokens**，原生首尾直连 **5,049,555 tokens**，减少 **87.01%**。分别为 59 次与 20 次生成；另有 14 次评分调用与 4 次预检生成，独立记账。没有未知用量、API 失败或预算越界。

## 准确率拆分与失败分析

| 子集 | 原生首尾直连 | Harness |
|---|---:|---:|
| 确定性官方评分 | 1/12 | 5/12 |
| Qwen 裁判评分 | 0/8 | 3/8 |
| 题目自带截图 | 0/4 | 1/4 |
| dynamic | 0/4 | 2/4 |
| gotchas | 0/4 | 1/4 |
| premise_awareness | 0/4 | 2/4 |
| procedure | 1/4 | 2/4 |
| static | 0/4 | 1/4 |
| web | 1/10 | 3/10 |
| enterprise | 0/10 | 5/10 |

逐题配对：两路都答对 1 题，只有 Harness 答对 7 题，只有首尾直连答对 0 题，两路都未答对 12 题。Harness 仍有 **12/20 答错且没有 UNKNOWN**，首尾直连为 11 道错误与 8 道 UNKNOWN；必须继续改善证据不足时的拒答校准。4 道带题图样本中，Harness 仅答对 1 道，尚不足以证明稳定的视觉记忆能力。

本轮 review 解析错误为 0，已登记的支持证据范围最终丢失计数为 0。这证明已声明证据的机械保留，不证明召回完整或证据语义正确。下一轮应将召回遗漏、状态时序判断、视觉定位与错误前提识别分别诊断；当前结果不再用于反复调优后冒充新留出测试。

## 上下文容量与预算

- **34,235,595 Qwen canonical-text tokens**; 93,694,606 characters; 3,358 states. **130.60×** native 262,144; **2089.57×** working ceiling 16,384.
- **31,476,162 Qwen canonical-text tokens**; 77,368,369 characters; 1,737 states. **120.07×** native 262,144; **1921.15×** working ceiling 16,384.

最大来源与最长答对来源均为 **34,235,595 个规范化文本 tokens**。这是完整轨迹 JSON 文本经 Qwen 精确计数的长度，不含截图像素，不等于官方论文的 tokenizer 口径；不是模型单次阅读长度，也没有测得 Harness 的硬极限。完整文本与原始截图留在本地外部记忆，模型按检索窗口读取证据。

完整直连与首尾直连：单次及累计 262,144 tokens、1 次生成、最多 1,024 输出。首尾直连截取两段原始文本，最多 16 张原图，按保留状态均匀抽取；完整多模态提示经精确计数后缩减至可准入。Harness：单次 16,384、累计 65,536 tokens、最多 8 次物理请求、最多 3 轮证据 review 加最终回答；每次最多 3 张历史原图，保留题图。**计算预算不相等**；这不是同预算因果估计，也未与全源视觉扫描、其他检索策略做消融。

## 新增适配与原能力保护

新增 `long_context_agent/trajectory_memory.py`：保留所有状态字段，建立轨迹／状态／字符范围绑定，将原始截图与 SHA-256 绑定；先组装图片再计数和推理。检索复用既有 SourceIndex 与证据缺口协议，并累积已验证范围；最终输入使用官方领域提示及原图，不将未经验证的临时答案作为事实。没有预生成截图描述、缩图或答案缓存。

原 `harness.py`、包入口及文本／结构化事实控制器未修改。新增 11 项检查连同 89 项回归检查共 **100 passed**。此前 LongBench 公开核验的 58 份源码指纹保持不变。没有改变原 Python 环境依赖；轨迹模块按需导入。

## 评分隔离、审计与复核

元数据分层哈希选样，seed=20261103，不用题目正文或答案选样。选样预处理仅将标签搬运至隔离目录，全部 60 条终态输出固定后才启用评分阶段的标签访问；推理子进程由 Mac 沙盒禁止读取标签目录、原始带答案问题文件及报告目录。使用固定官方评分函数和裁判提示，但 LLM 裁判由默认 gpt-5.2 改为 Qwen3.8-Flash-Next-FP8。故 8 道裁判题有同模型评判偏差风险，不能混称官方 LAFS 分数。

Offline audit: **60/60 ledgers**, **60/60 score re-parses**, **60/60 controller replays**, **507 image occurrences**, **79 generations** and **624 tokenizer events**. No provider/tokenizer calls during replay.

重放核对保存的精确计数与生产控制器，复核官方评分结果和图片绑定；它不是独立的人工语义验证。公开 JSON 仅含白名单元数据、逐调用预算收据与哈希，原题、标准答案、模型原始回复、原图和凭证不随报告发布。公开核验器能重算指标和预算、核对源码与审计哈希，不能从未公开内容独立判断答案语义。

```bash
python3 -S scripts/verify_longmemeval_v2_results.py
```

保留本机完整数据时，可运行 `python scripts/audits/audit_longmemeval_v2.py` 离线重放，不会再次生成。推理与评分入口为 `python -m scripts.benchmark_longmemeval_v2 {freeze,run,score}`；`run/score` 使用既有 provider 环境文件，冻结目录不会被覆盖。新机器复跑需从下述固定上游版本重新取得数据并生成同格式选样／图片清单，不是只下载报告即可重跑模型。

## 逐题结果

| ID | Domain | Ability | Query image | Full Direct | Native head/tail | Harness |
|---|---|---|---|---|---|---|
| c0696888 | enterprise | dynamic | False | capacity_denied | incorrect | correct |
| 882965b7 | enterprise | dynamic | False | capacity_denied | UNKNOWN | incorrect |
| c35bfeed | enterprise | gotchas | True | capacity_denied | incorrect | incorrect |
| ce9aa351 | enterprise | gotchas | True | capacity_denied | incorrect | incorrect |
| ebad58ed | enterprise | premise_awareness | False | capacity_denied | UNKNOWN | correct |
| 19935d2d | enterprise | premise_awareness | False | capacity_denied | incorrect | correct |
| 95ae824f | enterprise | procedure | False | capacity_denied | UNKNOWN | incorrect |
| 73cbefdc | enterprise | procedure | False | capacity_denied | UNKNOWN | correct |
| f1f9de44 | enterprise | static | False | capacity_denied | UNKNOWN | incorrect |
| aef57896 | enterprise | static | False | capacity_denied | UNKNOWN | correct |
| b828a6b2 | web | dynamic | False | capacity_denied | UNKNOWN | incorrect |
| fe18a443 | web | dynamic | False | capacity_denied | incorrect | correct |
| 2cc1b9ca | web | gotchas | True | capacity_denied | incorrect | incorrect |
| 914ab1d4 | web | gotchas | True | capacity_denied | incorrect | correct |
| cdb47c5d | web | premise_awareness | False | capacity_denied | UNKNOWN | incorrect |
| a2a71197 | web | premise_awareness | False | capacity_denied | incorrect | incorrect |
| 597645be | web | procedure | False | capacity_denied | incorrect | incorrect |
| 99ae54d1 | web | procedure | False | capacity_denied | correct | correct |
| 36fe48ef | web | static | False | capacity_denied | incorrect | incorrect |
| 762d34b8 | web | static | False | capacity_denied | incorrect | incorrect |

## Benchmark 来源与版本

[Official repository](https://github.com/xiaowu0162/LongMemEval-V2) · [Paper](https://arxiv.org/abs/2605.12493) · [Official dataset](https://huggingface.co/datasets/xiaowu0162/longmemeval-v2)

Code: `2cc8c540bdb87fe6761629b585e727e1c4704520`; data: `f152293e235517d504809563c833d7190b8c713b`; freeze SHA-256: `c9f2d218c783f4f6e71d0278c898007d3469bb34019703e77424fb14fe6cec25`.

上游小文件 checksums 清单不包含大型截图归档；两份归档按固定版本的官方 LFS 对象 SHA 校验，5095 张历史原图与 4 张题图另逐张核对。代码采用上游 Apache-2.0 许可，见第三方声明；不据此推定来源数据的再分发权。

端到端时间含计数和推理，离线建索引另存；不等于官方 memory-query 延迟。只报告 Qwen 用量，不推算美元费用。当前只覆盖 small、两个共享历史及一轮生成，不能外推到完整榜单或所有模型。

[Results JSON](../benchmarks/longmemeval_v2_qwen_20261103.json) · [Offline audit](../benchmarks/longmemeval_v2_qwen_20261103_audit.json) · [Third-party notice](../third_party/longmemeval_v2/README.md)
