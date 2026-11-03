# CRU-Harness V3.2 三模型迁移评测

**简体中文** | [English](openai_transfer_20261102.md)

在同一冻结 V3.2 算法下，对既有 OOLONG-synth cohort 5 的 12 个来源分别运行三个指定 OpenAI 模型。全部 36 个模型—样本对完成后统一评分。这里报告迁移重放成绩，不是新的盲测，也没有 direct 生成比较臂。

## 逐模型成绩

| 模型 | 满分题/12 | 满分率 | 官方均分/100 | 严格EM/12 | 非空 | 错误 | 描述性Wilson95%来源单位区间 |
|---|---:|---:|---:|---:|---:|---:|---|
| gpt-6-sol | 9 | 75.00% | 75.00 | 5 | 12/12 | 0 | 46.8%–91.1% |
| gpt-6-luna | 6 | 50.00% | 50.00 | 4 | 9/12 | 3 | 25.4%–74.6% |
| gpt-6.1-terra | 10 | 83.33% | 83.33 | 5 | 12/12 | 0 | 55.2%–95.3% |

官方均分包含数字题 `0.75 ** abs(error)` 的部分分，不等同于准确率。全部失败、空输出和未知留在每模型 12 题分母中。本轮有12个不同来源context，跨模型重用形成36对；不同ID及全文哈希不证明统计独立，底层记录重用未排除。Wilson区间仅按12个来源单位作描述性展示，不能据此确认稳定领先。

## 用量与费用

| 模型 | 生成调用 | 已知输入tokens | 已知输出tokens | 计费或预留tokens | 未知usage请求 | 费用估计上界USD | 已知usage基础I/O估计USD |
|---|---:|---:|---:|---:|---:|---:|---:|
| gpt-6-sol | 119 | 265,864 | 39,791 | 305,655 | 0 | 3.056550 | 0.929638 |
| gpt-6-luna | 82 | 194,168 | 38,636 | 232,804 | 0 | 0.116402 | 0.038735 |
| gpt-6.1-terra | 118 | 265,532 | 39,757 | 305,289 | 0 | 1.526445 | 0.464317 |

正式生成的保守费用估计总上界为 **$4.699397**，冻结上限 $15。它不是实际账单。每次先按 `(准入输入+预留输出)×r_max` 持久化预留，已知usage按实际输入输出结算同一包络；未知交付保留整额。基础I/O估计只覆盖已知usage，缓存、cache-write和reasoning细分不重复加到总量。单次16k低于272k长输入档阈值；非default服务档等偏离需单列。

Standard价格、长输入阈值及冻结单价包络的官方来源：[gpt-6-sol](https://developers.openai.com/api/docs/models/gpt-6-sol) · [gpt-6-luna](https://developers.openai.com/api/docs/models/gpt-6-luna) · [gpt-6.1-terra](https://developers.openai.com/api/docs/models/gpt-6.1-terra)

独立的最小API可用性预检为33输入+15输出=48 tokens，不计入正式成绩或上述费用。计数请求另行统计，不充当生成调用。

## 身份与计数

| 请求模型 | 实际返回模型集合 | 返回服务档集合 | 计数API请求 | 计数缓存命中 |
|---|---|---|---:|---:|
| gpt-6-sol | gpt-6-sol | default | 188 | 480 |
| gpt-6-luna | gpt-6-luna | default | 117 | 328 |
| gpt-6.1-terra | gpt-6.1-terra | default | 181 | 476 |

| 模型 | Provider response_status计数 | Provider incomplete_reason计数 |
|---|---|---|
| gpt-6-sol | completed: 119 | none |
| gpt-6-luna | completed: 79, incomplete: 3 | max_output_tokens: 3 |
| gpt-6.1-terra | completed: 118 | none |

`incomplete/max_output_tokens`表示服务已返回但未完成，可能发生于预算约束下最后一次较小的输出预留；若usage已知，其真实输入输出仍计入账本。它与HTTP失败、未知usage或一个语义错误答案分开报告，不能据此推断模型理解能力。

## 完整来源与direct容量控制

| 模型 | 来源输入封装tokens范围 | direct准入输入tokens范围 | 原生窗可容纳 | 16k可容纳 | 65k累计可容纳 | 完整计量 |
|---|---:|---:|---:|---:|---:|---:|
| gpt-6-sol | 651,081–1,582,486 | 651,263–1,582,625 | 6/12 | 0/12 | 0/12 | 12/12 |
| gpt-6-luna | 651,081–1,582,486 | 651,263–1,582,625 | 6/12 | 0/12 | 0/12 | 12/12 |
| gpt-6.1-terra | 651,081–1,582,486 | 651,263–1,582,625 | 6/12 | 0/12 | 0/12 | 12/12 |

这里区分原生 **1,050,000 tokens**、主动施加的单次16,384和累计65,536。direct可容纳性使用完整提示准入输入（包含16-token安全余量）加实际预留输出 **1,024**；预留不是自动按2,048输出上限填写。API测量是包含格式的Responses输入封装，不等于历史Qwen无特殊token原文计数。没有执行direct生成，准入拒绝不等于答错，也不能填写direct准确率0%。模型权重及单次注意力窗口没有改变。

三个精确模型的原生窗口规格来源：[gpt-6-sol](https://developers.openai.com/api/docs/models/gpt-6-sol) · [gpt-6-luna](https://developers.openai.com/api/docs/models/gpt-6-luna) · [gpt-6.1-terra](https://developers.openai.com/api/docs/models/gpt-6.1-terra)

### 超原生窗口描述性子组

按每模型 `direct准入输入+实际预留输出1,024 > 1,050,000` 定义，属于容量计量子组，与全部12题分开。这里不沿用Qwen扩容倍数，也不代表direct答错。

| 模型 | 子组n | 满分题 | 官方均分/100 | 最长满分来源封装tokens | 子组最大来源封装tokens |
|---|---:|---:|---:|---:|---:|
| gpt-6-sol | 6 | 4 | 66.67 | 1577931 | 1582486 |
| gpt-6-luna | 6 | 2 | 33.33 | 1569097 | 1582486 |
| gpt-6.1-terra | 6 | 5 | 83.33 | 1577931 | 1582486 |

## 逐来源成绩 调用 tokens

每格为官方分/100 · 生成调用 · 计费或预留tokens。极小正分可能显示为0.00，JSON保留精确值。

| Sample ID | gpt-6-sol | gpt-6-luna | gpt-6.1-terra |
|---|---:|---:|---:|
| oolong-synth-test-920090057 | 100.00 · 24 · 65,046 | 100.00 · 22 · 65,148 | 100.00 · 24 · 65,239 |
| oolong-synth-test-520050245 | 100.00 · 20 · 65,218 | 0.00 · 2 · 6,576 | 100.00 · 20 · 65,294 |
| oolong-synth-test-420040065 | 0.00 · 27 · 65,089 | 0.00 · 9 · 32,154 | 0.00 · 27 · 65,084 |
| oolong-synth-test-220020070 | 100.00 · 1 · 1,478 | 100.00 · 1 · 1,478 | 100.00 · 1 · 1,473 |
| oolong-synth-test-220020041 | 100.00 · 2 · 5,915 | 100.00 · 3 · 6,579 | 100.00 · 2 · 5,915 |
| oolong-synth-test-820080226 | 100.00 · 2 · 2,541 | 100.00 · 3 · 4,176 | 100.00 · 2 · 2,541 |
| oolong-synth-test-621060211 | 0.00 · 26 · 65,327 | 0.00 · 23 · 65,348 | 100.00 · 25 · 64,778 |
| oolong-synth-test-821080221 | 0.00 · 10 · 25,489 | 0.00 · 11 · 40,255 | 0.00 · 10 · 25,423 |
| oolong-synth-test-721070068 | 100.00 · 2 · 3,102 | 100.00 · 2 · 3,102 | 100.00 · 2 · 3,102 |
| oolong-synth-test-721070053 | 100.00 · 1 · 1,466 | 100.00 · 1 · 1,466 | 100.00 · 1 · 1,461 |
| oolong-synth-test-921090209 | 100.00 · 2 · 2,438 | 0.00 · 3 · 3,976 | 100.00 · 2 · 2,433 |
| oolong-synth-test-621060242 | 100.00 · 2 · 2,546 | 0.00 · 2 · 2,546 | 100.00 · 2 · 2,546 |

## 计划与停止诊断

| 模型 | plan验证错误 | plan修复 | 无新增事实停止 | 停止时仅1条未知 | 记录的最终事实丢失 |
|---|---:|---:|---:|---:|---:|
| gpt-6-sol | 0 | 0 | 0 | 0 | 0 (12/12 measured) |
| gpt-6-luna | 0 | 0 | 2 | 0 | 0 (9/12 measured) |
| gpt-6.1-terra | 0 | 0 | 0 | 0 | 0 (12/12 measured) |

这些诊断只来自计划类型、聚合未知量及停止计数，不包含语义标签、候选值或记录正文。单条未知尚存不证明已经定位语义错误；证据计算闭合也不提供独立语义真值保证。逐题错误类型、计数失败及安全HTTP code见JSON。

## 冻结协议与解释边界

Responses使用reasoning=none、store=false、truncation=disabled，请求default服务档，不发送temperature或seed；不宣称确定性。每模型—样本独立共享16k/65k/32次/2k输出账本。12个来源已用于此前V3.2实验，本轮不据gold调优；只有新增传输与美元预算桥。

本轮没有新跑direct、RLM、MemGPT或RAG，不能估计Harness对各底座的因果增益、宣称跨模型普遍领先或无限准确上下文；不是官方排行榜提交。旧Qwen成绩不并入本轮分母。公开数据预训练接触和底层记录重复未排除。

Freeze SHA-256: `7be98f287df0e854f6753a5811071009518e1c250b8d56cf43c33464f2e112c9`

来源ID/哈希、数据revision、官方scorer commit、原算法和新桥源码哈希、36条白名单结果、完整计数与费用定义见[公开JSON](../benchmarks/openai_transfer_20261102.json)。原始问题、答案、事实账本、响应轨迹和本机路径不随报告分发。

[OOLONG](https://github.com/abertsch72/oolong) · [Dataset](https://huggingface.co/datasets/oolongbench/oolong-synth) · [Token counting](https://developers.openai.com/api/docs/guides/token-counting)

## 独立机制重放审计

[公开完整性审计](../benchmarks/openai_transfer_integrity_20261102.json)核验了 **36/36 令牌账本**，并重放 **33/33 份可用机制产物**，没有违规或最终已接纳事实丢失。Luna 三份终态错误没有返回机制结果，严格标为**不可审**，不计作通过。

| 模型 | 可用机制通过 | 终态错误不可审 | 证据完整／不完整 |
|---|---:|---:|---:|
| gpt-6-sol | 12/12 | 0 | 7 / 5 |
| gpt-6-luna | 9/9 | 3 | 6 / 3 |
| gpt-6.1-terra | 12/12 | 0 | 8 / 4 |

重放核验了 250 条原始分类回复、14,419 个正文绑定、108,732 个来源引用／保留物理记录，以及 95,530 条正文等价转移。这些数字是各模型产物相加，不是跨模型去重后的记录数。三份条件分组产物均通过独立归约复算。审计与生产链共用了部分解析器，不是从零实现的完全独立解析链。来源绑定、事实守恒、覆盖完整性和语义正确性仍须分别判断。

## 失败模式与迁移结论

- `420040065` 三路均失败。条件分组任务仍有大量记录未完成分类：Sol、Terra 各有 1,576 条已知、20,551 条未知，在分类预算耗尽后停止；Luna 保留 192 条已知后因无新增事实停止。固定预算下的证据获取仍是瓶颈。
- `821080221` 三路也均未满分。Terra 已有 39,056 条已知、零未知记录，数值答案仍然错误；Sol 有 38,977 条已知、79 条未知。覆盖完整与账本计算正确，并不证明语义分类或查询解释正确。
- Luna 三项未完成响应中，两项触及完整的 2,048 输出上限，一项在接近每题总预算时仅预留了 818 输出 tokens。其余错误还包括无新增事实停止，以及报告覆盖完整却回答错误。只增加输出空间不足以证明语义准确率提高。

结果提供了不改算法即可迁移到另一提供方并完成这些任务的实际证据；Terra 在本组的点估计最高。样本少且复用了既有来源，尚不能确立稳定排名或普遍领先。后续有待验证的工程方向是按预期输出大小调整批次，以及独立复核语义判断；本轮没有应用这些修改。

## 代码与复现前提

新增 [Responses 适配器](../long_context_agent/openai_responses_transport.py)和[冻结迁移运行器](../scripts/benchmark_openai_transfer.py)在正式推理前通过 **54 项针对性离线检查**。[归档导出器](../scripts/export_openai_transfer_report.py)在统一评分后生成公开指标，同时提供[无凭证配置模板](../benchmarks/openai_transfer_config.example.json)及[脱敏 V3.2 算法指纹](../benchmarks/openai_transfer_v32_reference.json)。

推理仍需模板指定的原始本地 `inputs.jsonl`、`manifest.json`、`labels.json`。这些含来源的产物未随仓库分发。需取得固定 revision 的上游数据，并重建或提供符合公开哈希的产物；当前仓库不提供一条命令重建数据的流程。随附 scorer 为 `third_party/oolong/eval_helpers_original.py`。严格运行器使用 macOS `sandbox-exec`，在推理阶段拒绝读取标签。

脱敏指纹只保留原算法哈希，不复制包含本机路径的完整参考 freeze。使用它的新复现会记录新的 reference hash 和 run freeze，不冒充此次正式实验的冻结身份。所需输入在本地就绪后可执行：

```bash
python -m scripts.benchmark_openai_transfer freeze --config benchmarks/openai_transfer_config.example.json
# 仅在本机设置 OPENAI_API_KEY；不要放进配置文件或仓库。
python -m scripts.benchmark_openai_transfer run --config benchmarks/openai_transfer_config.example.json --execute-frozen-run
python -m scripts.benchmark_openai_transfer score --config benchmarks/openai_transfer_config.example.json
```

上述命令只有在显式调用时才会进行新的付费推理。归档导出器核对本次原始 run 身份，不是任意新 freeze 的通用导出器。

公开指标的离线算术与指纹核验不需要原始数据、凭证或网络：

```bash
python3 -S scripts/verify_openai_transfer_results.py
```

它核对公开成绩汇总、用量、容量子组、完整性审计与源码指纹；不替代使用原始输出和标签的评分重放。

[独立评分与费用终审](../benchmarks/openai_transfer_score_audit_20261102.json)逐题重执行固定官方 scorer 和本地解析器，36 项均与公开结果一致；319 笔费用预留、实际模型与服务档核验通过。该终审与上方机制重放分别进行。
