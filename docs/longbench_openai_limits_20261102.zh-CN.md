# LongBench v2：原生直连与有界 Harness 极长上下文对照

**简体中文** | [English](longbench_openai_limits_20261102.md)

本实验在12个未填充的自然LongBench来源上，分别运行三个精确模型的完整原生直连和有界证据Harness，共72条主逻辑结果，另有3条独立容量探针。第一阶段75条终态中有39条计数网络失败、没有发送任何生成；随后单独冻结这39条的恢复协议，新增39条执行记录。之后按预登记的长度和剩余资金条件运行独立首尾截断质量对照。全部原始、恢复及条件对照终态齐备（或明确预算跳过）后，才统一读取标签评分。

这是已消费来源的长度偏置回归评测，不是新的盲测、全503题成绩或官方榜单。两条路径使用不同计算预算，不能解释为同计算量准确率提升。

## 原始不完整试验与登记恢复

下表先保留第一阶段的正确交付结果，再给出恢复完成后的主表，避免把基础设施恢复藏成算法准确率提升。39条选择仅依据已冻结的generic-network错误、空生成账本、零物理生成及零令牌费用，不读取输出或正确性。所有已经发送生成的原始任务均未重跑。

| 模型 | 原始Direct正确交付／12 | 原始Harness正确交付／12 | 登记恢复任务数 | 原始生成请求 | 新增恢复生成请求 |
|---|---:|---:|---:|---:|---:|
| gpt-6-sol | 0/12 (0.00%) | 3/12 (25.00%) | 14 | 24 | 51 |
| gpt-6-luna | 0/12 (0.00%) | 4/12 (33.33%) | 13 | 40 | 37 |
| gpt-6.1-terra | 0/12 (0.00%) | 4/12 (33.33%) | 12 | 55 | 22 |

| 模型 | Direct恢复indices | Harness恢复indices | 独立探针恢复indices |
|---|---|---|---|
| gpt-6-sol | 35, 147, 335, 449, 453 | 33, 35, 143, 147, 257, 335, 395, 449 | 35 |
| gpt-6-luna | 143, 251, 260, 438, 449, 453 | 35, 143, 251, 395, 438, 453 | 35 |
| gpt-6.1-terra | 33, 35, 143, 147, 257, 395, 449, 453 | 35, 251, 257 | 35 |

原始75条记录和新增39条均不可变。恢复只增加了计数网络请求的最多3次尝试及5／15秒等待、错误数字字段观测；不做生成重试，不修改56份原算法／控制器源码，不重置已消费生成预算。原始39条均为零消耗，因此原任务与恢复的合计仍受同一逻辑预算限制。

## 恢复后的主结果与容量拒绝

| 模型 | Direct全部12任务正确交付 | Direct本地容量拒绝 | Direct公开规格准入组正确交付 | Harness全部12正确 | 超公开原生窗子组Harness正确 |
|---|---:|---:|---:|---:|---:|
| gpt-6-sol | 0/12 (0.00%) | 8 | 0/4 (0.00%) | 8/12 (66.67%) | 6/8 (75.00%) |
| gpt-6-luna | 0/12 (0.00%) | 8 | 0/4 (0.00%) | 6/12 (50.00%) | 5/8 (62.50%) |
| gpt-6.1-terra | 1/12 (8.33%) | 8 | 1/4 (25.00%) | 7/12 (58.33%) | 5/8 (62.50%) |

Direct的全部12题列衡量任务交付，容量拒绝没有语义答案，不标为模型答错。公开规格准入组的主指标保留服务端上下文拒绝、其他API失败、缺失和错误答案在分母中，属于正确交付率而非纯语义准确率。Harness主指标保留全部12题。

### Direct拒绝与基础设施失败分解

| 模型 | 本地超窗拒绝 | 服务端context拒绝 | 其中公开规格准入范围内 | 基础设施终态失败 | 准入未知 | 实际返回答案 |
|---|---:|---:|---:|---:|---:|---:|
| gpt-6-sol | 8 | 2 | 2 | 1 | 0 | 1 |
| gpt-6-luna | 8 | 2 | 2 | 0 | 0 | 2 |
| gpt-6.1-terra | 8 | 2 | 2 | 0 | 0 | 2 |

按105万公开规格精确计数可准入，不等于服务端实际接受。若服务端在该范围内返回context_length_exceeded，仍保留原始error状态并单列，不能写成语义答错、本地容量拒绝或已验证105万窗口可用。429和未知网络交付也分别归类。

## 原生可容纳来源的配对比较

| 模型 | 主配对n | 双方正确 | 仅Direct正确 | 仅Harness正确 | 双方未成功 | 双方有答案条件配对n（次要） |
|---|---:|---:|---:|---:|---:|---:|
| gpt-6-sol | 4 | 0 | 0 | 2 | 2 | 1 |
| gpt-6-luna | 4 | 0 | 0 | 1 | 3 | 2 |
| gpt-6.1-terra | 4 | 1 | 0 | 1 | 2 | 2 |

双方有答案的条件统计仅作为次要结果，避免丢弃失败后放大准确率。样本少且每路径只运行一次，不能据此确立稳定排名；容量之外的错误和基础设施限制需分别解释。

| 模型 | Direct实际context拒绝来源数 | 同批来源Harness正确交付 |
|---|---:|---:|
| gpt-6-sol | 2 | 1/2 (50.00%) |
| gpt-6-luna | 2 | 0/2 (0.00%) |
| gpt-6.1-terra | 2 | 1/2 (50.00%) |

该表只纳入实际HTTP context_length_exceeded，区别于因公开规格超窗而未发送生成的8题容量组。探针另列且不参与此表。

## 已观察到的长度范围

| 模型 | Direct最大实际返回答案提示 | Direct最短实际上下文拒绝提示 | Harness最大尝试提示 | Harness最大返回答案提示 | Harness最大正确提示 | 最长正确／公开原生窗 | 最长正确／16k工作窗 |
|---|---:|---:|---:|---:|---:|---:|---:|
| gpt-6-sol | 845,093 (index 147) | 997,650 (index 335) | 4,264,583 (index 143) | 4,264,583 (index 143) | 4,264,583 (index 143) | 4.06x | 260.29x |
| gpt-6-luna | 845,093 (index 147) | 997,650 (index 335) | 4,264,583 (index 143) | 4,264,583 (index 143) | 4,264,583 (index 143) | 4.06x | 260.29x |
| gpt-6.1-terra | 845,093 (index 147) | 997,650 (index 335) | 4,264,583 (index 143) | 4,264,583 (index 143) | 4,264,583 (index 143) | 4.06x | 260.29x |

长度单位是完整原生Direct提示的Responses准入输入，含资源说明和16-token安全余量；它不是无格式原文token数。105万原生窗口与16,384实验工作窗是两个不同分母。Direct实际成功与拒绝值只描述此次采样区间，不是精确硬上限；错误消息没有报告数值上限。最大尝试、返回答案和正确答案分别列出；单个多选题答对不能证明稳定理解整个来源，也不证明已找到绝对极限。

三模型原生规格来源：[gpt-6-sol](https://developers.openai.com/api/docs/models/gpt-6-sol) · [gpt-6-luna](https://developers.openai.com/api/docs/models/gpt-6-luna) · [gpt-6.1-terra](https://developers.openai.com/api/docs/models/gpt-6.1-terra)

## 三条独立API容量探针

| 模型 | 来源index／准入input tokens | 实际状态 | 安全HTTP/code | 生成错误中报告的context上限 | 物理请求 | 已知tokens | 未知usage预留tokens | 保守费用上界 |
|---|---:|---|---|---:|---:|---:|---:|---:|
| gpt-6-sol | 35 / 1,138,985 | provider_context_rejected | 400/context_length_exceeded | not_reported | 1 | 0 | 1,140,009 | USD 5.710285 |
| gpt-6-luna | 35 / 1,138,985 | provider_context_rejected | 400/context_length_exceeded | not_reported | 1 | 0 | 1,140,009 | USD 0.285514 |
| gpt-6.1-terra | 35 / 1,138,985 | provider_context_rejected | 400/context_length_exceeded | not_reported | 1 | 0 | 1,140,009 | USD 2.855142 |

探针只核验提供方接受／拒绝和传输结果，不参与准确率。HTTP400只有明确context_length_exceeded才解释为上下文拒绝；429为限流／配额，普通400或传输错误不冒充上下文上限。恢复包装器只保留白名单错误码、参数与数字候选，错误正文不公开；报告中的上限是提供方错误消息给出的数值，不等于独立实测极限。没有usage时保留整额预留，不把预留tokens写成实际推理用量。

## 用量与费用

| 模型／路径 | 生成请求 | 已知输入 | 已知输出 | 未知usage请求 | 未知预留tokens | 账本错误请求 | 费用估计上界 |
|---|---:|---:|---:|---:|---:|---:|---:|
| gpt-6-sol / direct | 4 | 845,077 | 11 | 3 | 2,166,905 | 3 | USD 14.790385 |
| gpt-6-sol / harness | 70 | 559,847 | 5,468 | 1 | 9,129 | 1 | USD 1.492480 |
| gpt-6-sol / capacity_probes | 1 | 0 | 0 | 1 | 1,140,009 | 1 | USD 5.710285 |
| gpt-6-sol / headtail_control | 2 | 1,532,419 | 22 | 0 | 0 | 0 | USD 7.662425 |
| gpt-6-luna / direct | 4 | 963,177 | 22 | 2 | 2,047,765 | 2 | USD 0.739011 |
| gpt-6-luna / harness | 72 | 563,898 | 6,438 | 2 | 21,207 | 2 | USD 0.077893 |
| gpt-6-luna / capacity_probes | 1 | 0 | 0 | 1 | 1,140,009 | 1 | USD 0.285514 |
| gpt-6-luna / headtail_control | 2 | 1,532,419 | 22 | 0 | 0 | 0 | USD 0.383121 |
| gpt-6.1-terra / direct | 4 | 963,177 | 25 | 2 | 2,047,765 | 2 | USD 7.390122 |
| gpt-6.1-terra / harness | 72 | 580,977 | 6,129 | 0 | 0 | 0 | USD 0.756866 |
| gpt-6.1-terra / capacity_probes | 1 | 0 | 0 | 1 | 1,140,009 | 1 | USD 2.855142 |
| gpt-6.1-terra / headtail_control | 2 | 1,532,419 | 22 | 0 | 0 | 0 | USD 3.831212 |

原始、恢复与条件首尾对照各阶段含探针共235次生成请求，已知输入加输出9,091,569 tokens，另保留未知usage额度9,712,798 tokens。各阶段正式生成保守费用上界为USD 45.974458；另加预检后合计USD 45.975822，共同低于USD50全局上限。这些是保守估算，不是实际账单。

| 运行阶段（不含准备与预检计数） | 物理生成 | 逻辑计数查询（含缓存） | 实际计数HTTP尝试 | 计数重试尝试 | 生成费用估计上界 |
|---|---:|---:|---:|---:|---:|
| original_infrastructure_incomplete | 119 | 523 | 215 | 0 | USD 9.890960 |
| registered_recovery_new_work | 110 | 445 | 170 | 0 | USD 24.206740 |
| headtail_quality_control | 6 | 24 | 6 | 0 | USD 11.876759 |
| combined | 235 | 992 | 391 | 0 | USD 45.974458 |

该运行阶段表保留原始失败计数尝试及恢复的逐次计数重试；它不包含单列的候选筛选预检与首尾裁剪准备。下表仅汇总本次实验已记录的计数请求范围，不代表项目全部历史计数。计数请求不占生成token。

| 计数范围 | 实际计数HTTP尝试 |
|---|---:|
| runtime_original_recovery_and_control | 391 |
| headtail_trimming_preparation | 6 |
| separate_candidate_selection_preflight | 34 |
| all_recorded_count_attempts | 431 |

每次生成前按输入的普通／缓存读／默认30分钟写入最高单价及输出单价持久化预留。输入超过272k时，整条请求输入／缓存单价乘2、输出乘1.5。已知usage按准入价格包络结算，未知usage保留整额；缓存和reasoning细分不重复加到token总量。

冻结单价与规格来源：[gpt-6-sol](https://developers.openai.com/api/docs/models/gpt-6-sol) · [gpt-6-luna](https://developers.openai.com/api/docs/models/gpt-6-luna) · [gpt-6.1-terra](https://developers.openai.com/api/docs/models/gpt-6.1-terra)

正式实验之外，三次API可用性预检共249输入＋15输出＝264tokens。计数API尝试及初始字段大小错误单列；它们不是生成token用量。

## 控制器捕获的内部错误

| 模型 | Harness终态错误 | 账本错误请求 | 捕获候选错误 | 捕获选项verifier错误 | 其中解析错误 | 显式verifier重试 |
|---|---:|---:|---:|---:|---:|---:|
| gpt-6-sol | 0 | 1 | 0 | 3 | 0 | 0 |
| gpt-6-luna | 0 | 2 | 0 | 2 | 0 | 0 |
| gpt-6.1-terra | 0 | 0 | 0 | 0 | 0 | 0 |

控制器可在候选或验证器出错后继续交付答案，因此终态无错误不代表内部所有请求成功。以物理账本为调用依据；明确的恢复调用和重试共同计入请求、token和费用预算。同一失败可能同时出现在账本与控制器捕获计数中，两者不可相加为独立失败数。

## 逐来源结果

每格为Direct预测／Harness预测，括号为正确标记；容量拒绝显示denied而非答错。A/B/C/D仅是已公开基准的短选项字母，不含选项正文。

| Index / ID | Domain | Reference | gpt-6-sol | gpt-6-luna | gpt-6.1-terra |
|---|---|---|---|---|---|
| 143 / 66f3df1e821e116aacb2f7be | Code Repository Understanding | A | denied / A ✓ | denied / A ✓ | denied / A ✓ |
| 260 / 66fb75adbb02136c067c7d73 | Code Repository Understanding | C | denied / C ✓ | denied / C ✓ | denied / C ✓ |
| 453 / 66f40c5e821e116aacb30ad2 | Code Repository Understanding | A | denied / B ✗ | denied / B ✗ | denied / B ✗ |
| 449 / 6724cae7bb02136c067d79be | Long Structured Data Understanding | B | denied / B ✓ | denied / B ✓ | denied / C ✗ |
| 438 / 66ec17e4821e116aacb1a6a7 | Multi-Document QA | D | denied / D ✓ | denied / C ✗ | denied / D ✓ |
| 33 / 66f40e44821e116aacb30b45 | Long In-context Learning | C | denied / D ✗ | denied / C ✓ | denied / B ✗ |
| 147 / 66ebd34d5a08c7b9b35e035d | Single-Document QA | C | B ✗ / B ✗ | B ✗ / B ✗ | B ✗ / B ✗ |
| 257 / 6719185cbb02136c067d40ab | Long-dialogue History Understanding | B | missing / B ✓ | A ✗ / B ✓ | B ✓ / B ✓ |
| 395 / 66f28794821e116aacb298ee | Long In-context Learning | A | provider_rejected / D ✗ | provider_rejected / B ✗ | provider_rejected / D ✗ |
| 335 / 6700d19dbb02136c067cb024 | Long Structured Data Understanding | C | provider_rejected / C ✓ | provider_rejected / A ✗ | provider_rejected / C ✓ |
| 35 / 66fa3843bb02136c067c655d | Code Repository Understanding | D | denied / D ✓ | denied / A ✗ | denied / D ✓ |
| 251 / 66fa1e4cbb02136c067c5edf | Code Repository Understanding | A | denied / A ✓ | denied / A ✓ | denied / A ✓ |

## 来源、协议与解释边界

来源为[LongBench v2官网](https://longbench2.github.io/)、[官方代码](https://github.com/THUDM/LongBench)及[官方数据](https://huggingface.co/datasets/zai-org/LongBench-v2)。HF split名为train。原始缓存没有记录上游revision，本报告保留null并公布缓存、输入和逐来源哈希；不事后补造版本。

完整直连使用1,050,000单次与累计上限、1次生成、1,024输出上限。Harness使用16,384单次、65,536累计、32次生成、2,048单次输出上限；三候选证据视图及选项支持／反对／无证据验证共享账本。所有模型reasoning=none、store=false、truncation=disabled、default服务档，不发送temperature或seed。不做隐式重试，也不复用Direct答案。

API初检发现单字符串10,485,760字符字段限制；长字符串改为同一消息中按原顺序的input_text片段，每片最多10,000,000字符，拼接后与原提示完全相同。计数与生成使用同一无损编码。该修复只解决传输字段限制，不增加模型原生上下文；本轮没有人工重复或填充来源。

本轮运行的是LongBench多窗口证据与选项验证路径，区别于OOLONG的V3.2结构化事实归约路径。完整来源保留在外部，模型只接收有限证据；输出正确不等于模型逐token阅读全部来源。12个长度偏置来源、一次运行和多选题偶然性限制了质量边界的解释。

Freeze SHA-256: 941c86120e66020439571fbe8888999c643ebcf83fc315d7ec7e1e0e1045208d
Recovery freeze SHA-256: febefea46171ba7d843104aa8857bff82670b597e6b0bb0de6b640e7c7f312c1

56份原算法源码、恢复及首尾对照包装器指纹、原始75条结果、39条恢复资格及记录来源、恢复后的72条主逻辑结果、3条探针与6条附加对照，以及精确计数、短选项字母、已知／未知用量与费用定义见[公开JSON](../benchmarks/longbench_openai_limits_20261102.json)。原始问题、来源正文、选项正文、证据窗口、模型原始回复、凭证及本机路径不随报告分发。

公开文件可用标准库独立核验器核对短选项计分、样本矩阵、各阶段用量与费用守恒、代码指纹及隐私字段；它不会调用API或读取私有标签。该核验不是对未公开API原始回复与访问日志的重新执行。

```bash
python3 -S scripts/verify_longbench_openai_limits_results.py
```

## 附加质量对照：有损首尾截断 Direct

该对照在开标签前，按剩余费用条件固定选择最长来源143和260（足额时），或仅143，或明确预算跳过。每题最多1次生成，完整提示输入加输出上限800,000、输出上限1,024；使用同一份按字符均衡保留的首尾文本，三模型共享保留跨度。它使用大于Harness的输入预算，是独立质量对照，不是原模型完整阅读，也不是同计算量比较。

| 模型 | 计划题数 | Head/tail兼容解析正确 | Head/tail严格原始回复正确 | Harness同批完整来源正确 | Head/tail有可解析答案 |
|---|---:|---:|---:|---:|---:|
| gpt-6-sol | 2 | 2/2 (100.00%) | 2/2 (100.00%) | 2/2 (100.00%) | 2 |
| gpt-6-luna | 2 | 1/2 (50.00%) | 1/2 (50.00%) | 2/2 (100.00%) | 2 |
| gpt-6.1-terra | 2 | 2/2 (100.00%) | 2/2 (100.00%) | 2/2 (100.00%) | 2 |

主质量对照使用与主实验相同的本地兼容选项提取。严格列只对head/tail的实际模型回复去除星号后匹配两条区分大小写的“The correct answer is”模式；裸A不自动算严格格式合格。Harness主实验是控制器最终选项，不把该单字母再解析成所谓原始回复严格得分。两种解析口径分别列示。

| 来源index | 原文字符 | 保留字符／比例 | 保留source spans | 丢弃中段span | 三模型准备阶段准入input tokens |
|---|---:|---:|---|---|---|
| 143 | 16,182,936 | 2,997,835 / 18.52% | [[0, 1498918], [14684019, 16182936]] | [1498918, 14684019] | gpt-6-sol: 786,193, gpt-6-luna: 786,193, gpt-6.1-terra: 786,193 |
| 260 | 11,918,963 | 2,639,965 / 22.15% | [[0, 1319983], [10598981, 11918963]] | [1319983, 10598981] | gpt-6-sol: 746,258, gpt-6-luna: 746,258, gpt-6.1-terra: 746,258 |

没有插入新字符；保留首段与尾段直接拼接，中段明确丢弃。原文、截断文本及完整截断提示哈希见JSON。该实现参考上游head/tail策略，但按字符跨度裁剪并精确计数整个请求，不是上游按token取两半的逐字节复现。

| 来源／模型 | 参考选项 | 兼容预测 | 严格预测 | 实际状态 | 已知tokens | 未知预留tokens | 费用估计上界 |
|---|---|---|---|---|---:|---:|---:|
| 143 / gpt-6-sol | A | A | A | returned | 786,188 | 0 | USD 3.931050 |
| 143 / gpt-6-luna | A | A | A | returned | 786,188 | 0 | USD 0.196552 |
| 143 / gpt-6.1-terra | A | A | A | returned | 786,188 | 0 | USD 1.965525 |
| 260 / gpt-6-sol | C | C | C | returned | 746,253 | 0 | USD 3.731375 |
| 260 / gpt-6-luna | C | B | B | returned | 746,253 | 0 | USD 0.186569 |
| 260 / gpt-6.1-terra | C | C | C | returned | 746,253 | 0 | USD 1.865688 |

上游代码固定为[c5ea10bcd06285223c58dfed76bbc92d22273709](https://raw.githubusercontent.com/THUDM/LongBench/c5ea10bcd06285223c58dfed76bbc92d22273709/pred.py)；这是代码pin，不是数据集revision。仅1或2个来源、一次推理及丢失中段的条件限制了结论，不将附加结果拼进主12题成绩。

Head/tail freeze SHA-256: dfb25fdb5cd26067a881dca5747b78ded10c7a8e5389091eacb2529a26890c3f

## 完成后的轨迹与原始回复诊断

独立[费用与评分审计](../benchmarks/longbench_openai_score_audit_20261102.json)核对了各阶段预算、模型身份、已知／未知用量、费用及主评分；[轨迹审计](../benchmarks/longbench_openai_trajectory_audit_20261102.json)另核对来源绑定、请求哈希及可重放的控制器流程。它们均在最终评分后离线执行，没有新增模型或计数API调用。轨迹重放复用冻结的生产选择、解析器与控制器，不是独立的语义蕴含或参考答案真值核验。

| 范围 | 记录数 | 完整轨迹重放通过 | 轨迹不可确证 |
|---|---:|---:|---:|
| original + recovery + auxiliary history | 120 | 80 | 40 |
| main logical outcomes including capacity probes | 75 | 74 | 1 |
| auxiliary head/tail | 6 | 6 | 0 |

全部120条token／费用账本核验通过，但不能声称全部轨迹重放通过。40条不可确证轨迹包括保留的39条原始零生成计数故障，以及gpt-6-sol的Harness index251：该原始任务执行过4次生成，缺少一次计数故障的原始异常payload，因此其完整轨迹与选中原始回复无法确证。原始主结果仍为正确且保持不变；账本通过不替代缺失的轨迹证据。

严格格式诊断只作用于当时实际固定选中的完整原始回复，绝不按严格解析结果重选候选。共81份逻辑收据（含3条不评分探针）中，可追溯原始回复46份：35条Harness、5条原生Direct及6条首尾对照；46份均可由上游两条严格模式提取，且与原始控制器选项零分歧。缺少原始回复标记为不可用，不当作格式失败。下表是可追溯子集的事后敏感性分析，不替换主准确率，也不是官方榜单成绩。

| 模型／路径 | 计划题数 | 可追溯原始回复 | 严格格式接受 | 严格正确／可追溯子集 | 不可追溯 | 原始主结果正确／全部计划 |
|---|---:|---:|---:|---:|---:|---:|
| gpt-6-sol / direct | 12 | 1 | 1 | 0/1 (0.00%) | 11 | 0/12 (0.00%) |
| gpt-6-sol / harness | 12 | 11 | 11 | 7/11 (63.64%) | 1 | 8/12 (66.67%) |
| gpt-6-sol / headtail_control | 2 | 2 | 2 | 2/2 (100.00%) | 0 | 2/2 (100.00%) |
| gpt-6-luna / direct | 12 | 2 | 2 | 0/2 (0.00%) | 10 | 0/12 (0.00%) |
| gpt-6-luna / harness | 12 | 12 | 12 | 6/12 (50.00%) | 0 | 6/12 (50.00%) |
| gpt-6-luna / headtail_control | 2 | 2 | 2 | 1/2 (50.00%) | 0 | 1/2 (50.00%) |
| gpt-6.1-terra / direct | 12 | 2 | 2 | 1/2 (50.00%) | 10 | 1/12 (8.33%) |
| gpt-6.1-terra / harness | 12 | 12 | 12 | 7/12 (58.33%) | 0 | 7/12 (58.33%) |
| gpt-6.1-terra / headtail_control | 2 | 2 | 2 | 2/2 (100.00%) | 0 | 2/2 (100.00%) |

两种分母不能直接当作同一准确率比较。Sol Harness的7／11严格可追溯子集，与8／12主结果不同，是因为index251的原始回复不可追溯，不能据此宣称它格式失败。收据只公开短选项、原始回复哈希、来源哈希及选中位置，不分发模型正文。
