# LongMemEval-V2：视觉召回与拒答优化后的问题留出对照

本轮两项新方案均未达到替换旧版的标准，默认路径保持不变。新 20 题：保守路由 Harness **11/20**，全量门控候选 **5/20**，旧版 **13/20**。六路均保留。

新题与首轮 20 题的 ID、规范化正文哈希均无重叠；web／enterprise 各 10 题，五种能力各 4 题，4 题带题图。历史仍是同两份 100 轨迹材料，共 200 轨迹、5095 原图。是问题留出试评，不是来源独立留出、451 题全量或官方 LAFS／排行榜成绩。

| 路径 | 答对／20 | UNKNOWN | 作答覆盖 | 已作答准确率 | 错误作答 | 输入＋输出 tokens |
|---|---:|---:|---:|---:|---:|---:|
| direct_full | capacity denied | 0 | 0% | — | 0 | 0 |
| direct_headtail_native | 5/20 (25%) | 6 | 70% | 35.7% | 9 | 5,048,130 |
| direct_headtail_native_concise | 3/20 (15%) | 6 | 70% | 21.4% | 11 | 5,047,168 |
| harness | 13/20 (65%) | 1 | 95% | 68.4% | 6 | 610,516 |
| harness_v2 | 5/20 (25%) | 13 | 35% | 71.4% | 2 | 1,084,553 |
| harness_visual_route | 11/20 (55%) | 4 | 80% | 68.8% | 5 | 753,210 |

完整直连的容量拒绝在生成前发生，不能作为语义答错。简短输出直连组仅增加简短 boxed 答案要求，其余来源首尾范围、选图与原生预算不变，用来检查格式差异是否解释收益。新版默认路径仍未替换旧版，避免把开发中的拒答策略自动推广到原能力。

**全量严格门控未提升总体准确率：它以更多拒答牺牲任务交付，不能作为默认升级。保守路由单独列出，不混同这项失败试验。**

## 预登记的保守视觉路由

扩展方案在任何留出标签访问前另行固定：16 道无题图问题直接执行旧控制器，4 道有题图问题启用感知匹配与证据门控。20 道均为新调用，没有答案复用。图像特征在独立 Pillow 11.3.0 环境内本地计算，原推理环境依赖未改；特征为双向 dHash 和 8×8 RGB 布局，属于相似度启发式，不是语义图像嵌入或已校准的召回概率。推理仍发送原始 PNG。

无题图分支首个请求提示哈希一致 16/16；回复全文哈希一致 0/16，解析后答案哈希一致 11/16；新／旧答对 10/16、10/16。全部 120 条输出早于任何一轮评分标签访问：True。扩展 freeze：`e2252bbfa80e166e8947c275cdea45b6cafba578a312c593952dccb4624f8310`。

无题图分支是同一旧算法的再次执行，底层服务仍可能产生输出差异；该部分的变化不能归因为新视觉检索。两组实验部分并行，计时仅作记录，不作为方法延迟优势的证据。

## 评分和视觉子集

| Subset | direct_headtail_native | direct_headtail_native_concise | harness | harness_v2 | harness_visual_route |
|---|---:|---:|---:|---:|---:|
| Deterministic | 2/12 | 2/12 | 9/12 | 2/12 | 10/12 |
| Qwen judged | 3/8 | 1/8 | 4/8 | 3/8 | 1/8 |
| Query image | 3/4 | 1/4 | 3/4 | 2/4 | 1/4 |

确定性评分函数沿用固定上游版本；裁判题使用原官方提示，将默认 gpt-5.2 换为 Qwen，可能存在同模型裁判偏差。拒答记为任务不正确，并另列覆盖率及已作答准确率；不能用拒答降低分母来包装整体成绩。

5099 张原图的数值特征已重新计算并与缓存逐项比较：pass，未调用模型；该检查不评价视觉证据的语义相关性。

## 已确认的问题

有题图的 4 题中，保守路由把旧版 1 道错题改为正确，却把旧版答对的 3 道变为拒答。全量门控的已作答准确率为 71.4%，仅略高于旧版 68.4%，但覆盖率从 95% 降到 35%。因此不能以少答题换来的少量精度改善宣称整体能力提升。

实现中还存在已知原因：只用原来的错题开发门控，缺少正确题作为拒答校准对照；引用格式校验会把某些可恢复的候选挡住；同一状态的 seen 标记可能阻止另一字段窗口的追检。这些机制需要后续独立验证。当前不再根据本轮留出结果改参数并沿用本轮分数。

## 技术改动与边界

1. 状态级观察索引与原始块索引融合，结合历史轨迹目录路由；稀有词加权选择原字段片段，避免仅凭长文本重叠选图。
2. 原图绑定轨迹、状态、字段范围及 SHA；动态问题同时检索相邻状态，操作字段明确表示到达当前状态之前的动作。
3. 跨轮保留可核对的部分支持，最终重附原文与原图；公开遗漏计数，不能将引用存在等同于语义成立。
4. 最终门控区分支持、前提已证伪、证据不足；按要求检查引用、原文短引、物理请求中的图片，以及缺失条件和冲突。
5. 最终短答案由程序封装；机械引用错误可在原任务预算内修复一次，不能靠重发获取更有利答案。

这些是机械约束和模型判定的组合，不是语义证明。置信度为模型自报，阈值 0.8 未经独立校准；负例识别仍可能出错。历史索引包含的旧 agent 思路仅能路由检索，不作为事实证据。

三条 Harness 路径共用单次 16,384、累计 65,536 tokens、最多 8 次生成、单次输出上限 1024；新路径每次最多 6 张历史原图，旧路径最多 3 张，题图均保留。原生直连单次及累计 262,144 tokens、1 次生成、1024 输出，首尾组最多 16 张图。因此原生与 Harness 计算预算不相等，图片暴露量也不同，不是独立消融。

当前仍有明确边界：同一状态一旦标记 seen，不能在另一轮换取其不同字段片段；字段窗口和最终六张图也可能遗漏信息。尚无逐题人工 oracle 证据标注，不能把图片绑定通过或视觉子集答对率称为证据召回率提升。

## 开发轨迹与原能力保护

- longmemeval_v2_visual_dev_20261103: 0/12 correct, 11/12 UNKNOWN; 636,318 tokens; freeze `c30910f45b5ef555937073af92d75f9765e884ff81733d1b97a1f2d96789dbd3`.
- longmemeval_v2_visual_dev2_20261103: 1/12 correct, 10/12 UNKNOWN; 601,502 tokens; freeze `1731fb9c8a3e8c8ba0c6517f94341e750d7050286394c13de0fb8a3688772947`.
- longmemeval_v2_visual_dev3_20261103: 1/12 correct, 11/12 UNKNOWN; 653,464 tokens; freeze `fb7b23847dd3f8a5484cf60fc93aa65b80793c1d45ba1724c23de53e846e7ca1`.

开发阶段均使用原先 12 道错题，不能当新留出成绩；初版过度拒答的失败记录没有删除。236 项检查通过，原文本／事实控制器与首次 20 题评测的 49 份冻结源码未改。整个任务没有更改原环境依赖、远端文件或 GitHub 仓库。

[12-case analysis](longmemeval_v2_visual_error_analysis_20261103.zh-CN.md)

评分后确定性题诊断：有 4 个最终候选与标准答案匹配却被门控拒绝；有 5 个不匹配候选被拒绝。这个诊断不证明被拒候选具有充分证据，也不作为再次调优后沿用本轮留出成绩的依据。

## 逐题对照

| ID | Domain | Ability | Image | direct_full | direct_headtail_native | direct_headtail_native_concise | harness | harness_v2 | harness_visual_route |
|---|---|---|---|---|---|---|---|---|---|
| 4dffe641 | enterprise | dynamic | False | capacity_denied | UNKNOWN | incorrect | incorrect | UNKNOWN | incorrect |
| f35bac4a | enterprise | dynamic | False | capacity_denied | incorrect | incorrect | correct | UNKNOWN | correct |
| af2ebaed | enterprise | gotchas | True | capacity_denied | correct | correct | correct | correct | UNKNOWN |
| 48a52262 | enterprise | gotchas | True | capacity_denied | correct | incorrect | correct | UNKNOWN | UNKNOWN |
| dbe1a814 | enterprise | premise_awareness | False | capacity_denied | UNKNOWN | UNKNOWN | incorrect | incorrect | incorrect |
| dc41b134 | enterprise | premise_awareness | False | capacity_denied | UNKNOWN | UNKNOWN | incorrect | UNKNOWN | incorrect |
| 58c34839 | enterprise | procedure | False | capacity_denied | incorrect | UNKNOWN | incorrect | UNKNOWN | correct |
| 0b50ca0d | enterprise | procedure | False | capacity_denied | incorrect | incorrect | correct | UNKNOWN | correct |
| f4b2b1fb | enterprise | static | False | capacity_denied | UNKNOWN | UNKNOWN | correct | UNKNOWN | correct |
| 9680f077 | enterprise | static | False | capacity_denied | UNKNOWN | UNKNOWN | incorrect | UNKNOWN | incorrect |
| 96497069 | web | dynamic | False | capacity_denied | correct | correct | correct | UNKNOWN | correct |
| 5c826b28 | web | dynamic | False | capacity_denied | UNKNOWN | UNKNOWN | correct | UNKNOWN | correct |
| 910135bb | web | gotchas | True | capacity_denied | correct | incorrect | correct | UNKNOWN | UNKNOWN |
| 07a0145f | web | gotchas | True | capacity_denied | incorrect | incorrect | incorrect | correct | correct |
| e532c01e | web | premise_awareness | False | capacity_denied | incorrect | incorrect | UNKNOWN | UNKNOWN | UNKNOWN |
| aa8d21f5 | web | premise_awareness | False | capacity_denied | incorrect | incorrect | correct | correct | incorrect |
| 8957a127 | web | procedure | False | capacity_denied | incorrect | incorrect | correct | incorrect | correct |
| 92642fe7 | web | procedure | False | capacity_denied | correct | correct | correct | UNKNOWN | correct |
| 0738c1ac | web | static | False | capacity_denied | incorrect | incorrect | correct | correct | correct |
| b8d56b89 | web | static | False | capacity_denied | incorrect | incorrect | correct | correct | correct |

## 来源、审计与复核

[Official repository](https://github.com/xiaowu0162/LongMemEval-V2) · [Paper](https://arxiv.org/abs/2605.12493) · [Official data](https://huggingface.co/datasets/xiaowu0162/longmemeval-v2)

Data `f152293e235517d504809563c833d7190b8c713b`; code `2cc8c540bdb87fe6761629b585e727e1c4704520`; freeze `8a9b321daee247c0429b5a4065f76b7bc70f040a970d033844aa1836073a5811`.

离线审计：120/120 账本、120/120 评分重算、120/120 控制流程重放，状态 pass。重放不调用服务端；使用保存的计数、回复及冻结生产逻辑，不构成独立人工语义验证。

```bash
python3 -S scripts/verify_longmemeval_v2_visual_results.py
```

最长来源仍为 34,235,595 个规范化 JSON 文本 Qwen tokens，不含像素；本轮没有扩大来源历史规模。端到端时间包含计数及推理，建索引另存，不等于官方查询延迟。公开资料不含原题、标准答案、原图、回复全文或凭证；完整本地审计需保留工作目录。

[Results JSON](../benchmarks/longmemeval_v2_visual_holdout_20261103.json) · [Primary audit](../benchmarks/longmemeval_v2_visual_holdout_20261103_audit.json) · [Route audit](../benchmarks/longmemeval_v2_visual_holdout_20261103_route_audit.json)
