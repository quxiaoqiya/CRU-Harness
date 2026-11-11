# MM-NIAH visual reasoning token use

## English

| MM-NIAH visual reasoning subset | Method compared with original-image Qwen direct | Fewer inference tokens at the same observed aggregate accuracy |
|---|---|---:|
| 50 development regression items previously used for diagnostics | CRU-Harness v3.1 forced memory | **97.20%** |
| 47 held-out items with complete paired results | Qwen direct with all source images capped at 512 px on the longest side | **32.43%** |

The development comparison uses the recorded original-image direct responses
for those same items. The held-out comparison uses the common set of items
with confirmed responses across the evaluation. On that set, the accuracy
match was observed between the two direct runs.

Both comparisons use Qwen3.8-Flash-Next-FP8 and sum the input and output tokens
in the service's usage records for each method. Each full original-image
prompt fits within the service's 262,144-token context limit.

The benchmark is the visual reasoning subset of the official MM-NIAH
validation data. [Dataset](https://huggingface.co/datasets/OpenGVLab/MM-NIAH) ·
[Upstream repository and scorer](https://github.com/OpenGVLab/MM-NIAH)

## 简体中文

| MM-NIAH 视觉推理子集 | 与 Qwen 原图直连比较的方法 | 整体实测准确率持平时的推理 token 降幅 |
|---|---|---:|
| 50 道曾用于诊断的开发回归题 | CRU-Harness v3.1 强制记忆 | **97.20%** |
| 47 道结果完整的留出配对题 | Qwen 直连，所有来源图的长边限制为 512 像素 | **32.43%** |

开发回归组沿用同一批题的原图直连历史回执。留出组采用各方法均有确认回执的共同题目，
实测准确率持平的结果来自两种直连方式。

两项对照均使用 Qwen3.8-Flash-Next-FP8，按服务回执汇总各方法的输入与输出 tokens。
所有原图完整输入均在服务的 262,144-token 上下文窗口内。

题目取自 MM-NIAH 官方验证集的视觉推理子集。
[数据集](https://huggingface.co/datasets/OpenGVLab/MM-NIAH) ·
[上游仓库与评分脚本](https://github.com/OpenGVLab/MM-NIAH)
