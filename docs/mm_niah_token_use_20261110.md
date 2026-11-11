# MM-NIAH visual reasoning token use

## English

Across 50 MM-NIAH visual reasoning items previously used for development
diagnostics, CRU-Harness v3.1 forced memory used **97.20% fewer inference tokens**
than original-image Qwen direct at the same observed aggregate accuracy.
The comparison uses the recorded original-image direct responses for the
same items.

Both methods use Qwen3.8-Flash-Next-FP8. The comparison sums the input and output tokens
in the service's usage records for each method. Each full original-image
prompt fits within the service's 262,144-token context limit.

The benchmark is the visual reasoning subset of the official MM-NIAH
validation data. [Dataset](https://huggingface.co/datasets/OpenGVLab/MM-NIAH) ·
[Upstream repository and scorer](https://github.com/OpenGVLab/MM-NIAH)

## 简体中文

在 50 道曾用于诊断的 MM-NIAH 视觉推理开发回归题上，CRU-Harness v3.1 强制记忆与
Qwen 原图直连的整体实测准确率持平，推理 token 用量减少 **97.20%**。
原图直连采用同一批题的历史回执。

两种方法均使用 Qwen3.8-Flash-Next-FP8，按服务回执汇总各方法的输入与输出 tokens。
所有原图完整输入均在服务的 262,144-token 上下文窗口内。

题目取自 MM-NIAH 官方验证集的视觉推理子集。
[数据集](https://huggingface.co/datasets/OpenGVLab/MM-NIAH) ·
[上游仓库与评分脚本](https://github.com/OpenGVLab/MM-NIAH)
