# LongMemEval-V2 attribution

LongMemEval-V2 is an external benchmark by Di Wu, Zixiang Ji, Asmi Kawatkar,
Bryan Kwan, Jia-Chen Gu, Nanyun Peng, and Kai-Wei Chang.
CRU-Harness does not claim authorship of the benchmark or its data.

- [Official repository](https://github.com/xiaowu0162/LongMemEval-V2)
- [Paper: LongMemEval-V2: Evaluating Long-Term Agent Memory Toward Experienced Colleagues](https://arxiv.org/abs/2605.12493)
- [Official dataset](https://huggingface.co/datasets/xiaowu0162/longmemeval-v2)
- Evaluation code commit: `2cc8c540bdb87fe6761629b585e727e1c4704520`.
- Dataset revision: `f152293e235517d504809563c833d7190b8c713b`.

The benchmark driver reads domain system prompts from the pinned upstream
`evaluation/harness.py` and loads scoring functions from the unmodified upstream
`evaluation/qa_eval_metrics.py` in the local workspace. The default upstream
LLM judge (`gpt-5.2`) is explicitly replaced by the evaluated Qwen endpoint;
deterministic scoring functions and judge prompts are preserved. Our local
20-question pilot is not an official LAFS or leaderboard result.

The upstream software is Apache License 2.0. Its unmodified license is included
as [LICENSE](LICENSE). No upstream source contexts, questions, answers, screenshot
pixels, or raw inference replies are redistributed in the derived report.
The upstream software license alone does not establish redistribution rights
for underlying source data. Obtain the dataset from its official source under
its applicable terms.

Project-authored additions are the opt-in trajectory/image adapter, orchestration,
auditors, synthetic tests, and derived numeric/hash metadata. See the
[full evaluation report](../../docs/longmemeval_v2_qwen_20261103.md).
