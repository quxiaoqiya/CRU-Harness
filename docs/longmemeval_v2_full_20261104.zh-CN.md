# LongMemEval-V2 Small 全量三路评测报告

本轮覆盖固定版本全部 451 道题，每题 100 条历史轨迹。此前已用 76 题、未用 375 题分开报告。
其中 29 道带截图，422 道无题图。V1 与 V3 的无图回退代码及参数完全相同，422 条结果明确共享；不是两次独立随机复现。

| 路径 | 全部451 | 未用375 | 已用76 | 视觉29 | 方法计量tokens | 方法生成次数 |
|---|---:|---:|---:|---:|---:|---:|
| 直连 Qwen（首尾截取） | 72/451（15.96%） | 59/375（15.73%） | 13/76（17.11%） | 8/29（27.59%） | 112,564,905 | 451 |
| 旧 V1 | 208/451（46.12%） | 172/375（45.87%） | 36/76（47.37%） | 11/29（37.93%） | 17,241,413 | 1537 |
| 新 V3 | 209/451（46.34%） | 173/375（46.13%） | 36/76（47.37%） | 12/29（41.38%） | 17,279,358 | 1534 |

方法计量表示该方法单独执行的用量，不能相加充当实际总调用；共享文本计算仅计费一次。
实际推理 2,093 次、130,747,706 tokens；裁判另计 256 次、330,210 tokens。

## 按能力分类

| 类型 | 题数 | 直连 | V1 | V3 |
|---|---:|---:|---:|---:|
| dynamic-environment | 86 | 17/86（19.77%） | 47/86（54.65%） | 47/86（54.65%） |
| dynamic-environment-abs | 41 | 2/41（4.88%） | 10/41（24.39%） | 10/41（24.39%） |
| errors-gotchas | 29 | 8/29（27.59%） | 11/29（37.93%） | 12/29（41.38%） |
| procedure | 74 | 19/74（25.68%） | 43/74（58.11%） | 43/74（58.11%） |
| procedure-abs | 32 | 0/32（0.00%） | 3/32（9.38%） | 3/32（9.38%） |
| static-environment | 134 | 24/134（17.91%） | 79/134（58.96%） | 79/134（58.96%） |
| static-environment-abs | 55 | 2/55（3.64%） | 15/55（27.27%） | 15/55（27.27%） |

`*-abs` 测的是识别并解释错误前提；仅输出 UNKNOWN 按上游规则记错，不能视为正确拒答。

## 成对得失

相对旧 V1：{'n': 451, 'both_correct': 205, 'new_gain': 4, 'new_loss': 3, 'both_wrong': 239}。
相对直连：{'n': 451, 'both_correct': 47, 'new_gain': 162, 'new_loss': 25, 'both_wrong': 217}。

## 逐题结果

| ID | 域 | 能力 | 已用/未用 | 直连 | V1 | V3 |
|---|---|---|---|---|---|---|
| 00aa905a | web | dynamic-environment | unused_question | 对 | 对 | 对 |
| 01307e07 | enterprise | dynamic-environment | unused_question | 错 | 对 | 对 |
| 01f6e679 | web | static-environment-abs | unused_question | 错 | 错 | 错 |
| 025db8ef | enterprise | procedure | previously_consumed | 错 | 对 | 对 |
| 0382ed92 | web | procedure-abs | unused_question | 错 | 错 | 错 |
| 0401f0c8 | web | dynamic-environment-abs | unused_question | 错 | 对 | 对 |
| 0574c69a | web | errors-gotchas | previously_consumed | 对 | 对 | 错 |
| 057a2d4d | enterprise | static-environment | unused_question | 错 | 对 | 对 |
| 059974dd | enterprise | procedure-abs | unused_question | 错 | 对 | 对 |
| 05cce9b3 | web | static-environment | unused_question | 错 | 对 | 对 |
| 06a5a25f | web | dynamic-environment | unused_question | 错 | 对 | 对 |
| 06e965cf | web | static-environment | unused_question | 对 | 对 | 对 |
| 0738c1ac | web | static-environment | previously_consumed | 错 | 对 | 对 |
| 07a0145f | web | errors-gotchas | previously_consumed | 错 | 对 | 对 |
| 07ab3723 | web | procedure | unused_question | 错 | 对 | 对 |
| 07b49858 | web | procedure | unused_question | 错 | 错 | 错 |
| 07d86be1 | web | dynamic-environment | unused_question | 错 | 错 | 错 |
| 07ffeedf | enterprise | procedure | unused_question | 错 | 错 | 错 |
| 07fff330 | web | procedure | unused_question | 错 | 错 | 错 |
| 080b0218 | enterprise | static-environment | unused_question | 错 | 对 | 对 |
| 0a13e290 | web | dynamic-environment-abs | unused_question | 错 | 错 | 错 |
| 0a4a47e6 | web | static-environment | unused_question | 对 | 错 | 错 |
| 0ace674b | web | static-environment-abs | unused_question | 错 | 错 | 错 |
| 0af83ccc | web | static-environment-abs | unused_question | 错 | 错 | 错 |
| 0b50ca0d | enterprise | procedure | previously_consumed | 错 | 对 | 对 |
| 0c0fdcd7 | web | static-environment | unused_question | 错 | 错 | 错 |
| 0c6b0c60 | web | static-environment | unused_question | 对 | 对 | 对 |
| 0cd6dc43 | web | dynamic-environment-abs | unused_question | 错 | 错 | 错 |
| 0cf979c4 | enterprise | static-environment | unused_question | 错 | 对 | 对 |
| 0d07143c | enterprise | dynamic-environment | unused_question | 错 | 错 | 错 |
| 0f8c85a6 | enterprise | static-environment | unused_question | 错 | 对 | 对 |
| 0f970f01 | enterprise | static-environment | unused_question | 错 | 对 | 对 |
| 0fe2c676 | web | procedure | unused_question | 对 | 错 | 错 |
| 10003186 | web | static-environment | unused_question | 错 | 错 | 错 |
| 100ff132 | enterprise | procedure | unused_question | 错 | 错 | 错 |
| 10466872 | web | static-environment | unused_question | 错 | 对 | 对 |
| 106c321b | enterprise | static-environment | unused_question | 对 | 错 | 错 |
| 109b334c | enterprise | dynamic-environment-abs | unused_question | 错 | 错 | 错 |
| 11cc7ac2 | enterprise | dynamic-environment-abs | unused_question | 错 | 错 | 错 |
| 11dac74b | web | dynamic-environment | previously_consumed | 错 | 错 | 错 |
| 12286151 | enterprise | dynamic-environment-abs | unused_question | 错 | 错 | 错 |
| 123b74fb | enterprise | static-environment-abs | unused_question | 错 | 错 | 错 |
| 12f8cfd2 | enterprise | procedure-abs | unused_question | 错 | 错 | 错 |
| 14a823df | enterprise | dynamic-environment | unused_question | 对 | 错 | 错 |
| 15e5efa6 | web | static-environment | unused_question | 错 | 对 | 对 |
| 17a03f9b | enterprise | static-environment | unused_question | 错 | 对 | 对 |
| 17d63ad8 | web | dynamic-environment | unused_question | 错 | 对 | 对 |
| 18b91103 | enterprise | errors-gotchas | previously_consumed | 错 | 错 | 错 |
| 18f9d965 | web | procedure | unused_question | 对 | 对 | 对 |
| 191184f3 | enterprise | dynamic-environment-abs | unused_question | 错 | 错 | 错 |
| 19367bc7 | enterprise | static-environment | unused_question | 错 | 对 | 对 |
| 196a68fb | web | dynamic-environment-abs | unused_question | 对 | 错 | 错 |
| 19935d2d | enterprise | static-environment-abs | previously_consumed | 错 | 对 | 对 |
| 19c641bd | web | procedure | unused_question | 对 | 对 | 对 |
| 1a89c3ac | web | dynamic-environment-abs | previously_consumed | 错 | 错 | 错 |
| 1b43f7df | web | procedure | unused_question | 错 | 错 | 错 |
| 1bfd2eac | enterprise | static-environment | unused_question | 错 | 错 | 错 |
| 1cf04706 | enterprise | procedure | unused_question | 错 | 对 | 对 |
| 1d864942 | web | errors-gotchas | previously_consumed | 错 | 错 | 错 |
| 1defc293 | web | dynamic-environment | unused_question | 对 | 错 | 错 |
| 1e011eb5 | web | procedure | unused_question | 错 | 对 | 对 |
| 1e77eb4b | web | errors-gotchas | previously_consumed | 错 | 错 | 对 |
| 1fec75e5 | web | procedure-abs | unused_question | 错 | 对 | 对 |
| 21df0107 | web | dynamic-environment-abs | unused_question | 错 | 错 | 错 |
| 21f3228c | web | static-environment-abs | unused_question | 错 | 错 | 错 |
| 229675cd | enterprise | static-environment | unused_question | 对 | 对 | 对 |
| 233f9f09 | enterprise | dynamic-environment | unused_question | 错 | 错 | 错 |
| 237d2cc5 | enterprise | static-environment | unused_question | 错 | 错 | 错 |
| 238c3621 | web | procedure | unused_question | 错 | 对 | 对 |
| 23aecb38 | web | procedure | unused_question | 错 | 对 | 对 |
| 23c3e290 | enterprise | static-environment | unused_question | 错 | 错 | 错 |
| 2408b115 | web | static-environment-abs | unused_question | 错 | 错 | 错 |
| 247bf724 | enterprise | static-environment-abs | unused_question | 错 | 对 | 对 |
| 25853d7f | web | static-environment-abs | unused_question | 错 | 对 | 对 |
| 25b00876 | enterprise | procedure | previously_consumed | 错 | 对 | 对 |
| 25dc1d7d | web | dynamic-environment-abs | unused_question | 错 | 对 | 对 |
| 26a038ce | web | static-environment | previously_consumed | 对 | 对 | 对 |
| 2721ca7f | enterprise | dynamic-environment | unused_question | 错 | 错 | 错 |
| 285db8ec | enterprise | errors-gotchas | previously_consumed | 错 | 错 | 错 |
| 289aea4a | web | procedure | unused_question | 错 | 错 | 错 |
| 2975030d | enterprise | static-environment | previously_consumed | 错 | 错 | 错 |
| 2ad054b5 | enterprise | static-environment-abs | unused_question | 错 | 错 | 错 |
| 2b1a8fc6 | web | procedure | unused_question | 对 | 对 | 对 |
| 2b96f402 | enterprise | static-environment | unused_question | 错 | 对 | 对 |
| 2c45ecbb | web | static-environment | unused_question | 错 | 对 | 对 |
| 2ca2d443 | web | procedure | unused_question | 对 | 错 | 错 |
| 2cc1b9ca | web | errors-gotchas | previously_consumed | 错 | 错 | 错 |
| 2d4e08b9 | enterprise | dynamic-environment | unused_question | 错 | 错 | 错 |
| 2e67d04f | web | static-environment | unused_question | 错 | 对 | 对 |
| 2e90de97 | enterprise | static-environment | unused_question | 错 | 对 | 对 |
| 2ee130d2 | web | dynamic-environment | unused_question | 对 | 对 | 对 |
| 2f6a21cd | enterprise | errors-gotchas | previously_consumed | 错 | 错 | 错 |
| 2fab8b79 | enterprise | dynamic-environment-abs | unused_question | 错 | 错 | 错 |
| 302bf60e | enterprise | errors-gotchas | previously_consumed | 对 | 对 | 对 |
| 30fab98c | web | procedure | unused_question | 对 | 对 | 对 |
| 31f146ba | web | procedure | unused_question | 错 | 错 | 错 |
| 3269a3c5 | web | procedure | unused_question | 错 | 对 | 对 |
| 32909aa1 | enterprise | procedure | unused_question | 错 | 对 | 对 |
| 329742b1 | enterprise | dynamic-environment-abs | unused_question | 错 | 错 | 错 |
| 32d04f31 | enterprise | dynamic-environment | unused_question | 错 | 对 | 对 |
| 32e719c2 | enterprise | static-environment | unused_question | 错 | 对 | 对 |
| 358c13ce | web | dynamic-environment-abs | unused_question | 错 | 错 | 错 |
| 36fe48ef | web | static-environment | previously_consumed | 错 | 错 | 错 |
| 38726e61 | enterprise | static-environment | unused_question | 错 | 错 | 错 |
| 3a131a42 | enterprise | static-environment | unused_question | 错 | 错 | 错 |
| 3a2e9368 | enterprise | procedure | unused_question | 错 | 错 | 错 |
| 3a7a7880 | enterprise | static-environment-abs | unused_question | 错 | 错 | 错 |
| 3a836b31 | enterprise | static-environment-abs | unused_question | 错 | 对 | 对 |
| 3b375e30 | enterprise | static-environment | unused_question | 错 | 错 | 错 |
| 3b8210a1 | web | procedure-abs | unused_question | 错 | 错 | 错 |
| 3c0c1b19 | web | procedure | unused_question | 对 | 错 | 错 |
| 3daef058 | enterprise | static-environment-abs | unused_question | 错 | 错 | 错 |
| 3f366e15 | web | static-environment | unused_question | 对 | 对 | 对 |
| 3fd81278 | enterprise | dynamic-environment | unused_question | 错 | 对 | 对 |
| 401566e6 | web | procedure | unused_question | 错 | 对 | 对 |
| 40758d8b | web | static-environment | unused_question | 对 | 对 | 对 |
| 4121647c | web | static-environment-abs | unused_question | 错 | 错 | 错 |
| 41a18e5c | enterprise | dynamic-environment | unused_question | 错 | 错 | 错 |
| 41bb40df | web | static-environment | unused_question | 对 | 对 | 对 |
| 4252cc1c | web | static-environment | unused_question | 错 | 错 | 错 |
| 4266cb67 | web | static-environment | unused_question | 错 | 错 | 错 |
| 42a55762 | enterprise | static-environment | unused_question | 错 | 对 | 对 |
| 42d13006 | web | errors-gotchas | previously_consumed | 对 | 对 | 对 |
| 4329b535 | web | dynamic-environment | unused_question | 错 | 错 | 错 |
| 436c58d5 | web | static-environment | unused_question | 错 | 对 | 对 |
| 43d9c5d0 | enterprise | dynamic-environment-abs | unused_question | 错 | 错 | 错 |
| 44c05e59 | web | static-environment | unused_question | 对 | 错 | 错 |
| 45d1e7f1 | enterprise | static-environment-abs | unused_question | 错 | 错 | 错 |
| 4755f316 | enterprise | static-environment | unused_question | 错 | 对 | 对 |
| 48a52262 | enterprise | errors-gotchas | previously_consumed | 错 | 对 | 错 |
| 48ab5c3d | web | static-environment | unused_question | 错 | 错 | 错 |
| 4964afed | enterprise | errors-gotchas | previously_consumed | 错 | 错 | 对 |
| 499488a6 | web | static-environment | unused_question | 对 | 对 | 对 |
| 4a4aad22 | web | procedure-abs | unused_question | 错 | 错 | 错 |
| 4b0c5275 | web | dynamic-environment-abs | unused_question | 错 | 错 | 错 |
| 4c6d1eb7 | enterprise | dynamic-environment | unused_question | 错 | 对 | 对 |
| 4ca9f7d7 | web | static-environment | unused_question | 对 | 对 | 对 |
| 4cb090a9 | web | procedure-abs | unused_question | 错 | 错 | 错 |
| 4d7dda89 | web | procedure | unused_question | 对 | 错 | 错 |
| 4df5e6b4 | enterprise | procedure | unused_question | 错 | 错 | 错 |
| 4dffe641 | enterprise | dynamic-environment | previously_consumed | 错 | 错 | 错 |
| 4ed01379 | enterprise | static-environment | unused_question | 错 | 错 | 错 |
| 4f92aa38 | web | procedure | unused_question | 错 | 错 | 错 |
| 50711c52 | enterprise | dynamic-environment | unused_question | 错 | 对 | 对 |
| 507a0fe8 | enterprise | procedure | unused_question | 错 | 错 | 错 |
| 50bfa714 | enterprise | dynamic-environment-abs | unused_question | 错 | 错 | 错 |
| 50d92f55 | enterprise | errors-gotchas | previously_consumed | 错 | 错 | 错 |
| 51764e11 | web | procedure-abs | unused_question | 错 | 错 | 错 |
| 521b124f | web | static-environment | unused_question | 对 | 对 | 对 |
| 52dd33bb | enterprise | procedure | unused_question | 错 | 错 | 错 |
| 5387d23d | web | dynamic-environment | unused_question | 错 | 对 | 对 |
| 549c0657 | web | static-environment-abs | unused_question | 错 | 对 | 对 |
| 571b841c | enterprise | dynamic-environment-abs | previously_consumed | 错 | 错 | 错 |
| 577ab4ee | web | static-environment-abs | unused_question | 错 | 错 | 错 |
| 57817b87 | web | static-environment-abs | unused_question | 错 | 错 | 错 |
| 579557d8 | enterprise | dynamic-environment-abs | unused_question | 错 | 对 | 对 |
| 57cd3560 | enterprise | dynamic-environment-abs | unused_question | 错 | 对 | 对 |
| 5849c18b | enterprise | static-environment-abs | previously_consumed | 错 | 错 | 错 |
| 58c34839 | enterprise | procedure | previously_consumed | 错 | 对 | 对 |
| 597645be | web | procedure | previously_consumed | 错 | 错 | 错 |
| 597ca614 | web | static-environment | unused_question | 错 | 对 | 对 |
| 5a85ca0f | web | dynamic-environment | previously_consumed | 错 | 错 | 错 |
| 5aaf0f61 | enterprise | static-environment | unused_question | 错 | 对 | 对 |
| 5b249b5c | enterprise | static-environment | unused_question | 错 | 对 | 对 |
| 5c826b28 | web | dynamic-environment | previously_consumed | 错 | 对 | 对 |
| 5ca112bf | enterprise | static-environment-abs | unused_question | 错 | 错 | 错 |
| 5cb7289b | web | dynamic-environment | unused_question | 错 | 对 | 对 |
| 5d6b993a | web | static-environment | unused_question | 错 | 对 | 对 |
| 5d6ecdeb | web | procedure-abs | unused_question | 错 | 错 | 错 |
| 5db4899c | web | static-environment | unused_question | 对 | 对 | 对 |
| 5edd2533 | enterprise | dynamic-environment | previously_consumed | 错 | 对 | 对 |
| 5f100f22 | enterprise | static-environment | unused_question | 错 | 错 | 错 |
| 609acb91 | web | dynamic-environment | unused_question | 错 | 对 | 对 |
| 60c7ca22 | web | procedure | unused_question | 对 | 对 | 对 |
| 60f2b293 | web | static-environment | unused_question | 对 | 对 | 对 |
| 616c238d | enterprise | procedure | unused_question | 错 | 对 | 对 |
| 626e401e | web | errors-gotchas | previously_consumed | 对 | 对 | 对 |
| 631667c9 | web | static-environment | unused_question | 错 | 对 | 对 |
| 634973c3 | enterprise | static-environment | unused_question | 错 | 错 | 错 |
| 63a1b94b | web | dynamic-environment | unused_question | 对 | 对 | 对 |
| 6517165d | web | static-environment-abs | unused_question | 错 | 错 | 错 |
| 658fa827 | enterprise | dynamic-environment | unused_question | 错 | 错 | 错 |
| 65fbcbdf | web | dynamic-environment | unused_question | 错 | 错 | 错 |
| 6652c337 | web | dynamic-environment | unused_question | 错 | 对 | 对 |
| 668f7b74 | enterprise | static-environment | unused_question | 错 | 错 | 错 |
| 66ed4c72 | web | static-environment-abs | unused_question | 错 | 错 | 错 |
| 674a6972 | web | procedure | unused_question | 错 | 对 | 对 |
| 679111c6 | enterprise | static-environment | unused_question | 错 | 错 | 错 |
| 68107cba | enterprise | static-environment | unused_question | 错 | 错 | 错 |
| 691cd1da | web | dynamic-environment | unused_question | 错 | 对 | 对 |
| 6a23675f | web | dynamic-environment | unused_question | 错 | 对 | 对 |
| 6a2e74de | web | static-environment | unused_question | 错 | 错 | 错 |
| 6a5bbcdb | web | dynamic-environment | unused_question | 错 | 错 | 错 |
| 6a73e32b | web | dynamic-environment | unused_question | 错 | 对 | 对 |
| 6a7808c5 | enterprise | static-environment | unused_question | 错 | 对 | 对 |
| 6a7f0d24 | web | dynamic-environment-abs | unused_question | 错 | 错 | 错 |
| 6c285a23 | enterprise | dynamic-environment-abs | unused_question | 错 | 错 | 错 |
| 6cb662d7 | web | static-environment | unused_question | 错 | 错 | 错 |
| 6cb8ce37 | enterprise | dynamic-environment | unused_question | 对 | 对 | 对 |
| 6d975a08 | web | static-environment | unused_question | 对 | 错 | 错 |
| 6e0759a2 | enterprise | dynamic-environment-abs | unused_question | 错 | 错 | 错 |
| 6f9cb5fc | web | dynamic-environment | unused_question | 对 | 错 | 错 |
| 6fc8c511 | web | static-environment-abs | unused_question | 错 | 错 | 错 |
| 6fdda2fc | enterprise | static-environment-abs | unused_question | 错 | 错 | 错 |
| 6feb33d4 | web | dynamic-environment-abs | unused_question | 对 | 错 | 错 |
| 70e79657 | web | static-environment | unused_question | 对 | 对 | 对 |
| 71026f57 | enterprise | static-environment-abs | unused_question | 错 | 错 | 错 |
| 7195f494 | web | dynamic-environment-abs | unused_question | 错 | 对 | 对 |
| 71e61e15 | enterprise | static-environment | unused_question | 错 | 对 | 对 |
| 73297783 | web | procedure | unused_question | 错 | 错 | 错 |
| 733699a5 | web | static-environment-abs | unused_question | 错 | 对 | 对 |
| 7351cbcc | enterprise | procedure-abs | unused_question | 错 | 错 | 错 |
| 7376c10d | web | dynamic-environment-abs | unused_question | 错 | 对 | 对 |
| 73b6738b | enterprise | static-environment | unused_question | 错 | 错 | 错 |
| 73cbefdc | enterprise | procedure | previously_consumed | 错 | 对 | 对 |
| 744dbdad | web | dynamic-environment | unused_question | 错 | 对 | 对 |
| 756a7434 | enterprise | procedure | unused_question | 错 | 对 | 对 |
| 75816b26 | enterprise | procedure | unused_question | 对 | 对 | 对 |
| 7586cf7c | enterprise | errors-gotchas | previously_consumed | 错 | 错 | 错 |
| 762d34b8 | web | static-environment | previously_consumed | 错 | 错 | 错 |
| 767e4106 | enterprise | procedure | unused_question | 错 | 对 | 对 |
| 77258cda | web | errors-gotchas | previously_consumed | 对 | 对 | 对 |
| 77db204d | web | static-environment | unused_question | 错 | 对 | 对 |
| 78686f4e | enterprise | dynamic-environment | unused_question | 错 | 对 | 对 |
| 7a07e9a0 | enterprise | static-environment | unused_question | 错 | 错 | 错 |
| 7bfbd5f1 | web | static-environment | unused_question | 错 | 对 | 对 |
| 7c8efee5 | web | static-environment | previously_consumed | 对 | 对 | 对 |
| 7df9e2ff | enterprise | procedure-abs | unused_question | 错 | 错 | 错 |
| 7e32e4a2 | enterprise | procedure | unused_question | 错 | 错 | 错 |
| 7ea13f14 | enterprise | dynamic-environment | unused_question | 错 | 错 | 错 |
| 7ebb0b95 | web | static-environment-abs | unused_question | 对 | 对 | 对 |
| 81180ae9 | web | errors-gotchas | previously_consumed | 错 | 错 | 错 |
| 820d56e2 | web | dynamic-environment | unused_question | 错 | 错 | 错 |
| 823d133a | enterprise | static-environment-abs | unused_question | 错 | 错 | 错 |
| 82cdfcc3 | enterprise | dynamic-environment | unused_question | 错 | 对 | 对 |
| 83a77f9d | web | dynamic-environment-abs | unused_question | 错 | 错 | 错 |
| 83c35034 | enterprise | static-environment | unused_question | 错 | 错 | 错 |
| 84238c8f | web | procedure-abs | unused_question | 错 | 错 | 错 |
| 85251f36 | enterprise | static-environment | unused_question | 错 | 对 | 对 |
| 85fc5722 | enterprise | dynamic-environment | unused_question | 错 | 错 | 错 |
| 866a0f2e | web | procedure-abs | unused_question | 错 | 错 | 错 |
| 86f00073 | web | static-environment-abs | unused_question | 错 | 错 | 错 |
| 86fa86eb | web | procedure | unused_question | 对 | 错 | 错 |
| 876c720a | enterprise | dynamic-environment | unused_question | 错 | 对 | 对 |
| 87711b62 | enterprise | dynamic-environment | unused_question | 错 | 错 | 错 |
| 882965b7 | enterprise | dynamic-environment | previously_consumed | 错 | 错 | 错 |
| 89439d62 | enterprise | static-environment | unused_question | 错 | 对 | 对 |
| 8955b988 | web | dynamic-environment | unused_question | 错 | 对 | 对 |
| 8957a127 | web | procedure | previously_consumed | 错 | 对 | 对 |
| 896017e0 | web | static-environment | unused_question | 对 | 错 | 错 |
| 89a20f48 | web | static-environment-abs | unused_question | 错 | 对 | 对 |
| 8a8f6aac | web | dynamic-environment-abs | unused_question | 错 | 错 | 错 |
| 8b09f378 | enterprise | static-environment | unused_question | 错 | 错 | 错 |
| 8b2a9dd3 | enterprise | static-environment | unused_question | 错 | 错 | 错 |
| 8bc6d989 | enterprise | procedure-abs | unused_question | 错 | 错 | 错 |
| 8c55d6bc | web | static-environment | unused_question | 错 | 对 | 对 |
| 8cb3280c | web | dynamic-environment | unused_question | 错 | 对 | 对 |
| 8d0400e5 | enterprise | static-environment-abs | unused_question | 错 | 错 | 错 |
| 8da5c407 | web | dynamic-environment-abs | unused_question | 错 | 对 | 对 |
| 8e21c6e5 | enterprise | errors-gotchas | previously_consumed | 对 | 错 | 对 |
| 8e320e42 | enterprise | static-environment | unused_question | 错 | 错 | 错 |
| 8f70bc0e | web | static-environment | unused_question | 错 | 对 | 对 |
| 8f752563 | enterprise | static-environment | unused_question | 错 | 错 | 错 |
| 901b7d17 | enterprise | dynamic-environment | unused_question | 错 | 错 | 错 |
| 90dc7e66 | web | dynamic-environment | unused_question | 对 | 对 | 对 |
| 90df8253 | enterprise | static-environment | unused_question | 错 | 对 | 对 |
| 910135bb | web | errors-gotchas | previously_consumed | 对 | 对 | 对 |
| 914ab1d4 | web | errors-gotchas | previously_consumed | 错 | 对 | 错 |
| 91d0775e | web | dynamic-environment | unused_question | 错 | 对 | 对 |
| 922a0c56 | web | static-environment-abs | previously_consumed | 错 | 对 | 对 |
| 92642fe7 | web | procedure | previously_consumed | 对 | 对 | 对 |
| 927c65a2 | enterprise | static-environment | unused_question | 对 | 错 | 错 |
| 93d0613b | enterprise | dynamic-environment-abs | unused_question | 错 | 错 | 错 |
| 95ae824f | enterprise | procedure | previously_consumed | 对 | 对 | 对 |
| 96497069 | web | dynamic-environment | previously_consumed | 错 | 错 | 错 |
| 9680f077 | enterprise | static-environment | previously_consumed | 错 | 对 | 对 |
| 97d5309a | web | procedure | unused_question | 错 | 对 | 对 |
| 97e1d977 | web | static-environment-abs | unused_question | 错 | 错 | 错 |
| 984022b1 | enterprise | dynamic-environment | unused_question | 对 | 错 | 错 |
| 98b62f3d | web | static-environment | unused_question | 对 | 对 | 对 |
| 98e6d369 | web | static-environment-abs | unused_question | 错 | 错 | 错 |
| 99ae54d1 | web | procedure | previously_consumed | 对 | 对 | 对 |
| 99b6e22f | web | procedure | unused_question | 对 | 对 | 对 |
| 99bc5d0d | enterprise | procedure-abs | unused_question | 错 | 对 | 对 |
| 99fa7c96 | web | procedure | previously_consumed | 错 | 错 | 错 |
| 9b34f55b | enterprise | procedure-abs | unused_question | 错 | 错 | 错 |
| 9c074855 | enterprise | static-environment | unused_question | 错 | 对 | 对 |
| 9c24f41e | enterprise | dynamic-environment | unused_question | 错 | 错 | 错 |
| 9c32bca6 | enterprise | static-environment | unused_question | 错 | 对 | 对 |
| 9cae869c | web | procedure | unused_question | 错 | 对 | 对 |
| 9e05978c | enterprise | procedure | unused_question | 错 | 对 | 对 |
| 9ed5a14d | web | procedure-abs | unused_question | 错 | 错 | 错 |
| 9ff0ffff | enterprise | dynamic-environment-abs | unused_question | 错 | 错 | 错 |
| a12b7094 | web | dynamic-environment | unused_question | 错 | 错 | 错 |
| a2673690 | web | dynamic-environment-abs | unused_question | 错 | 错 | 错 |
| a2a71197 | web | procedure-abs | previously_consumed | 错 | 错 | 错 |
| a363341d | web | procedure-abs | unused_question | 错 | 错 | 错 |
| a3a7b079 | enterprise | static-environment | unused_question | 错 | 错 | 错 |
| a42071db | web | static-environment | unused_question | 对 | 对 | 对 |
| a444cec8 | enterprise | static-environment | previously_consumed | 错 | 错 | 错 |
| a45edf3e | enterprise | static-environment | unused_question | 错 | 错 | 错 |
| a482c0d9 | enterprise | static-environment | unused_question | 错 | 对 | 对 |
| a58c5548 | enterprise | static-environment-abs | unused_question | 错 | 错 | 错 |
| a59d50f7 | web | procedure | unused_question | 错 | 对 | 对 |
| a62de1f1 | web | static-environment | unused_question | 错 | 错 | 错 |
| a667402d | enterprise | static-environment-abs | unused_question | 错 | 错 | 错 |
| a7da9ab3 | web | static-environment | unused_question | 错 | 对 | 对 |
| a7edfb7a | enterprise | static-environment | unused_question | 错 | 对 | 对 |
| a8959d04-abs | enterprise | static-environment-abs | unused_question | 错 | 错 | 错 |
| a92dff92 | web | procedure | unused_question | 对 | 对 | 对 |
| aa4a4b7b | enterprise | static-environment | unused_question | 错 | 对 | 对 |
| aa64a8bd | enterprise | dynamic-environment | unused_question | 错 | 对 | 对 |
| aa8d21f5 | web | static-environment-abs | previously_consumed | 错 | 错 | 错 |
| abadf765 | web | static-environment | unused_question | 错 | 错 | 错 |
| aca57021 | enterprise | procedure | unused_question | 错 | 错 | 错 |
| ad903be2 | enterprise | static-environment | unused_question | 错 | 对 | 对 |
| adddeaa5 | enterprise | static-environment | unused_question | 错 | 错 | 错 |
| aedd338d | enterprise | static-environment | unused_question | 错 | 对 | 对 |
| aef57896 | enterprise | static-environment | previously_consumed | 错 | 对 | 对 |
| af2ebaed | enterprise | errors-gotchas | previously_consumed | 错 | 错 | 错 |
| b05cf470 | web | static-environment-abs | unused_question | 错 | 对 | 对 |
| b15314c0 | enterprise | static-environment-abs | unused_question | 错 | 错 | 错 |
| b2595523 | web | dynamic-environment | unused_question | 错 | 对 | 对 |
| b339209c | web | errors-gotchas | previously_consumed | 错 | 对 | 对 |
| b4dcf650 | web | procedure | previously_consumed | 错 | 对 | 对 |
| b54161f8 | enterprise | dynamic-environment | unused_question | 错 | 对 | 对 |
| b5a25676 | web | dynamic-environment | unused_question | 对 | 错 | 错 |
| b7cc4da2 | enterprise | procedure | unused_question | 错 | 对 | 对 |
| b828a6b2 | web | dynamic-environment | previously_consumed | 错 | 错 | 错 |
| b82d0dd6 | enterprise | procedure-abs | unused_question | 错 | 错 | 错 |
| b8cabd09 | enterprise | dynamic-environment | unused_question | 错 | 对 | 对 |
| b8d56b89 | web | static-environment | previously_consumed | 错 | 对 | 对 |
| ba047686 | web | static-environment | unused_question | 错 | 对 | 对 |
| bb7ebd5c | web | procedure-abs | unused_question | 错 | 错 | 错 |
| bdb825a3 | enterprise | dynamic-environment | unused_question | 错 | 错 | 错 |
| bdf0e84b | web | dynamic-environment | unused_question | 错 | 错 | 错 |
| bdfef1b9 | enterprise | static-environment-abs | unused_question | 错 | 错 | 错 |
| be541aa6 | web | static-environment | unused_question | 对 | 对 | 对 |
| bf4f8b37 | enterprise | static-environment-abs | unused_question | 错 | 对 | 对 |
| bf98bb38 | enterprise | static-environment | unused_question | 错 | 对 | 对 |
| bfb3bcc4 | enterprise | procedure | unused_question | 错 | 错 | 错 |
| c0593d69 | enterprise | static-environment | unused_question | 错 | 错 | 错 |
| c0696888 | enterprise | dynamic-environment | previously_consumed | 错 | 错 | 错 |
| c1c000eb | enterprise | static-environment | unused_question | 错 | 对 | 对 |
| c216174c | web | procedure-abs | unused_question | 错 | 错 | 错 |
| c35bfeed | enterprise | errors-gotchas | previously_consumed | 错 | 错 | 错 |
| c36b0d68 | web | dynamic-environment | unused_question | 对 | 对 | 对 |
| c39f6edf | web | static-environment | unused_question | 错 | 对 | 对 |
| c3dbec17 | enterprise | static-environment | unused_question | 错 | 对 | 对 |
| c4c0e117 | enterprise | dynamic-environment-abs | unused_question | 错 | 错 | 错 |
| c56e69ca | web | static-environment | unused_question | 错 | 错 | 错 |
| c5dfa558 | web | procedure-abs | unused_question | 错 | 错 | 错 |
| c60f4681 | enterprise | procedure | unused_question | 错 | 错 | 错 |
| c6124506 | web | procedure | unused_question | 错 | 对 | 对 |
| c738b934 | web | dynamic-environment | unused_question | 错 | 错 | 错 |
| c7e367e2 | web | static-environment | unused_question | 对 | 对 | 对 |
| c80ca967 | enterprise | static-environment | unused_question | 错 | 对 | 对 |
| c843c4ec | web | dynamic-environment | unused_question | 错 | 错 | 错 |
| c9568c9b | enterprise | procedure | unused_question | 错 | 对 | 对 |
| c9c6a06a | enterprise | static-environment | unused_question | 错 | 错 | 错 |
| ca8e070b | enterprise | static-environment | unused_question | 错 | 错 | 错 |
| cbcd74ae | enterprise | static-environment-abs | unused_question | 错 | 错 | 错 |
| ccf29914 | enterprise | procedure | unused_question | 错 | 对 | 对 |
| cdb47c5d | web | dynamic-environment-abs | previously_consumed | 错 | 错 | 错 |
| cdc46516 | web | static-environment-abs | unused_question | 错 | 对 | 对 |
| cdd0e687 | enterprise | procedure | unused_question | 错 | 错 | 错 |
| ce1cdb2d | enterprise | dynamic-environment-abs | unused_question | 错 | 错 | 错 |
| ce9aa351 | enterprise | errors-gotchas | previously_consumed | 错 | 错 | 错 |
| cf62efad | enterprise | static-environment | unused_question | 错 | 错 | 错 |
| d0d61088 | enterprise | dynamic-environment | previously_consumed | 错 | 错 | 错 |
| d0e556eb | enterprise | static-environment-abs | unused_question | 错 | 错 | 错 |
| d16a18fe | web | procedure | unused_question | 错 | 错 | 错 |
| d1d8cf9f | web | static-environment | unused_question | 错 | 错 | 错 |
| d1f82e80 | enterprise | procedure-abs | unused_question | 错 | 错 | 错 |
| d3300354 | web | dynamic-environment | unused_question | 错 | 错 | 错 |
| d3d7ec75 | enterprise | static-environment | unused_question | 错 | 错 | 错 |
| d47ca5c8 | web | dynamic-environment-abs | unused_question | 错 | 错 | 错 |
| d5b9fda4 | web | static-environment | unused_question | 错 | 对 | 对 |
| d63e8a7f | enterprise | procedure | unused_question | 错 | 对 | 对 |
| d6da3938 | web | static-environment | unused_question | 错 | 对 | 对 |
| d72fb64d | enterprise | procedure-abs | unused_question | 错 | 错 | 错 |
| d7c94023 | web | dynamic-environment-abs | unused_question | 错 | 对 | 对 |
| d83c4bf7 | enterprise | dynamic-environment | unused_question | 对 | 错 | 错 |
| d886b018 | web | procedure | unused_question | 对 | 错 | 错 |
| dae9f7e9 | web | dynamic-environment | unused_question | 错 | 错 | 错 |
| dbe1a814 | enterprise | dynamic-environment-abs | previously_consumed | 错 | 错 | 错 |
| dc41b134 | enterprise | procedure-abs | previously_consumed | 错 | 错 | 错 |
| dd53ab77 | enterprise | dynamic-environment | unused_question | 错 | 对 | 对 |
| de0abc47 | enterprise | dynamic-environment-abs | unused_question | 错 | 错 | 错 |
| dea446d0 | web | dynamic-environment | unused_question | 对 | 对 | 对 |
| deb008ed | web | dynamic-environment | unused_question | 错 | 对 | 对 |
| e033e796 | enterprise | dynamic-environment | unused_question | 错 | 对 | 对 |
| e334d5c6 | enterprise | procedure | unused_question | 错 | 对 | 对 |
| e3a1d7f3 | web | static-environment | unused_question | 错 | 对 | 对 |
| e4d36b58 | web | static-environment | unused_question | 错 | 对 | 对 |
| e532c01e | web | static-environment-abs | previously_consumed | 错 | 错 | 错 |
| e71f7c92 | web | dynamic-environment | unused_question | 对 | 对 | 对 |
| e7292dab | enterprise | static-environment | unused_question | 错 | 错 | 错 |
| e77d1287 | web | static-environment | unused_question | 错 | 对 | 对 |
| e823a94a | enterprise | static-environment | unused_question | 错 | 错 | 错 |
| e83245c5 | web | dynamic-environment | unused_question | 对 | 对 | 对 |
| e9bb843a | web | dynamic-environment | unused_question | 错 | 对 | 对 |
| e9fde9fe | enterprise | static-environment | unused_question | 错 | 错 | 错 |
| ea361b20 | web | errors-gotchas | previously_consumed | 错 | 对 | 对 |
| eab07e81 | web | dynamic-environment | unused_question | 错 | 对 | 对 |
| eaba5c44 | web | static-environment-abs | unused_question | 错 | 错 | 错 |
| eb3cfd03 | enterprise | procedure | unused_question | 错 | 错 | 错 |
| eb8cf28b | enterprise | static-environment-abs | unused_question | 错 | 错 | 错 |
| ebad58ed | enterprise | static-environment-abs | previously_consumed | 错 | 对 | 对 |
| ecca89b2 | web | procedure-abs | unused_question | 错 | 错 | 错 |
| ecd9dda3 | web | procedure-abs | unused_question | 错 | 错 | 错 |
| ecdb7d5e | enterprise | procedure | unused_question | 错 | 错 | 错 |
| edb69441 | web | dynamic-environment | unused_question | 错 | 对 | 对 |
| edc22f24 | web | procedure-abs | unused_question | 错 | 错 | 错 |
| edea0219 | web | dynamic-environment | unused_question | 错 | 对 | 对 |
| ee5252a3 | enterprise | dynamic-environment-abs | unused_question | 错 | 对 | 对 |
| ee68431d | web | dynamic-environment | unused_question | 对 | 对 | 对 |
| ee75b921 | web | static-environment-abs | unused_question | 错 | 对 | 对 |
| eec85674 | web | static-environment-abs | unused_question | 错 | 错 | 错 |
| eef545a0 | web | procedure | unused_question | 错 | 对 | 对 |
| eff00409 | web | static-environment-abs | unused_question | 错 | 对 | 对 |
| f02bec54 | enterprise | procedure-abs | unused_question | 错 | 错 | 错 |
| f078f429 | web | procedure-abs | unused_question | 错 | 错 | 错 |
| f0882f6e | enterprise | static-environment | unused_question | 错 | 错 | 错 |
| f13eb38d | web | procedure | unused_question | 错 | 错 | 错 |
| f1d8a168 | enterprise | static-environment | unused_question | 错 | 对 | 对 |
| f1f9de44 | enterprise | static-environment | previously_consumed | 错 | 对 | 对 |
| f23807ed | web | procedure | unused_question | 错 | 对 | 对 |
| f2b221fd | web | errors-gotchas | previously_consumed | 对 | 错 | 错 |
| f3496ca5 | enterprise | static-environment | unused_question | 错 | 错 | 错 |
| f35bac4a | enterprise | dynamic-environment | previously_consumed | 错 | 对 | 对 |
| f43926d9 | web | static-environment-abs | unused_question | 对 | 错 | 错 |
| f4b2b1fb | enterprise | static-environment | previously_consumed | 错 | 对 | 对 |
| f61a096f | web | static-environment | unused_question | 错 | 对 | 对 |
| f90f2255 | web | static-environment-abs | unused_question | 错 | 错 | 错 |
| f928c075 | web | static-environment | unused_question | 错 | 错 | 错 |
| f93031d2 | web | dynamic-environment | unused_question | 错 | 错 | 错 |
| f9c65fe6 | enterprise | procedure | unused_question | 对 | 错 | 错 |
| f9f9cd61 | web | dynamic-environment | unused_question | 对 | 错 | 错 |
| fa504f5e | enterprise | errors-gotchas | previously_consumed | 错 | 错 | 错 |
| fb56a896 | enterprise | errors-gotchas | previously_consumed | 错 | 错 | 错 |
| fc0dbc35 | web | static-environment-abs | unused_question | 错 | 错 | 错 |
| fc7d35d5 | web | procedure-abs | unused_question | 错 | 错 | 错 |
| fd006924 | web | dynamic-environment-abs | unused_question | 错 | 对 | 对 |
| fdeddf71 | web | errors-gotchas | unused_question | 错 | 错 | 对 |
| fe18a443 | web | dynamic-environment | previously_consumed | 错 | 对 | 对 |
| fe41b553 | enterprise | static-environment | unused_question | 错 | 对 | 对 |
| fec652a9 | web | procedure | unused_question | 对 | 对 | 对 |
| ff311d07 | enterprise | static-environment | unused_question | 错 | 错 | 错 |
| ff8f3a22 | web | static-environment | unused_question | 错 | 对 | 对 |
| ffa4e57e | web | dynamic-environment | unused_question | 错 | 对 | 对 |

## 边界与审计

- 直连上限 262,144 tokens／1次生成，采用原文首尾截取；不是完整历史直连。Harness 单次 16,384、每题累计 65,536、最多8次生成；输出均最多4096。
- 模型 Qwen3.8-Flash-Next-FP8，temperature0、seed20261103、thinking关闭；同模型不同预算配置。索引CPU和下载成本不折算成模型tokens。
- 数据来源 [官方代码](https://github.com/xiaowu0162/LongMemEval-V2/tree/2cc8c540bdb87fe6761629b585e727e1c4704520) 与 [固定数据版本](https://huggingface.co/datasets/xiaowu0162/longmemeval-v2/tree/f152293e235517d504809563c833d7190b8c713b)。
- 复用两份历史；未用问题不代表来源或模板独立。视觉29题中28题此前已用，不能把视觉全量当作未见验证。
- 沿用官方匹配／LLM评分规则，Qwen替代默认裁判。不是官方榜单提交或LAFS分数；得分不证明所有附加解释都有证据。
- 失败记录、UNKNOWN和未解析裁判不从计划分母剔除；裁判失败另列为待定，未静默重试不确定推理。
- 离线审计状态：`pass`。独立任务重放通过 931/931；无法重放 0；共享结果422。审计没有调用服务或tokenizer。
- 冻结SHA：`05089494bc500451268c6ca47bf3b84ea453adff0c78bc60907f73c0495c5735`。本地Mac编排落盘，未修改远端文件。

[完整脱敏JSON](../benchmarks/longmemeval_v2_full_20261104.json) · [1353条CSV](../benchmarks/longmemeval_v2_full_20261104.csv)

私有原始输出、标签、账本和来源保存在 `work/longmemeval_v2_full_qwen_20261104/`。
