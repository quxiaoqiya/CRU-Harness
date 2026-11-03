# 模型能力网关

任务只声明 `planner`、`text_reader`、`vision_reader`、`reconciler`、`verifier` 或 `summarizer`。`ModelCapability` 记录 context window、vision、structured output、reasoning、最大输出、并发和 `capability_digest`。Provider 通过运行时注册注入，核心逻辑不包含具体厂商或模型分支。

内置 `MockProvider` 可生成 deterministic JSON；`deterministic-only` 模式拒绝真实网络调用。结构化响应先按 schema 校验，再转换为 MemoryPoint。缓存键绑定 content hash、prompt hash、schema version、capability digest 和 behavior hash；更换模型/能力会自然失效缓存，不改变 document_id/entity_id。
