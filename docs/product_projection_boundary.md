# Agent Memory 与产品投影边界

Document Memory 经过 `scope_pruning → evidence_validation → unit_validation → record_grain_validation → duplicate_validation → product_projection` 后才可交给现有发布管线。

Chunk/Section/Document Memory、原始模型响应、思考过程、大段摘要、未解决冲突、阅读计划、未验证锚点和重复候选永远不进入 DuckDB 或 Neo4j。最终投影只含成熟实体、实验记录、规范化值、必要条件、必要证据和已验证关系。每个投影带同一 manifest digest，DuckDB current 与 Neo4j projection 必须使用同一 manifest；未完成或门禁失败时返回空投影并保留诊断。
