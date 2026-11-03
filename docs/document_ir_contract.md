# DocumentIR 合同

`independent_ingestion.parse(source)` 将路径、bytes 或文本解析为稳定、可序列化的 `DocumentIR`。IR 字段包含 `document_id`、`source_sha256`、`source_type`、`parser_version`、`pages`、`sections`、`blocks`、`tables`、`figures`、`equations`、`reading_order`、`source_spans`、`asset_refs`、`references` 和 `provenance`。

每个 section/block/table/figure 使用 `stable_id(document_hash, kind, ordinal, local_span)` 生成。每个对象保留页码、字符/ token 偏移、所属章节、原始文件 hash、解析器版本和图表引用。相同输入、相同 parser version 和相同字节序列得到完全一致的 ID；修改源文件会改变 `source_sha256` 和所有相关身份。

解析器通过 `ParserRegistry` 注册，以 source type 和扩展名选择。TXT、Markdown、HTML、XML、LaTeX 和图片目录使用标准库/轻量可选依赖；PDF 优先使用 `pypdf`，缺失时产生带诊断的文本 fallback。图片目录只生成资产 hash 和 figure 记录，不把原始图片字节写入 IR。`DocumentIR.to_dict()/from_dict()` 用于 checkpoint，不能携带外部服务对象。
