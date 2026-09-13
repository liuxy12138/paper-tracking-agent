# Industry Competitive Research Agent：项目深挖面试问答

> 使用方式：先掌握“90 秒项目介绍”和每题第一段。面试官追问时，再展开代码参数、数据流和边界。不要把“可以优化”说成“已经实现”。

## 一、90 秒项目介绍

我做的是一个面向行业和竞品研究的 Agentic RAG 系统。它解决的问题是：行业报告、竞品白皮书、产品资料和公告分散且篇幅长，普通大模型容易遗漏关键维度、混淆不同产品，结论也缺少证据。

系统的离线链路是：用 Docling 解析 PDF，失败时降级到 PyPDF；把不同文档标题归一为 overview、features、pricing、technology 等竞品研究章节；按 800 字符、120 overlap 切块；使用 BGE-M3 生成归一化 Dense 向量，并写入 Milvus。Milvus 同时根据原始 content 生成 BM25 稀疏向量。

在线链路由 LangGraph 编排：Planner 读取长期记忆并制定查询计划，Retrieval 做查询改写和工具调用，检索阶段分别执行 Dense 和 BM25 召回，在应用层按 0.75/0.25 融合，再用 BGE Reranker 重排；Analysis 先做证据分析，Summary 生成带来源标题引用的回答，Reflection 用规则分与 LLM 分检查证据数、来源数、引用覆盖率和检索质量，不达标最多补检一次。

存储上，Redis 保存带 7 天滑动 TTL 的近期消息和历史摘要；独立 Milvus collection 保存用户偏好、画像、研究兴趣及交互摘要；SQLite 只用于 LangGraph checkpoint；MySQL 保存文档元数据、问答运行、工具调用、检索证据和评测记录。系统通过 FastAPI/SSE 提供接口，并有检索消融和基础评测脚本。

我认为当前最需要补强的三点是：元数据过滤参数还没有真正下推 Milvus；引用评测只是标题匹配，不是 claim-level entailment；文档更新采用先删旧 chunk 再插入，缺少原子版本切换。

---

## 二、项目动机与业务价值

### 1. 为什么要做这个项目？普通聊天机器人不够吗？

**参考回答：** 普通聊天依赖模型参数知识或一次性上下文，竞品信息又经常更新，且定价、功能限制、技术路线等结论必须能追溯来源。本项目的核心不是“让模型知道更多”，而是建立一条从资料解析、证据召回、分析、引用到质量检查的可审计链路。

真正的业务价值有三点：减少人工阅读长报告的时间；用统一维度比较竞品，降低遗漏；让每个关键结论尽量能回到文档标题、章节、页码和原文片段。

### 2. 目标用户是谁？典型输入输出是什么？

**参考回答：** 目标用户是产品经理、战略分析、投研或售前人员。输入可以是本地 PDF、公开 PDF URL、实时 Web 查询或自然语言问题；输出是有证据的竞品问答，或者包含市场趋势、竞品对比、差异化、技术路线和风险的 Markdown 调研简报。

### 3. 为什么做成 Agent，而不是一条固定 RAG chain？

**参考回答：** 固定 chain 适合“单问题—单检索—单生成”。竞品研究问题常包含多个维度，需要先拆查询、决定是否调用知识库或实时 Web、判断证据缺口并补检。因此用 LangGraph 显式表达 Planner、Retrieval、Analysis、Summary、Reflection 和 Finalize，比把逻辑全部塞进一个 prompt 更容易控制、重试和观测。

但我不会说它是完全自主的多 Agent 系统。代码里本质上是一个有状态图，每个节点承担一种角色，工具集合也是受控的。

### 4. 项目从 0 到 1 你会怎么拆阶段？

**参考回答：** 我会按可验证的最小闭环推进：先做单文档解析和 Dense 召回；再补元数据和引用；然后加入 BM25 与重排并做消融；之后才加入 LangGraph、反思补检和记忆；最后补 FastAPI/SSE、MySQL 轨迹与评测。这样每加一层都能回答“它比上一版提升了什么”，而不是一次堆满组件后无法定位问题。

### 5. 这个项目最难的地方是什么？

**参考回答：** 不是把 API 串起来，而是三个边界：不同 PDF 结构不一致；Dense、BM25、Reranker 分数分布不同，融合需要评测；LLM 的计划和反思是非确定输出，不能直接当程序控制信号。所以代码里分别做了章节归一、分数归一与分层排序、Pydantic 校验和确定性质量门禁。

---

## 三、总体架构与请求链路

### 6. 用户问一个问题后，代码具体经过哪些对象？

**参考回答：** `CompetitiveResearchAgent.ask()` 调用 `LangGraphResearchWorkflow.invoke()`，初始 state 放入 question、thread/user id、messages、plan、retrieval_results 等。图按 Planner → Retrieval → Analysis → Summary → Reflection 执行；Reflection 根据 `should_retry` 回 Retrieval 或进入 Finalize。完成后 pipeline 将结果写入 MySQL trace 和 JSONL 性能日志。

### 7. Planner 实际做了什么？LLM 挂了怎么办？

**参考回答：** Planner 先用用户问题从长期记忆中取最多 4 条相关记忆，再要求 LLM 输出 objective、steps、search_queries、tool_calls、answer_format。输出会先做手工 normalize，再由 `PlanSchema` 校验。LLM 不可用时，系统用确定性计划：至少检索知识库；Tavily 可用时再加入 Web 搜索；research brief 模式会补充主题查询。

### 8. 查询改写怎么做？为什么最多 3 条？

**参考回答：** 原始问题、Planner 的查询和 Reflection 的 retry focus 先去重；LLM 可用时再生成覆盖市场、功能、定价、技术、限制等维度的查询，默认最多 3 条。限制数量是为了控制检索延迟、工具调用数和重复候选。它不是理论最优值，应该通过查询复杂度和 Recall/延迟曲线调整。

### 9. Analysis 和 Summary 为什么分开？

**参考回答：** Analysis 先输出 key findings、evidence map、gaps、confidence，把“证据判断”与“语言组织”分离；Summary 再根据 analysis 和最多 8 条证据写最终答案。这样更容易发现是召回失败、证据映射失败，还是最终表达失败，也方便 Reflection 检查。

### 10. Reflection 如何决定重试？

**参考回答：** 它不是完全听 LLM。规则分由证据数 25%、来源数 20%、标题引用覆盖率 30%、前几条平均检索分 25% 组成；默认规则分占最终分 65%，LLM 分占 35%。总分阈值为 0.6，同时还有证据至少 3 条、来源至少 2 个、引用覆盖率至少 0.5、平均检索分至少 0.45 的门禁。默认最多重试一次，重试时把 `retry_focus` 加入查询。

### 11. 为什么最多只反思一次？

**参考回答：** 反思循环不是越多越好。证据库没有相关资料时，多轮只会放大成本和延迟，甚至重复生成。当前值是一个工程保护；更成熟的做法是按问题类型、证据增益和成本预算动态停止，例如新一轮没有新增来源就提前终止。

### 12. LangGraph checkpoint 和 Redis 会话历史是不是重复？

**参考回答：** 有交集但职责不同。SQLite checkpointer保存图执行状态，支持同一 thread 的状态持久化和流程恢复；Redis 保存面向提示词的近期对话与压缩摘要，并设置 TTL。当前实现同时使用会增加一致性复杂度，生产环境应明确谁是会话真源，并设计清理和恢复策略。

---

## 四、文档解析与切块

### 13. 为什么选 Docling？为什么还保留 PyPDF？

**参考回答：** 竞品资料常包含多栏、表格和层级标题，Docling 更适合保留版面和 Markdown 结构，便于章节识别；但它依赖更重、启动和解析成本更高，也可能遇到兼容问题，所以用 PyPDF 作为轻量降级，保证文本仍可进入索引。代价是降级后表格和版面语义会损失。

### 14. 为什么还要做章节归一？

**参考回答：** Docling 只能较好地恢复文档结构，不能保证不同厂商都使用相同标题。Parser 将中英文标题映射为 overview、market、product、features、pricing、customers、competition、technology、limitations、roadmap、case_studies、key_takeaways。这样检索结果能携带统一业务 section，也便于后续按竞品维度过滤和解释。

### 15. 为什么 chunk_size=800、overlap=120？

**参考回答：** 800 是字符数，不是 token 数。对竞品资料而言，它通常能容纳一个相对完整的功能或定价描述；120 overlap 用于减少段落边界切断上下文。当前参数是经验初值，应通过不同 chunk size 的 Recall@K、重排质量、上下文长度和延迟做实验，而不是声称它天然最优。

### 16. `max_chunks_per_document=60` 真的是每篇最多 60 块吗？

**参考回答：** 不是，这是一个容易被代码追问出来的细节。当前限制写在 `_split_section()` 内，因此实际上是“每个 section 最多 60 块”；多个 section 相加可能明显超过 60。变量名有误导性，应改为 `max_chunks_per_section`，或者在整篇文档构建完成后再做全局上限和采样。

### 17. 页码是怎么来的？准确吗？

**参考回答：** Parser 保存页面文本，切章节后通过文本片段在 pages 中定位大致 page_start/page_end。它适合提供页码提示，但对 OCR、断词、表格或重复文本不一定精确，不能宣称是严格坐标级引用。更强的方案是在解析阶段保留 block/span 与 page 的映射，让 chunk 继承原始 page provenance。

### 18. 文档重复上传如何处理？

**参考回答：** 以 `document_id` 为逻辑标识。写新文档前先按 `document_id` 删除 Milvus 中旧 chunks，再生成 embedding 并分批 100 条插入；MySQL 文档记录走 upsert。缺点是删除和插入非原子，插入失败会出现短暂空窗或不完整版本。生产方案应使用 content hash、version/status 字段，完成新版本写入后再原子切 active 版本。

---

## 五、Embedding、Milvus 与混合检索

### 19. 为什么选择 BGE-M3？

**参考回答：** 这个项目有中英文混合的行业资料和查询，需要一个多语言、通用检索表现较好的模型；BGE-M3 与 BGE reranker 属于同一模型家族，工程组合相对自然，也能本地部署，避免每次 embedding 调外部 API。代码使用 `normalize_embeddings=True` 并配合 COSINE。

我不会只说“因为它最好”。正确选型应把它与至少一个轻量多语言模型或云 embedding 做离线对比，观察 Recall@K、模型体积、CPU/GPU 延迟和吞吐。当前代码动态执行一次 probe 获取向量维度，而不是把维度写死，这能在换模型时检测 collection 维度不匹配。

### 20. 为什么选择 Milvus，而不是 FAISS、Chroma 或 Elasticsearch？

**参考回答：** 本项目不仅要 Dense 向量检索，还要服务化持久化、标量元数据、独立 collection、BM25 sparse function 和后续扩展过滤。FAISS 更像向量索引库，持久化、元数据和多用户服务能力需要自己补；Chroma 适合原型，但本项目希望把 Dense 与原生 BM25 放在同一个向量基础设施；Elasticsearch 的全文检索成熟，但向量/RAG 侧的使用方式和团队栈不同。

Milvus 也不是没有代价：本地 standalone 运维比嵌入式库重，schema/embedding 维度迁移需要重建，规模很小时可能过度设计。因此答案要落到本项目需要的 Dense + Sparse、持久化和扩展性，而不是泛泛说“性能高”。

### 21. Milvus collection 里有哪些关键字段？

**参考回答：** 主键是随机 `chunk_id`；业务标识是 `document_id`。标量字段包括 industry、company、product_line、document_type、language、source、title、section、source_url、published、page range、status 和 metadata JSON；向量字段是 Dense `embedding` 和 `bm25_sparse`。content 开启 analyzer，并由 Milvus BM25 function 生成 sparse vector。

### 22. 为什么用 COSINE？embedding 为什么要归一化？

**参考回答：** 归一化后向量长度统一，余弦相似度更直接反映方向相似性，也能降低模长差异造成的影响。对单位向量，cosine 与点积排序等价。必须保证入库文档和查询使用同样的模型及归一化方式，否则分数阈值没有可比性。

### 23. Dense Retrieval 解决什么，BM25 解决什么？

**参考回答：** Dense 擅长语义改写，例如“企业安全控制”和“组织级治理能力”措辞不同仍可能相似；BM25 对 Cursor、Copilot、套餐名、版本号、价格数字和功能专有名词更敏感。竞品研究同时存在概念问题和精确实体问题，因此混合召回比单路更稳。

### 24. 你用的是 Milvus 原生 Hybrid Search 吗？

**参考回答：** 不是。代码分别调用 Dense search 和 BM25 search，再在 Python 应用层按候选 key 合并。这一点必须说准确。好处是逻辑透明，能记录 semantic/BM25/hybrid/rerank 分解分数；缺点是两次网络查询，融合和归一化由应用负责，也没有利用 Milvus 原生 ranker 的统一执行能力。

### 25. Dense 和 BM25 的分数怎么融合？

**参考回答：** Dense 直接使用 COSINE distance；BM25 命中分除以本次结果最大分，归一到近似 0~1；然后 `hybrid = 0.75 * semantic + 0.25 * bm25`。先取最多 12 个候选给 reranker。

这个方案简单但有明显边界：BM25 按每个 query 的最大值归一，跨 query 分数不可严格比较；没有命中某一路时该路记 0，可能惩罚单路强相关文档。更稳的替代是 RRF，或者基于标注集对分数做校准后再加权。

### 26. 候选怎么去重？有问题吗？

**参考回答：** 当前 key 是 `(source, section, content 前 120 字符)`，多查询合并时类似地用 source、section 和正文前缀。它比只按 source 去重细，但不是稳定 chunk id；两个正文开头相同的块可能误合并，文档重建后随机 UUID 也无法用于稳定引用。应使用 content hash + document version + section/local index 形成稳定主键。

### 27. 为什么还需要 BGE Reranker？

**参考回答：** Bi-encoder 在召回时分别编码 query/document，速度快但交互不足；CrossEncoder 把 query-document 成对输入，能更细地判断相关性，因此只对候选 Top 12 重排。代码用 sigmoid 把模型输出映射到 0~1，再按 `0.85 * model_score + 0.15 * hybrid_score` 得到最终分，最后返回 Top 6。

### 28. Reranker 挂了会怎样？

**参考回答：** 懒加载 reranker；加载失败会设置 `_reranker_load_failed`，默认 `reranker_fallback=True`，后续直接用 hybrid score，避免每次重复加载失败。如果关闭 fallback，则抛异常。还可以通过 warmup 提前暴露模型下载、内存和兼容问题。

### 29. 为什么 `reranker_max_length=512`？会截断什么？

**参考回答：** 这是延迟和显存/内存的折中。chunk 按字符切到 800，中英文 token 比例不同，query + chunk 仍可能超过 512 token，CrossEncoder 会截断尾部，关键结论若在后半段会丢失。应统计真实 token 长度分布，或采用标题/章节前缀、较小 chunk、滑窗 rerank 等方法。

### 30. Milvus 的 `AUTOINDEX` 有什么含义？为什么不手选 HNSW？

**参考回答：** 当前把 Dense 索引策略交给 Milvus 自动选择，简化本地与云环境适配，但可解释和可控性较弱。数据规模、QPS 和召回目标明确后，应通过 benchmark 比较 HNSW、IVF 等索引及 ef/nprobe 参数。面试中不能声称当前已经调过 HNSW，因为代码没有。

### 31. 多语言 BM25 怎么处理？

**参考回答：** content 开启 Milvus multi-analyzer，按 `language` 字段选择 English、Chinese 或 ICU default；查询时也根据字符启发式检测语言并传 `analyzer_name`。这比统一空格分词适合中英文混合，但代码里的语言检测较粗，混合语言、产品名和代码符号仍可能切分不理想。

### 32. 元数据过滤实现了吗？

**参考回答：** schema 和工具入参已经有 industry、company、document_type，但 `search_knowledge_base` 目前注释明确写着“retrieval currently ranks globally”，参数没有下推到 Milvus filter。回答时必须说“接口和数据已预留，但过滤尚未完成”。这是优先级很高的优化，否则多行业、多租户或同名产品场景容易串数据。

---

## 六、Redis、SQLite、MySQL、Milvus 分工

### 33. 四种存储各自做什么？为什么不能只用一个数据库？

**参考回答：** Milvus负责文档向量/BM25检索和长期记忆向量检索；Redis负责低延迟、带 TTL 的近期会话和摘要；SQLite 是 LangGraph 的本地 checkpoint；MySQL 是结构化业务与审计存储，保存文档目录、QA run、工具调用、证据和评测。它们是按访问模式拆分，不是为了堆技术。

如果是小型原型，可以简化为 SQLite/PostgreSQL + pgvector，减少运维；当前组合更像为后续独立扩展检索、缓存和审计做准备。

### 34. MySQL 具体有哪些表？

**参考回答：** `research_documents` 保存文档元数据和索引状态；`qa_runs` 保存问题、答案、来源、plan、reflection 和报告路径；`tool_calls` 保存参数、状态、耗时和结果预览；`retrieval_evidence` 保存每次运行的证据、分数与元数据；`eval_runs/eval_items` 保存评测汇总和单题结果。数据库关闭时，文档目录降级为 JSON 文件，trace 不保存。

### 35. 为什么不用 MySQL 保存向量？

**参考回答：** MySQL 在这里承担事务型和审计查询，Milvus承担 ANN 和 sparse 检索。把大规模 embedding 与业务表混在一起会让检索索引、扩缩容和生命周期管理相互耦合。当然，规模很小时使用支持向量索引的单一数据库可能更经济，选型取决于数据量和团队运维能力。

### 36. SQLite 在这个项目里保存聊天记录吗？

**参考回答：** 不能简单说保存聊天记录。它由 `SqliteSaver` 用于 LangGraph state checkpoint，连接设置 `check_same_thread=False`；面向提示词的近期消息实际由 Redis 管理。SQLite 适合单机开发，生产多实例会换成共享 checkpointer，并考虑连接池和并发写。

---

## 七、长短期记忆

### 37. 短期记忆怎么做？

**参考回答：** Redis 每个 thread 有两个 key：messages list 和 summary string。保存时只保留最近 `max_history_messages`，默认 6 条；溢出的旧消息被压缩进最多 1200 字符的摘要，下一次加载时作为 SystemMessage 放在近期消息前。两个 key 默认 TTL 604800 秒，也就是 7 天；读取和写入都会刷新 TTL，所以是滑动过期。

### 38. 历史压缩依赖 LLM 吗？失败怎么办？

**参考回答：** 开启压缩且 LLM 可用时，要求 LLM 合并旧摘要与溢出消息；否则把文本拼接、压缩空白并截取最后 1200 字符作为确定性 fallback。因此记忆链路不会因为 LLM 不可用而完全失败，但 fallback 可能丢掉早期重要信息。

### 39. 长期记忆存什么？为什么和知识库分 collection？

**参考回答：** 长期记忆存 preference、profile、research_interest 和 interaction_summary，字段含 user_id、kind、memory_key、importance、时间、访问次数、状态和 embedding。它与研究文档分开，避免用户偏好被当作商业事实证据召回，也便于按 user_id 强过滤和独立生命周期管理。

### 40. 长期记忆怎么召回和排序？

**参考回答：** 先用 BGE embedding 在 `user_id + active` 过滤下取最多 12 个候选，COSINE 小于 0.45 的丢弃。最终分是语义 65% + importance 15% + 90 天指数衰减的 recency 10% + memory kind 权重 10%，返回最多 4 条给 Planner。

### 41. 长期记忆怎么去重和更新？

**参考回答：** 新记忆先在同 user、同 kind 内检索最相似项，COSINE ≥ 0.92 就 upsert 原 memory_id。结构化偏好还会设置 `memory_key`，例如 output_language、response_detail、output_format；新偏好先删除同 key 旧值再插入，因此“喜欢中文”可被“喜欢英文”替换。

### 42. 所有问答都值得进入长期记忆吗？

**参考回答：** 不值得。当前 Finalize 会保存显式偏好/画像/兴趣，再额外保存一条 LLM 或确定性压缩的 interaction summary。这有记忆污染和无限增长风险。更成熟的方案需要记忆写入门控：只保存跨会话可复用、置信度高、用户允许的内容；设置容量、衰减、合并、删除和隐私策略。

### 43. 用户隔离做得怎么样？

**参考回答：** 长期记忆检索有 `user_id` filter，短期消息按 thread key 隔离。但这只是数据层逻辑隔离，不等于完整安全。API 层还需要认证，并从可信 token 推导 user_id/thread_id，不能完全信任客户端传参；知识库也需要 tenant filter，目前还未下推元数据过滤。

---

## 八、Pydantic、数据模型与工具系统

### 44. 为什么使用 Pydantic？具体用在哪里？

**参考回答：** LLM 输出和工具调用是最不可信的结构化边界。项目用 `PlanSchema`、`ReflectionSchema`、`EvidenceItemSchema` 校验 LLM JSON，用不同 Args model 校验工具参数。例如知识库 `top_k` 限制 1~20，Web `max_results` 限制 1~20，工具状态限定为 success/error/timeout。这样错误能在执行前暴露，也能统一 `model_dump()` 后的数据形状。

### 45. 为什么配置和内部 record 用 dataclass，而不是全用 Pydantic？

**参考回答：** 当前设计把可信的内部配置和轻量 record 用 dataclass，把外部/LLM 边界用 Pydantic，减少内部对象开销和依赖。但配置同样来自 JSON/环境变量，严格来说也值得用 Pydantic Settings 做类型、URL、范围和跨字段校验。现有 `_build_config` 只能靠 dataclass 构造报基础类型错误，约束较弱。

### 46. `extra="allow"` 是不是会掩盖 LLM 错误？

**参考回答：** 会有这个取舍。允许额外字段可以兼容模型偶尔附加解释字段，减少脆弱性；但关键控制结构更适合 `extra="forbid"`，否则拼错字段可能被静默保留，而默认值又让错误看起来像成功。当前代码在 Pydantic 前做 normalize 和默认计划兜底，但生产版应对关键 schema 更严格并记录 validation failure。

### 47. 工具调用怎样处理超时、重试和错误？

**参考回答：** 先用 `ToolCallSpec` 和具体 Args schema 校验；未知工具直接记录 error；调用在线程池中运行，默认 timeout 30 秒，失败最多重试 1 次，并记录 attempt、status、elapsed_ms、error_type 和 preview。需要指出：`execute_calls` 外层仍是顺序循环，线程池主要用于 timeout 包装，并未并行执行多个工具；而 Future 超时后也没有真正取消底层任务，这是当前改进点。

### 48. 为什么工具结果还要 normalize？

**参考回答：** 知识库、Web 搜索、PDF 收集返回结构不同，后续 Analysis/Reflection 需要统一 evidence 结构。normalize 后至少包含 content、source、title、score、section、origin 和 metadata，减少下游 prompt 分支，也便于统一写入 MySQL `retrieval_evidence`。

---

## 九、可观测性、评测与性能

### 49. 你如何证明 Hybrid/Reranker 有用？

**参考回答：** 代码提供 `dense_only`、`hybrid`、`hybrid_rerank` 三个消融变体，比较 Recall@3/5、citation accuracy、关键词/答案要点覆盖和工具成功率。正确做法是固定文档、问题集、embedding 和生成配置，多次运行并报告均值、延迟和成本，而不是只展示一个成功案例。

仓库中已有的部分历史结果仍使用旧的 `paper_*` collection/字段名，与当前行业竞品版本不完全一致。因此它们只能说明评测框架跑过，不能直接当成当前版本的有效性能证明；应重新灌入当前 fixtures 后跑一轮并保存环境、时间和配置。

### 50. 当前 Recall@K 是怎么算的？有什么缺陷？

**参考回答：** 根据输出 sources 前 K 项是否包含期望文档的 id、title 或文件名做字符串匹配。它简单可审计，但不是 chunk-level relevance judgement；同名或标题变体可能误判，也无法评估“找到文档但没找到正确段落”。应增加 query-chunk 标注、nDCG/MRR 和人工 relevance grade。

### 51. citation accuracy 真的能判断引用正确吗？

**参考回答：** 当前只判断来源字符串能否匹配已知文档，Reflection 的 citation coverage 也只检查回答里是否出现 `[标题]`。这能检查“有没有引用已知来源”，不能检查某个 claim 是否被对应原文蕴含。下一步应拆 claim，检索对应 evidence，再用 NLI/LLM judge 加人工抽检做 entailment、完整性和冲突评估。

### 52. 你记录哪些性能指标？

**参考回答：** 节点级耗时、query/document embedding、Milvus search、BM25、rerank、工具状态/耗时/重试、LLM token/JSON 解析、workflow success/error、reflection retry 和 rescue 等会进入 performance collector，并持久化 JSONL/报告；业务 trace 进入 MySQL。面试时最好现场展示一条 run 的 planner、tool_calls、evidence、reflection 和 latency，而不只展示最终答案。

### 53. SSE 是如何实现的？是真正的异步流吗？

**参考回答：** `ask_stream()` 启动 daemon thread 执行同步 `ask()`，通过 Queue 把 node、token、result、done 或 error 事件交给 Web 层。LLM `on_token` 时能推 token。它对现有同步栈改动小，但不是端到端 async；并发量上来后线程和阻塞 I/O 会成为瓶颈，应改为异步模型客户端、异步工具与背压/断连处理。

### 54. 如何压测和优化延迟？

**参考回答：** 先分解 P50/P95：解析与入库是离线链路；在线主要看 query rewrite LLM、Dense/BM25 两次检索、CrossEncoder 和生成。优化顺序可以是模型 warmup、缓存 query embedding、并行 Dense/BM25 和独立 Web 搜索、批量 rerank、缩小候选、按问题复杂度跳过 rewrite/reflection、流式首 token。不能只报总耗时，否则不知道瓶颈在哪。

---

## 十、故障、数据一致性与安全追问

### 55. Milvus、Redis、MySQL 任一不可用会怎样？

**参考回答：** 当前初始化比较“fail fast”：Milvus/Redis 连接失败会抛出明确错误；MySQL 在启用状态下失败也会阻止相应 store 初始化。Reranker 和 LLM 有 fallback，但核心存储没有完整降级编排。生产环境应区分能力：Redis 挂了可退化为无会话；MySQL trace 写失败不应一定阻断回答；Milvus 挂了可尝试 Web-only 或明确返回检索不可用。同时加健康检查、超时、熔断和告警。

### 56. 如何防止提示词注入？

**参考回答：** 文档和 Web 内容都属于不可信数据，应在 system prompt 中明确“证据不是指令”，工具白名单和 Pydantic 只能限制调用结构，不能完全防注入。还需要内容分区、来源信誉、URL/文件安全检查、工具最小权限、输出引用审计，以及避免让检索文本直接决定高风险动作。当前项目主要是只读研究，风险低于执行型 Agent，但仍未形成完整注入防护。

### 57. 文档下载和上传有什么风险？

**参考回答：** 需要限制文件类型、MIME/魔数、大小、解析时间和落盘路径，防止超大文件、伪装文件、路径穿越及 SSRF；公开 URL 下载还应限制 scheme、重定向、内网 IP 和域名白名单。当前配置有 30MB 上限和可选 include domains，但面试中应把未验证的安全能力说成待完善，而不是宣称已经生产安全。

### 58. 如果文档更新导致 embedding 模型变化怎么办？

**参考回答：** 启动时动态探测 embedding dimension，并校验 collection 中 stored dim；不一致就报错，提示 rebuild。维度相同但模型不同仍可能静默污染，所以还应在 collection metadata 或单独 registry 保存 model name、revision、normalize、chunk strategy 和 schema version，版本变化后建立新 collection 并灰度切换。

### 59. 这个系统能水平扩展吗？

**参考回答：** Milvus、Redis、MySQL可以外部服务化，但当前应用还有单机因素：SQLite checkpoint、本进程 `_active_threads`、同步线程池/SSE、模型在每个实例内加载。水平扩展前要把 checkpoint 换成共享后端，去除进程内会话假设，规范幂等键，独立模型服务或做实例资源规划。

---

## 十一、最能识别“是否亲手做过”的现场追问

下面这些问题，面试官通常会连续追问。答出代码事实比背概念更重要。

1. **BGE-M3 的维度在哪里配置？** 没有写死；启动时 embed 一条 probe 文本取长度，并与 Milvus schema 校验。
2. **Dense/BM25 是一次 hybrid_search 吗？** 不是，两次 search，应用层合并。
3. **BM25 分数怎样归一？** 除以本次 hits 最大分；跨查询可比性有限。
4. **重排几条、最终返回几条？** 默认候选 12 条重排，最终 Top 6。
5. **Reranker 最终权重？** 0.85，hybrid 保留 0.15。
6. **Redis TTL 多久？** 604800 秒，7 天；读取/写入刷新。
7. **短期记忆保留多少消息？** Graph 默认 6 条，溢出内容压缩进 summary。
8. **长期记忆过滤条件？** user_id、status=active，可选 kind。
9. **长期记忆去重阈值？** 同 kind 的 cosine ≥ 0.92。
10. **长期记忆最终排序公式？** semantic 0.65 + importance 0.15 + recency 0.10 + kind 0.10。
11. **SQLite 保存什么？** LangGraph checkpoint，不是 MySQL trace，也不是 Redis 近期消息。
12. **MySQL 关闭后呢？** 文档元数据用 JSON，workflow trace 不保存。
13. **Reflection 默认最多几次补检？** 1 次。
14. **工具是否并行？** 当前多个 call 顺序执行，线程池用于单 call timeout 包装。
15. **元数据过滤是否完成？** 字段和参数有，Milvus 检索尚未应用。
16. **更新文档是否原子？** 不是，先 delete 再 insert。
17. **`max_chunks_per_document` 真是全篇上限吗？** 不是，当前实际按 section 限制。
18. **引用准确率是否是 claim-level？** 不是，目前主要是来源/标题字符串匹配。

---

## 十二、面试时不要这样说

- 不要说“用了 Milvus 原生混合检索”；实际是应用层融合。
- 不要说“元数据过滤已经实现”；当前只预留 contract/schema。
- 不要说“引用绝对可靠”；页码是近似定位，引用评测也不是蕴含判断。
- 不要说“工具是并发调用”；当前循环是顺序的。
- 不要说“历史消融数字就是当前系统结果”；已有部分 artifact 带旧 `paper_*` 命名，应重跑。
- 不要说“所有记忆都会越用越准”；当前还存在记忆污染、容量和隐私问题。
- 不要说“参数是调出来的”，除非你能展示实验记录。更诚实的说法是“这是经验初值，项目已有消融框架，下一步用数据校准”。

## 十三、建议准备的现场证据

面试前至少亲自完成并截图/记录以下动作：

1. 导入一份 PDF，展示 Docling backend、章节结果、chunk 数和 Milvus字段。
2. 对同一问题分别关闭 BM25、关闭 reranker，比较 Top 6、分解分数和耗时。
3. 连续两轮同 thread 对话，查看 Redis messages/summary 与 TTL。
4. 输入“以后请用中文简洁回答”，查看长期记忆 kind、memory_key 和下一轮召回。
5. 查询 MySQL 的一条 `qa_runs`，关联 `tool_calls` 和 `retrieval_evidence`。
6. 人为让回答缺引用，展示 Reflection gate、retry_focus 和一次补检。
7. 让 reranker 不可用，展示 fallback 到 hybrid score。
8. 用当前 collection 和 fixtures 重新执行消融，保留配置、日志、结果与 bad cases。

如果这八件事你都亲自跑过，面试官把问题追到代码层，你也不会慌，因为回答来自你观察过的系统行为，而不是背诵。
