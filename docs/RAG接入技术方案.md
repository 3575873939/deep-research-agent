# Deep Research Agent —— RAG 接入技术方案

| 项 | 内容 |
|---|---|
| 文档版本 | v1.0 |
| 对应代码 | `deep-research-agent/src/dra/rag/`、`src/dra/graph/researcher.py` |
| 状态 | 已实施、已测试、已实测 |

---

## 1. 背景与问题

改造前，调研节点的取数链路是「**整页灌入**」：

```
search（多 query 串行）→ fetch top5 页面 → 每页按 token 级截断（16,000 字符）→ 全部拼进 researcher 的 context → LLM 写摘要
```

实测暴露三个问题：

| 问题 | 表现 |
|---|---|
| **噪声稀释** | 抓回的网页里，导航栏、广告、无关段落占大头。子问题只需其中几段，却把整页塞进 context |
| **token 浪费** | 单题的 5 个来源合计可达 9–10 万字符，其中大部分与子问题无关 |
| **注意力分散** | 长 context 中真正相关的证据被淹没，模型倾向抓取显眼但不相关的内容 |

**核心判断**：这不是「模型不够强」，而是**证据构造方式不对**——把「检索」退化成了「下载」。

---

## 2. 目标与非目标

### 目标
1. 把进 context 的材料从「网页全文」收敛为「相关段落」，显著降低 researcher 节点 token
2. **不牺牲可验证性**：引用体系（`[cite:N]` / SQLite 快照 / sha256）语义保持不变
3. **不引入硬依赖**：任何可选组件缺失时自动降级，主链路必须始终可跑（CI / 无 GPU / 断网）

### 非目标
- 不引入向量数据库（每 run 语料仅几十页，进程内暴力检索足够）
- 不改动 verifier / synthesizer / eval 的逻辑与语义
- 不追求检索指标的极致，优先保证「不退化 + 可降级」

---

## 3. 总体设计

### 3.1 架构

```
                        ┌──────────────────────── researcher 节点 ────────────────────────┐
                        │                                                                  │
  search ──► fetch top5 │  ① 原文快照入库（每来源一条引用，快照=整页原文）                   │
     页面文本            │        ↓                                                          │
                        │  ② 切分 chunk（500 token / 50 overlap，句边界对齐，附 sha256）      │
                        │        ↓                                                          │
                        │  ③ 双路召回        ┌── BM25Plus（词面精确）  top10 ──┐               │
                        │                    └── 向量索引（语义）      top10 ──┤               │
                        │        ↓                                            ↓               │
                        │  ④ RRF 融合（k=60，免调权）→ top(3×top_k)                            │
                        │        ↓                                                          │
                        │  ⑤ 可选 reranker（bge-reranker cross-encoder）→ top_k               │
                        │        ↓                                                          │
                        │  ⑥ 按来源聚合 → 组装证据块（带 [citation_id]）→ LLM 摘要            │
                        └──────────────────────────────────────────────────────────────────┘

  降级：②③④⑤ 任一步不可用/无命中 → 回退「整页截断」（改造前的原始行为）
```

### 3.2 模块划分

| 模块 | 文件 | 职责 |
|---|---|---|
| 切分 | `rag/ingest.py` | 滑窗切分、句边界对齐、内容哈希 |
| 稀疏+融合检索 | `rag/retriever.py` | 中英混合分词、BM25Plus、双路召回、RRF 融合 |
| 向量索引 | `rag/embedding.py` | bge-small-zh 编码、余弦检索、阈值过滤、模型缓存 |
| 精排 | `rag/reranker.py` | bge-reranker cross-encoder，不可用则退化为 RRF 顺序 |
| 查询改写 | `rag/query_rewrite.py` | 子问题 → 检索式（独立能力，本次未接入主链路） |
| 接入点 | `graph/researcher.py` | `_build_evidence()` / `_fallback_evidence()` |

### 3.3 责任边界（关键设计）

> **引用粒度不变**：每条引用仍对应「一个来源 URL + 整页快照」。chunk 只携带 `src_idx` 指回来源，
> 不单独建引用。

这条边界决定了本次改造的**风险面极小**：

- `verifier` 读的是 `citations.get(cid).snapshot`（整页），完全不受选块影响
- `eval` 的引用可验证率 / 幻觉率口径不变
- `synthesizer` 的输入仍是 findings（带 `[cite:N]`），格式不变

换句话说：**RAG 只改变「喂给 researcher 的上下文」，不改变「引用链」**。

---

## 4. 详细设计

### 4.1 切分（`rag/ingest.py`）

| 参数 | 值 | 说明 |
|---|---|---|
| `CHUNK_CHARS` | 1200 字符 | ≈ 500 token（中文按 1 token ≈ 2.4 字符保守估算，见源码注释） |
| `OVERLAP_CHARS` | 120 字符 | ≈ 50 token，防止答案被切断在边界 |
| 边界对齐 | `。` / `\n` | 断点落在窗口后半段时才回退到句边界，避免产生碎块 |
| `content_hash` | sha256(chunk) | 内容指纹，溯源用 |

**为什么不用语义切分**：语义切分需要额外模型调用，成本与复杂度不匹配本场景；滑窗 + 句边界对齐已足够，且完全确定性、可测试。

### 4.2 分词（中英混合）

```python
英文/数字：re.findall(r"[a-z0-9]+")      # 按词
中文：      单字 + 相邻 bigram             # 免 jieba 依赖
```

**取舍**：中文 bigram 的召回不如专业分词器精确，但**零依赖、确定性、可离线**。对本场景（判断「这一段是否与子问题相关」）足够——真正的选优交给 RRF 与向量分支。

### 4.3 稀疏召回：BM25Plus

- 实现：`rank_bm25.BM25Plus`，取 top10
- 选 Plus 变体而非 Okapi：小语料下 Okapi 的 IDF 可能为负导致分数全 0

**已修复的缺陷**：`BM25Plus` 公式含 `delta` 项，**零命中文档也会得到正分**，导致 `score > 0` 的过滤形同虚设、无关块被选入。修复方式是改为按**词项重叠**过滤：

```python
# 索引构建时缓存每个 chunk 的词项集合
self._token_sets = [set(t) for t in tokens]
# 检索时要求与 query 至少共享一个词项
rankings.append([i for i in bm25_top if self._token_sets[i] & q_tokens])
```

该缺陷是**被新增的集成测试暴露出来的**（见 §8）。

### 4.4 稠密召回：向量索引（`rag/embedding.py`）

| 项 | 值 |
|---|---|
| 模型 | `BAAI/bge-small-zh-v1.5` |
| 相似度 | 归一化 embedding 后矩阵点积 = 余弦 |
| 检索 | 进程内暴力 top-k（几十~几百 chunk 规模足够） |
| `min_score` | **0.45**（见下） |

**阈值 0.45 的由来**：向量检索没有「零命中」概念，无条件返回 top-k 会把无关块灌进证据（实测有子任务因此拿到完全不相关的材料、最终**零引用**）。为定阈值，实测了 bge-small-zh 的分数分布：

```
相关对：0.390 ~ 0.688（均值 0.520）
无关对：0.162 ~ 0.412（均值 0.304）
```

取 0.45：在样本上剔除全部无关对与最弱的相关对。**偏向严格**是刻意的——噪声的代价（证据被污染）远大于漏召的代价（BM25 分支仍可兜底）。

**两个健壮性设计**：

| 问题 | 现象 | 方案 |
|---|---|---|
| HF 联网检查卡死 | 不设离线模式时，加载模型会挂起数分钟（无法直连 huggingface.co） | `local_files_only=True` 默认只读本地缓存；无缓存则直接降级 BM25-only |
| 模型重复加载 | 多个子问题并行时各加载一次，各耗十几秒且同步阻塞事件循环 | 进程级 `_MODEL_CACHE`，按 `(模型, device, local_files_only)` 缓存 |

### 4.5 RRF 融合

```python
score(doc) = Σ_lists 1 / (k + rank + 1)      # k = 60
```

**为什么用 RRF 而不是加权求和**：无需为 BM25 与余弦分数调权重（两者量纲完全不同），对分数尺度不敏感，工程上更稳。

在 `_build_evidence` 中，先取 `3 × rag_top_k` 粗排，再交给（可选）reranker 精排到 `rag_top_k`。

### 4.6 可选精排：reranker

- `BAAI/bge-reranker-base` cross-encoder，对 (query, chunk) 逐对打分
- 默认**关闭**（`rag_use_reranker=False`）：模型更重，收益在 fetch_topk=5 的小语料上有限
- 依赖/模型不可用时 `fallback()` 退化为 RRF 顺序

### 4.7 按来源聚合与证据组装

选出的 chunk 需要重新组装成 LLM 能用的证据块：

1. 按 `src_idx` 分组；组间按**最佳名次**排序（越相关的来源越靠前）
2. 组内去重后以 `\n…\n` 拼接（相邻 chunk 有 120 字符重叠，避免重复占位）
3. 每个来源生成一个证据块，沿用改造前的格式：

```
[citation_id=1837] 标题
URL: https://...
<untrusted_web_content>
   命中的段落（可能多段，… 分隔）
</untrusted_web_content>
```

**注意**：`<untrusted_web_content>` 隔离包裹与注入清洗（`guard/injection.py`）在两条路径下都保留。

### 4.8 降级链（设计红线）

「任何可选组件缺失都不能阻断主链路」被落实为四级降级：

| 触发条件 | 行为 |
|---|---|
| `rag_enabled = False` | `_build_evidence` 返回空 → 走 `_fallback_evidence`（整页截断，改造前行为） |
| 无 `sentence-transformers` / 模型未缓存 | `build_vector_index` 返回 `None` → HybridRetriever 仅用 BM25 |
| 无 `bge-reranker` | `Reranker.fallback()` → 直接采用 RRF 顺序 |
| 切分为空 / 检索无命中 | 返回空列表 → 调用方回退整页截断 |

**验证方式**：在 `sentence-transformers` 不可用的环境下（本机曾因 `huggingface-hub` 版本冲突处于该状态），整套流程仍可完整跑通（`bm25-only` 模式）——这是刻意的设计目标而非偶然。

### 4.9 接入点改动（`graph/researcher.py`）

`_research_one` 的改动局限在一处：

```python
# 改造前：入库 + 截断 + 拼证据，三段耦合在一个循环里
# 改造后：入库与证据构造分离

# 2.5) 原文快照入库（引用粒度不变）
sources = []
for f in fetched:
    cid = citations.add(run_id=run_id, url=f["url"], title=f["title"], snapshot=f["text"])
    sources.append({"cid": cid, "url": ..., "title": ..., "text": f["text"]})

# 3) 证据构造：先试混合检索，不可用/无命中则回退整页截断
if sources:
    evidence_blocks = _build_evidence(subquestion, sources, settings)
    if not evidence_blocks:
        evidence_blocks = _fallback_evidence(sources, settings)
```

---

## 5. 配置项

```bash
# 检索增强（RAG）
RAG_ENABLED=true                    # 总开关，false 即回到改造前行为
RAG_TOP_K=16                        # 送进 LLM 的证据块数（chunk 级）
RAG_USE_VECTOR=true                 # 需 sentence-transformers，缺失自动降级 BM25-only
RAG_USE_RERANKER=false              # 需 bge-reranker，缺失自动降级 RRF 顺序
RAG_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
RAG_RERANKER_MODEL=BAAI/bge-reranker-base
RAG_LOCAL_FILES_ONLY=true           # 只读本地模型缓存，避免无外网时阻塞
RAG_VECTOR_MIN_SCORE=0.45           # 向量召回相关性下限

# 相关既有配置（降级路径使用）
RESEARCHER_FETCH_TOPK=5
RESEARCHER_MAX_TOKENS_PER_SOURCE=4000
```

---

## 6. 可观测性

| 信号 | 位置 | 用途 |
|---|---|---|
| 节点级 token | `trace.token_usage`（planner / researcher / synthesizer / verifier） | 归因 token 变化 |
| 工具调用明细 | `trace.tool_calls`（含 `ok` / `error_type` / `latency_s`） | 区分「检索效果」与「抓取失败」 |
| 引用快照 | SQLite `citations` 表 | 事后确定性复现（见 §7.2） |

---

## 7. 实验与实测数据

### 7.1 端到端 A/B（同模型、同题）

**条件**：ModelScope `deepseek-ai/DeepSeek-V4.1-Flash`，题号 `t01`（LangGraph interrupt/checkpoint）

| 节点 | RAG-OFF | RAG-ON | 变化 |
|---|---|---|---|
| planner | 475 | 416 | −12% |
| **researcher** | 44,553 | 25,840 | **−18,713（−42%）** |
| synthesizer | 7,317 | 4,591 | −37% |
| verifier | 104,638 | 58,710 | −44% |
| **合计** | **156,983** | **89,557** | **−67,426（−43%）** |
| LLM 调用次数 | 86 | 50 | −42% |

**质量**：两臂均为 **引用可验证率 1.000 / 幻觉结论率 0.000**，未退化。

### 7.2 确定性复现（零 LLM 成本）

利用一个实现事实：**同一任务内 5 次 `citations.add` 之间没有 `await`**，因此同任务的引用 id 在 SQLite 中必然连续 → 可用已存快照还原「每任务的 5 个来源」，把两条路径各重放一遍。

对历史 **47 次运行**复现结果：

```
压缩率：均值 51.2%   中位 52.8%   范围 16.1% ~ 69.2%   (n=47)
总字符：5,942,532 → 2,782,508    综合压缩 53.2%
```

**交叉验证**：对 `t01` 同一批语料，确定性方法预测 **42.2%**，端到端实测 researcher 节点 **−42%**——两条独立路径吻合，说明效果是机制性的。

复现脚本：`evals/rag_ab_replay_all.py`

**压缩最弱的情形**（值得关注）：压缩率最低的题是 prompt injection 类（16.1%），原因是抓到的页面高度同质、几乎每段都相关，没有噪声可滤。这说明 **RAG 的收益与「语料噪声比」正相关**。

### 7.3 方法学局限（如实声明）

- 端到端 A/B 两臂相隔一天，抓取到的网页不完全相同（工具调用 30 vs 24 次）
- −43% 的总量差包含**连带效应**：OFF 臂产出的报告结论更多（79 vs 44 条），verifier 逐条核查故调用更多。**严格可归因于 RAG 的是 researcher 的 −42%**
- 确定性复现的 n=47 是**运行次数**，非 47 道不同题目
- 该复现量化的是 **context 体积**，token 换算需用字符/token 比估算

---

## 8. 测试策略

`tests/test_rag.py`（全量 23 passed）新增 4 个用例：

| 用例 | 验证点 |
|---|---|
| `test_hybrid_retriever_vector_branch_when_bm25_misses` | 用 stub VectorIndex 证明向量分支确实并入 RRF（不依赖真实 embedding，可离线跑） |
| `test_evidence_selects_relevant_chunk_and_keeps_citation_id` | 选块只保留相关段落，且 `[citation_id]` 仍指回整页快照 |
| `test_evidence_disabled_returns_empty` | 开关关闭时正确返回空，交由调用方降级 |
| `test_fallback_evidence_truncates_per_source` | 降级路径的截断行为符合预期 |

**测试驱动发现的缺陷**：`test_evidence_selects_...` 首次运行失败，暴露出 §4.3 的 `BM25Plus` delta 问题——这是「先写测试」的直接收益。

---

## 9. 已知限制与后续优化

| 限制 | 说明 | 可选优化 |
|---|---|---|
| 向量分支需额外安装 | 需 `sentence-transformers` 且模型已缓存（本机为 bge-small-zh，约 91MB） | 生产可预置模型到镜像 |
| 阈值 0.45 是基于小样本定标 | 相关/无关分布有重叠区（0.39 vs 0.41） | 用更大标注集调参，或改为自适应阈值 |
| 中文 bigram 分词较粗糙 | 召回不如专业分词器 | 可选接入 jieba（仍需保持可降级） |
| `query_rewrite` 未接入 | 现有 `_expand_queries` 是规则式，零成本确定性 | 可用 light 模型改写提升召回（会增加调用数） |
| reranker 默认关闭 | 未在真实链路验证收益 | 在 fetch_topk 更大时再评估 |
| 效率 | `build_vector_index` 为同步调用，会短暂阻塞事件循环（仅首次，受模型缓存缓解） | 改为 `asyncio.to_thread` |

---

## 10. 风险与回滚

| 风险 | 缓解 |
|---|---|
| 检索质量差导致报告变差 | 四级降级；`RAG_ENABLED=false` 一键回到改造前行为 |
| 可选依赖拖慢启动 | `local_files_only` + 进程级模型缓存；不可用即降级 |
| 引用链被破坏 | 引用粒度设计上不变，verifier / eval 零改动 |
| 免费通道限流 | 客户端节流 `DRA_MIN_CALL_INTERVAL_S` + 429 退避重试 |

**回滚方式**：设 `RAG_ENABLED=false` 即可，无需改代码。

---

## 11. 附：面试问答要点

**Q：为什么不是简单加大 top-k 或换更强的模型？**
A：问题不在模型能力，而在证据构造。整页灌入把「检索」做成了「下载」，噪声稀释了注意力。RAG 选块是把 context 从「网页全文」收敛为「相关段落」。

**Q：为什么不用向量数据库？**
A：每 run 语料只有几十页，进程内暴力余弦检索足够。引入向量库会增加部署复杂度与故障面，收益为零。规模上来再换。

**Q：向量检索最大的坑是什么？**
A：它没有「零命中」概念。BM25 查不到就是 0 分，向量永远返回 top-k。不设相似度阈值就会把无关块灌进证据——我实测出现过子任务拿到完全不相关的材料、最终零引用。所以加了 0.45 阈值，而且刻意偏向严格。

**Q：怎么证明 RAG 真的有效，而不是随机波动？**
A：两条独立路径互相验证。一是端到端 A/B（同模型同题，researcher 节点 −42%）；二是确定性复现——用 SQLite 里已存的网页快照把两条路径重放，对 47 次运行的 context 体积中位压缩 52.8%，且对同一题预测 42.2%，与实测吻合。

**Q：这个改造最大的风险是什么？**
A：破坏引用可验证性。所以我把「引用粒度不变」定为红线——chunk 只带 `src_idx` 指回来源，引用仍是一来源一条、快照仍是整页，verifier 和 eval 完全不用动。

---

## 附：文件清单

```
src/dra/rag/
├── ingest.py          # 滑窗切分 + 句边界对齐 + sha256
├── retriever.py       # 中英分词 + BM25Plus + 双路召回 + RRF
├── embedding.py       # bge-small-zh 向量索引 + 阈值 + 模型缓存
├── reranker.py        # bge-reranker（可选，退化到 RRF 顺序）
└── query_rewrite.py   # 查询改写（独立能力，未接入主链路）

src/dra/graph/researcher.py   # _build_evidence / _fallback_evidence / 接入点
src/dra/config.py             # RAG 开关

tests/test_rag.py             # 单元 + 集成测试
evals/rag_ab_replay_all.py    # 确定性复原脚本（零 LLM 成本）
```