# Paper Agent 涉及的 AI 技术全解

> 每一行代码对应的 AI 知识点，帮你理解「为什么这么写」

---

## 一、整体架构涉及的概念

### 1. Agent（智能体）

**是什么**：让 LLM 不只是「一问一答」，而是能调用工具、多步推理、自主完成任务。

**在项目中**：整个 Paper Agent 就是一个 Agent——用户给一个 query，它自己决定搜什么、读什么、怎么写综述。

**对应文件**：`graph.py`（LangGraph 编排）、`main.py`（入口）

### 2. Pipeline vs ReAct

| 模式 | Paper Agent 的做法 |
|:---|:---|
| **Pipeline（管道）** | `plan → search → rank → download → parse → extract → compare → synthesize → review` 是固定流程 |
| **ReAct（推理-行动循环）** | 如果搜索结果不够，可以重新规划再搜（进阶阶段） |

**在项目中**：当前用 Pipeline（MVPM 确定性更强），`graph.py` 里 10 个节点串联。

### 3. LangGraph（Agent 编排框架）

**是什么**：LangChain 团队出的框架，用「图」（节点+边）来描述 Agent 流程。

**核心概念**：

| 概念 | 在 Paper Agent 中的对应 |
|:---|:---|
| **StateGraph** | `build_graph()` — 构建整个 Agent 的图 |
| **Node（节点）** | `plan`、`search`、`rank` 等 10 个节点，每个是一个 async 函数 |
| **Edge（边）** | `add_edge("plan", "search")` — 定义节点间的流转 |
| **ConditionalEdge（条件边）** | `_should_compare()` — 当 claims ≥ 2 篇论文时才走对比节点 |
| **State（状态）** | `AgentState` — 所有节点共享的 TypedDict，传递数据 |
| **Reducer（累加器）** | `Annotated[list, add]` — 多个节点返回的结果自动追加到列表 |

**对应文件**：`graph.py`、`state.py`

---

## 二、LLM 调用涉及的概念

### 4. LLM（大语言模型）

**是什么**：训练在海量文本上的神经网络，给定上文，预测下一个 token。

**在项目中**：
- Anthropic Claude（默认）: `anthropic.AsyncAnthropic`
- DeepSeek: `openai.AsyncOpenAI` + `base_url=https://api.deepseek.com/v1`

**对应文件**：`llm/client.py`

### 5. System Prompt vs User Prompt

| 类型 | 作用 | 示例 |
|:---|:---|:---|
| **System Prompt** | 定义角色、规则、输出格式 | "你是一个学术综述撰写专家" |
| **User Prompt** | 具体的任务输入 | "请对以下论文生成综述：..." |

**在项目中**：每个 Agent 节点都有专属的 System Prompt（如 `PLANNER_SYSTEM`、`SYNTHESIS_SYSTEM`、`EXTRACTION_SYSTEM`）。

**对应文件**：`planner.py`、`synthesizer.py`、`extract.py`、`compare.py`、`reviewer.py`

### 6. Structured Output / JSON Mode

**是什么**：让 LLM 输出结构化的 JSON 而不是自由文本，方便程序解析。

**在项目中**：`generate_with_json()` 函数——让 LLM 输出 `{"claims":[{"claim_type":"method",...}]}`，然后解析为 Pydantic 对象。

**对应文件**：`llm/client.py`（`generate_with_json` 方法）、`extract.py`、`compare.py`

### 7. Token 和 Max Tokens

**是什么**：
- Token ≈ 0.75 个英文单词 ≈ 0.5 个中文字
- `max_tokens` 限制 LLM 一次最多输出多少 token

**在项目中**：`LLMConfig.max_tokens = 4096`（约 3000 字），每个 LLM 调用都限制在这个范围内。

### 8. Temperature（温度参数）

**是什么**：控制 LLM 输出的「创造性」。0 = 确定性强（适合事实性任务），1 = 随机性高（适合创意写作）。

**在项目中**：全部设为 `0.3`（偏确定——学术论文场景需要准确性）。

---

## 三、RAG 涉及的概念

### 9. RAG（检索增强生成）

**是什么**：不靠 LLM 记忆回答问题，而是先从外部知识库检索相关文档，再把文档喂给 LLM 生成答案。**LLM 的记忆不可靠，RAG 让它「有据可查」。**

**完整的 RAG 流程**：

```
用户 Query → 检索(Retrieval) → 排序(Rank) → 阅读(Read) → 生成(Generate) → 带引用输出
```

**在项目中**：Paper Agent 就是 RAG 在学术领域的完整实现——知识库是 arXiv + Semantic Scholar，生成的是综述。

**对应文件**：几乎所有文件都参与

### 10. Multi-source Retrieval（多源检索）

**是什么**：从多个数据源检索，合并去重，提高覆盖率。

**在项目中**：
- arXiv API → 预印本元数据 + PDF 链接
- Semantic Scholar API → 引用关系 + 影响力排序
- `asyncio.gather()` 并发调两个 API → 去重合并

**对应文件**：`search.py`

### 11. Re-ranking（重排序）

**是什么**：检索返回的结果通常很多，需要按相关性重新排序。

**在项目中**：`rank()` 按标题去重，按 `max_papers` 截断。进阶可以按 `citation_count` 或 LLM 相关度评分排序。

**对应文件**：`ranker.py`

---

## 四、证据与可追溯性涉及的概念

### 12. Evidence Grounding（证据锚定）

**是什么**：AI 生成的每一条结论，都必须附上来源证据（哪篇论文、哪个章节、哪一页）。

**在项目中**：每个 `Claim` 对象包含：
```python
Claim(paper_id="2301.12345", claim_text="...", evidence="原文引用", section="3. Method", page=3)
```

**对应文件**：`models/claim.py`、`extract.py`

### 13. Hallucination（幻觉）

**是什么**：LLM「编造」不存在的事实。

**如何缓解**：
- 强制每条 claim 带原文引用（`extract_claims`）
- Reviewer 检查无证据声明（`reviewer.py`）
- Eval 统计 `hallucination_rate`

**对应文件**：`extract.py`、`reviewer.py`、`eval/analyze.py`

---

## 五、数据模型与工程涉及的概念

### 14. Pydantic（数据校验）

**是什么**：Python 的类型校验库。定义数据模型，自动校验类型、做序列化。

**在项目中**：所有的 PaperMetadata、Claim、ComparisonTable 等都是 Pydantic `BaseModel`，保证数据格式不出错。

**对应文件**：`models/paper.py`、`models/claim.py`、`models/comparison.py`、`models/trace.py`

### 15. 异步编程（async/await）

**是什么**：Python 并发的方式。等 I/O（网络请求、文件读写）时干别的事，不阻塞。

**在项目中**：
- `search_papers` 用 `asyncio.gather()` 并发调 arXiv + S2
- `download_pdf` 用 `httpx.AsyncClient` 并发下载多篇 PDF
- LangGraph 所有节点都是 `async def`

**对应文件**：所有 `async def` 函数

### 16. Singleton Pattern（单例模式）

**是什么**：全局只有一个实例，避免重复初始化。

**在项目中**：`get_llm()` 是全局单例（且线程安全），整个 Agent 共享一个 LLM 客户端。

**对应文件**：`llm/client.py` 底部

---

## 六、Eval（评测）涉及的概念

### 17. Eval（模型评测）

**是什么**：系统性地评估 AI 系统表现，而不是「看着像就行」。

**5 个核心指标**：

| 指标 | 含义 | 怎么算 |
|:---|:---|:---|
| `paper_relevance@k` | 前 k 篇论文是否相关 | 人工标注 |
| `citation_precision` | 结论引用是否真的支持该说法 | claim 有 evidence 的比例 |
| `coverage_score` | 是否覆盖方法、数据、实验、局限 | 出现了几种 claim_type |
| `hallucination_rate` | 无证据断言占比 | claim 无 evidence 的比例 |
| `avg_latency / avg_steps` | 平均耗时 / 步数 | 从 trace 统计 |

**对应文件**：`eval/analyze.py`

### 18. Trace / Observability（追踪/可观测性）

**是什么**：记录 Agent 执行的每一步（什么时候开始、结束、是否有错、耗时多少）。

**在项目中**：每个节点的返回值里都有 `TraceEvent`，最终写入 `trace.jsonl`。

**对应文件**：`models/trace.py`、`store/output_writer.py`

---

## 七、PDF 处理涉及的概念

### 19. PDF Parsing（PDF 解析）

**是什么**：从 PDF 文件中提取文本、识别标题和段落结构。

**在项目中**：
- PyMuPDF 提取每页文本
- 通过字号（font-size）和粗体（bold）检测标题 → 分段
- 保留页码，用于证据引用

**对应文件**：`pdf.py`

---

## 八、摘要：你从这个项目能学到什么

| AI 领域 | 具体技能 | 简历关键词 |
|:---|:---|:---|
| **Agent 开发** | LangGraph 图编排、多节点 Pipeline | LangGraph、Agent Architecture |
| **RAG** | 检索→排序→阅读→生成全链路 | RAG、Multi-source Retrieval |
| **Prompt Engineering** | System Prompt 设计、结构化输出 | Prompt Engineering、JSON Mode |
| **LLM 集成** | 多 Provider 切换、API 调用、错误处理 | LLM Integration、Multi-provider |
| **证据与可追溯性** | Claim 抽取、证据引用、幻觉检测 | Grounding、Citation、Hallucination |
| **工程能力** | async/await、Pydantic、单例、配置管理 | Python、AsyncIO、Type Safety |
| **评测** | Eval 体系设计、5 维度指标 | LLM Evaluation、Benchmarking |
