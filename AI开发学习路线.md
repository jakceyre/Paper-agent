# AI 开发工程师学习路线（零基础 → 入职）

> 目标岗位：AI 应用开发工程师、RAG 工程师、AI Agent 开发工程师

---

## 总览：四个阶段，约 6-12 个月

```
阶段一（2 个月）       阶段二（2-3 个月）      阶段三（2-3 个月）       阶段四（持续）
   Python 基础     →    机器学习基础      →    LLM & Agent 开发   →   项目 + 面试
   编程 + 工具          神经网络原理            RAG / Agent / Prompt    Paper Agent 等
```

---

## 阶段一：Python 基础（2 个月）

### 学习内容

| 主题 | 具体内容 | 资源 |
|:---|:---|:---|
| Python 语法 | 变量、类型、条件、循环、函数、类 | [Python官方教程](https://docs.python.org/zh-cn/3/tutorial/) |
| 数据结构 | list, dict, set, tuple 的选择和操作 | 同上 |
| 文件 IO | 读写文件、JSON、CSV 处理 | |
| 命令行 | pip、venv、git 基础 | |
| 异步编程 | async/await（Agent 开发必备） | [Real Python - AsyncIO](https://realpython.com/async-io-python/) |

### 检验标准
- 能用 Python 写一个命令行小工具
- 会用 `pip install` 装包、用 `venv` 建虚拟环境
- 看项目里的 `pyproject.toml` 知道在干什么

---

## 阶段二：机器学习基础（2-3 个月）

### 学习内容

| 主题 | 具体内容 | 重要度 |
|:---|:---|:---:|
| **NumPy / Pandas** | 数组操作、数据处理 | ⭐⭐⭐ |
| **机器学习基本概念** | 监督学习、无监督学习、训练/测试集、过拟合 | ⭐⭐⭐ |
| **常见模型** | 线性回归、决策树、随机森林 | ⭐⭐ |
| **深度学习入门** | 神经网络、反向传播、梯度下降 | ⭐⭐⭐ |
| **Transformer 架构** | Self-Attention、Encoder-Decoder（面试必问） | ⭐⭐⭐⭐⭐ |
| **PyTorch 基础** | Tensor、Dataset、DataLoader、训练循环 | ⭐⭐⭐ |

### 核心资源

- **课程**: [吴恩达《机器学习》](https://www.coursera.org/learn/machine-learning)（理论）+ [李宏毅《机器学习》2023](https://www.youtube.com/@HungyiLeeNTU)（生动）
- **Transformer**: 读原文 [Attention Is All You Need](https://arxiv.org/abs/1706.03762) + [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)（图解版，必看）
- **PyTorch**: [PyTorch 官方教程](https://pytorch.org/tutorials/) 60 分钟入门

### 检验标准
- 能解释：Transformer 的 Self-Attention 是什么，Q/K/V 分别代表什么
- 能用 PyTorch 跑一个分类任务

---

## 阶段三：LLM & Agent 开发（2-3 个月）⭐ 核心

### 这部分直接对应 Paper Agent 项目

| 主题 | 具体内容 | 对应 Paper Agent |
|:---|:---|:---|
| **LLM 基础** | GPT/Claude 原理、Token、Prompt 概念 | `llm/client.py` |
| **Prompt Engineering** | System Prompt、Few-shot、Chain-of-Thought | `planner.py`、`extract.py`、`synthesizer.py` |
| **RAG（检索增强生成）** | 向量检索、Embedding、检索→生成流程 | 整个项目的核心模式 |
| **Agent 架构** | Tool Use、ReAct、Plan-Execute | `graph.py` 10 节点 pipeline |
| **LangChain / LangGraph** | StateGraph、Node、Edge、ConditionalEdge | `graph.py`、`state.py` |
| **Function Calling / Tool Use** | LLM 调用外部 API、结构化输出 | `search.py`、`compare.py` |
| **Pydantic 结构化输出** | 用 Pydantic 定义数据模型，LLM 输出 JSON | `models/` 整个目录 |

### 核心资源

- **Prompt Engineering**: [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- **LangGraph**: [LangGraph 官方教程](https://langchain-ai.github.io/langgraph/tutorials/)（必看 3-5 个 tutorial）
- **RAG**: [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)
- **Agent**: [Anthropic Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)（必读）
- **DeepSeek API**: [DeepSeek API 文档](https://platform.deepseek.com/api-docs/)

### 检验标准
- 能解释 RAG 的完整流程：检索 → 排序 → 阅读 → 生成
- 能用 LangGraph 画一个 3 节点的 Agent
- 能写 Prompt 让 LLM 输出指定的 JSON 格式

---

## 阶段四：项目 + 面试准备

### 简历项目建议

1. **Paper Agent**（你正在做的）— 覆盖 Agent、RAG、Prompt Engineering
2. **RAG 问答系统** — 一个简单的文档 QA 机器人（用 LangChain 或纯 Python）
3. **Prompt 优化工具** — 对比不同 Prompt 的效果，自动迭代

### 面试高频题

| 类别 | 问题示例 |
|:---|:---|
| Transformer | "Self-Attention 的计算公式是什么？为什么需要缩放？" |
| LLM | "GPT 是怎么生成下一个 token 的？什么是温度参数？" |
| RAG | "RAG 的流程是什么？检索阶段用什么方法？怎么评估检索质量？" |
| Agent | "什么是 ReAct？相比直接生成有什么优势？" |
| Prompt | "怎么让 LLM 稳定输出 JSON？有哪些技巧？" |
| 系统设计 | "设计一个论文推荐系统 / 客服 Agent / 代码审查 Agent" |
| Python | "async/await 是什么？list 和 dict 的内部实现？" |

### 学习节奏建议

```
每天 2-3 小时：
  - 1 小时：看课程/论文
  - 1 小时：写代码/做 Paper Agent
  - 30 分钟：整理笔记

每周：
  - 复盘学到的东西，写成文章或笔记
  - 读 1 篇论文摘要（arXiv 上找感兴趣的）
```

---

## 推荐的完整资源包

| 类别 | 资源 | 投入时间 |
|:---|:---|:---|
| **Python** | Python官方教程 + Real Python 文章 | 1-2 月 |
| **ML 理论** | 吴恩达 ML 课程 + 李宏毅 2023 | 2-3 月 |
| **DL+Transformer** | Attention Is All You Need + Jay Alammar 图解 | 2 周 |
| **LLM 应用开发** | LangGraph 教程 + Anthropic Agent 文章 | 1 月 |
| **项目实战** | Paper Agent + 自己再做一个 RAG 项目 | 持续 |
| **追踪前沿** | arXiv、Hugging Face Daily Papers、Twitter/X AI 社区 | 每天 15 分钟 |

---

## 记住几点

1. **先会用，再理解原理** — 先跑通 Paper Agent，再去读 Transformer 论文
2. **面试不看你学了多少课，看你做了什么项目** — 2 个好项目 > 10 门课
3. **Python 和 Prompt Engineering 是最高频使用的技能** — 这两样要熟练
4. **Agent 开发是当下的风口** — Paper Agent 这种项目简历上很加分
