# Agent-Memory

<div align="center">

**A LangChain-based Reimplementation of MemoryOS with LangGraph Store Interface**

[![Paper](https://img.shields.io/badge/Paper-arXiv:2506.06326-b31b1b)](https://arxiv.org/pdf/2506.06326)
[![Original](https://img.shields.io/badge/Original-BAI--LAB-blue)](https://github.com/BAI-LAB/MemoryOS)

[English](#english) | [中文](#中文)

</div>

---

## 中文

### 项目简介

Agent-Memory 是对论文 [MemoryOS](https://arxiv.org/pdf/2506.06326) 的 LangChain 复现实现，专门适配了 LangGraph 的 `BaseStore` 接口。该项目实现了一个受人类认知系统启发的三层记忆架构：

- **短期记忆 (Short-Term Memory)**: 基于 FIFO 队列的工作记忆，存储最近的对话历史
- **中期记忆 (Mid-Term Memory)**: 基于主题分段的情景记忆，支持语义检索和热度管理
- **长期记忆 (Long-Term Memory)**: 持久化的语义知识，包括用户画像和知识提取

### 核心特性

- **三层记忆架构**: 模拟人类认知系统的记忆分层机制
- **自动记忆巩固**: 基于热度 (Heat) 的记忆自动转移和整合
- **语义检索**: 结合嵌入向量和关键词匹配的混合检索
- **对话连续性检测**: 自动关联多轮对话，构建对话链
- **并行处理**: 使用多线程并行检索和知识提取
- **LangGraph 集成**: 完全兼容 LangGraph 的存储接口
- **可插拔存储**: 支持自定义存储后端（默认文件存储）

### 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                         用户输入                              │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                    短期记忆 (STM)                             │
│  - FIFO 队列 (默认 10 条)                                     │
│  - 存储最近对话历史                                           │
└──────────────────────────┬──────────────────────────────────┘
                           ↓ 队列满时触发转移
┌─────────────────────────────────────────────────────────────┐
│                    中期记忆 (MTM)                             │
│  - 主题分段存储 (Segments + Pages)                           │
│  - 语义相似度 + 关键词匹配分组                                │
│  - LFU 淘汰 + 热度保护                                        │
│  - Page Chain 对话链                                         │
└──────────────────────────┬──────────────────────────────────┘
                           ↓ 热度超过阈值时触发
┌─────────────────────────────────────────────────────────────┐
│                    长期记忆 (LTM)                             │
│  - 90 维用户画像                                             │
│  - 用户知识 (User Knowledge)                                 │
│  - 助手知识 (Assistant Knowledge)                            │
│  - FAISS 向量检索                                            │
└─────────────────────────────────────────────────────────────┘
```

### 安装

```bash
# 克隆仓库
git clone https://github.com/your-username/Agent-Memory.git
cd Agent-Memory

# 安装依赖
pip install langchain-core langgraph langchain-chat-models
pip install langchain-huggingface faiss-cpu numpy
```

### 快速开始

```python
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from memory_core.memoryos import MemoryOS

# 初始化 LLM 和嵌入模型
llm = init_chat_model("deepseek-chat")  # 或其他支持的语言模型
embedding = HuggingFaceEmbeddings(
    model_name="path/to/your/embedding/model"
)

# 创建 MemoryOS 实例
mem = MemoryOS(
    llm=llm,
    embedding_fuc=embedding,
    storage_dir="./memory_storage",
    short_term_max_capacity=10,
    mid_term_max_capacity=2000,
    mid_term_heat_threshold=5.0
)

# 保存对话记忆
mem.save_memory(
    user_id="user123",
    session_id="session456",
    user_query="什么是人工智能？",
    assistant_response="人工智能（AI）是计算机科学的一个分支..."
)

# 搜索相关记忆
results = mem.search_memory(
    user_id="user123",
    query="AI 相关概念",
    top_k=5
)

print(results)
# {
#     "short_term_memory": [...],
#     "mid_term_memory": [...],
#     "user_background_context": [...],
#     "assistant_knowledge": [...],
#     "conversation_metadata": {...}
# }
```

### 核心组件

| 组件 | 文件 | 功能 |
|------|------|------|
| MemoryOS | `memoryos.py` | 主入口，实现 LangGraph BaseStore 接口 |
| 短期记忆 | `short_term.py` | FIFO 队列管理最近对话 |
| 中期记忆 | `mid_term.py` | 主题分段和情景记忆 |
| 长期记忆 | `long_term.py` | 用户画像和知识管理 |
| 记忆更新器 | `updater.py` | 记忆在各层间的转移和整合 |
| 记忆检索器 | `retriever.py` | 并行检索三层记忆 |
| 多任务 LLM | `multitask_llm.py` | LLM 任务编排 |
| 存储层 | `storage/` | 可插拔的存储后端 |

### 配置参数

```python
MemoryOS(
    llm=BaseLanguageModel,              # LLM 实例
    embedding_fuc=Embeddings,            # 嵌入模型
    storage_dir="./memory_storage",      # 存储目录

    # 短期记忆配置
    short_term_max_capacity=10,          # STM 队列最大容量

    # 中期记忆配置
    mid_term_max_capacity=2000,          # MTM 最大容量
    topic_similarity_threshold=0.6,      # 主题分组相似度阈值
    mid_term_heat_threshold=5.0,         # 触发巩固的热度阈值

    # 长期记忆配置
    knowledge_capacity=100,              # 知识队列容量

    # 检索配置
    retrieval_page_topk=7                # 每层检索返回的 Top-K 数量
)
```

### 记忆热度计算

中期记忆的 Segment 通过以下公式计算热度：

```
H = α·N_visit + β·L_interaction + γ·R_recency
```

- `N_visit`: 访问次数
- `L_interaction`: 交互长度
- `R_recency`: 时间衰减因子

当热度超过阈值时，触发知识提取并转移到长期记忆。

### 存储结构

```
storage_dir/
├── short_term/
│   └── {user_id}.json      # 短期记忆
├── mid_term/
│   └── {user_id}.json      # 中期记忆
└── long_term/
    └── {user_id}.json      # 长期记忆
```

### LangGraph 集成示例

```python
from langgraph.graph import StateGraph
from langgraph.store.memory import InMemoryStore
from memory_core.memoryos import MemoryOS

# 使用 MemoryOS 替代默认存储
memory_store = MemoryOS(
    llm=llm,
    embedding_fuc=embedding,
    storage_dir="./agent_memory"
)

# 在 LangGraph 中使用
graph = StateGraph(state)
# ... 配置 graph ...
```

### 原论文

本项目复现自以下论文：

```bibtex
@article{memoryos2025,
  title={MemoryOS: Building Long-term Memory for AI Assistants with Three-Stage Memory System},
  author={...},
  journal={arXiv preprint arXiv:2506.06326},
  year={2025}
}
```

- [论文地址](https://arxiv.org/pdf/2506.06326)
- [原始实现](https://github.com/BAI-LAB/MemoryOS)

### 许可证

MIT License

### 贡献

欢迎提交 Issue 和 Pull Request！

---

## English

### Overview

Agent-Memory is a LangChain-based reimplementation of [MemoryOS](https://arxiv.org/pdf/2506.06326), adapted for LangGraph's `BaseStore` interface. It implements a three-tier memory architecture inspired by human cognitive systems:

- **Short-Term Memory (STM)**: FIFO queue-based working memory for recent conversations
- **Mid-Term Memory (MTM)**: Topic-segmented episodic memory with semantic retrieval and heat management
- **Long-Term Memory (LTM)**: Persistent semantic knowledge including user profiles and extracted knowledge

### Key Features

- **Three-Tier Memory Architecture**: Mimics human cognitive memory systems
- **Automatic Memory Consolidation**: Heat-based memory transfer and integration
- **Semantic Retrieval**: Hybrid search combining embeddings and keyword matching
- **Conversation Continuity Detection**: Automatically links multi-turn conversations
- **Parallel Processing**: Multi-threaded retrieval and knowledge extraction
- **LangGraph Integration**: Full compatibility with LangGraph's store interface
- **Pluggable Storage**: Custom storage backends supported (default: file-based)

### Quick Start

```python
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from memory_core.memoryos import MemoryOS

# Initialize LLM and embedding model
llm = init_chat_model("deepseek-chat")
embedding = HuggingFaceEmbeddings(
    model_name="path/to/embedding/model"
)

# Create MemoryOS instance
mem = MemoryOS(
    llm=llm,
    embedding_fuc=embedding,
    storage_dir="./memory_storage"
)

# Save conversation memory
mem.save_memory(
    user_id="user123",
    session_id="session456",
    user_query="What is AI?",
    assistant_response="Artificial Intelligence is a branch of computer science..."
)

# Search relevant memories
results = mem.search_memory(
    user_id="user123",
    query="AI concepts",
    top_k=5
)
```

### Core Components

| Component | File | Description |
|-----------|------|-------------|
| MemoryOS | `memoryos.py` | Main entry, implements LangGraph BaseStore |
| Short-Term | `short_term.py` | FIFO queue for recent conversations |
| Mid-Term | `mid_term.py` | Topic-segmented episodic memory |
| Long-Term | `long_term.py` | User profile and knowledge management |
| Updater | `updater.py` | Memory transfer and consolidation |
| Retriever | `retriever.py` | Parallel retrieval across all tiers |
| Storage | `storage/` | Pluggable storage backend |

### Original Paper

This project is a reimplementation of:

```bibtex
@article{memoryos2025,
  title={MemoryOS: Building Long-term Memory for AI Assistants with Three-Stage Memory System},
  journal={arXiv preprint arXiv:2506.06326},
  year={2025}
}
```

- [Paper](https://arxiv.org/pdf/2506.06326) | [Original Implementation](https://github.com/BAI-LAB/MemoryOS)

### License

MIT License

