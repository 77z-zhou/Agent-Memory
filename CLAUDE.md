# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Agent-Memory is a LangChain-based reimplementation of [MemoryOS](https://arxiv.org/pdf/2506.06326), adapted for LangGraph's `BaseStore` interface. It implements a three-tier memory hierarchy inspired by human cognitive systems: short-term (working memory), mid-term (episodic memory), and long-term (semantic memory) with automatic consolidation and retrieval.

## Architecture

The system uses a hierarchical memory architecture with automatic transfer between tiers:

- **Short-Term Memory** (STM): FIFO queue storing recent conversation history (default: 10 messages)
- **Mid-Term Memory** (MTM): Topic-segmented episodic memory with heat-based consolidation
- **Long-Term Memory** (LTM): Persistent semantic knowledge (user profile, user knowledge, assistant knowledge)

### Key Components

**`memoryos.py`** - Main entry point implementing LangGraph's `BaseStore`
- Coordinates all memory systems
- Public API: `save_memory(user_id, session_id, user_query, assistant_response)` and `search_memory(user_id, query, top_k)`

**`short_term.py`** - FIFO queue with automatic overflow triggering transfer to mid-term

**`mid_term.py`** - Segmented episodic memory
- **Segments**: Topic-grouped conversations with metadata (summary, keywords, heat, visit frequency)
- **Pages**: Individual conversation turns with prev/next pointers (conversation chains)
- Uses semantic similarity + keyword overlap for segment insertion
- LFU eviction with heat-based protection

**`long_term.py`** - Semantic knowledge base
- 90-dimension user personality profile
- User knowledge (facts/preferences) and assistant knowledge (learned capabilities)
- FAISS-based vector similarity search

**`updater.py`** - Memory transfer orchestration
- STM → MTM: Converts conversations to pages, checks continuity, generates summaries, inserts into segments
- MTM → LTM: Hot segments trigger parallel profile analysis and knowledge extraction

**`retriever.py`** - Parallel retrieval across all tiers using `ThreadPoolExecutor`

**`multitask_llm.py`** - LLM task orchestration with structured outputs (summarization, continuity detection, profile analysis, knowledge extraction)

**`storage/`** - Pluggable storage backend
- `BaseStorage` interface for implementing custom backends
- `FileStorage` provides JSON-based persistence

## Memory Transfer Flow

```
save_memory() → STM.add_memory()
    ↓ (when STM full)
STM.pop_oldest() → Updater.process_short_term_to_mid_term()
    ↓ (create page, check continuity, generate summary)
MTM.insert_page() → segment matching via similarity/keywords
    ↓ (when segment heat exceeds threshold)
Updater.process_hot_segment() → parallel:
  - UserProfileAnalysis (90-dim profile)
  - KnowledgeExtraction (user facts + assistant knowledge)
    ↓
LTM.update()
```

## Heat Calculation

Segments accumulate "heat" from interactions: `H = α·N_visit + β·L_interaction + γ·R_recency`

- High heat → triggers consolidation to long-term memory
- Prevents LFU eviction of important segments
- Resets after consolidation

## Key Algorithms

**Conversation Continuity**: LLM determines if conversations are semantically continuous, creating page chains across multiple turns.

**Segment Insertion**: Hybrid similarity scoring combining semantic similarity (embeddings) + keyword overlap (Jaccard similarity)

**Parallel Retrieval**: Simultaneous queries to STM, MTM, and LTM with combined scoring

## Usage Example

```python
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from memory_core.memoryos import MemoryOS

llm = init_chat_model("deepseek-chat")
embedding = HuggingFaceEmbeddings(model_name="path/to/embedding/model")

mem = MemoryOS(
    llm=llm,
    embedding_fuc=embedding,
    storage_dir="./memory_storage",
    short_term_max_capacity=10
)

# Save conversation
mem.save_memory(
    user_id="user123",
    session_id="session456",
    user_query="What is the capital of France?",
    assistant_response="The capital of France is Paris."
)

# Search memory
results = mem.search_memory(user_id="user123", query="French geography", top_k=5)
```

## Configuration Parameters

- `short_term_max_capacity`: STM FIFO size (default: 10)
- `mid_term_max_capacity`: MTM segment/page limit (default: 2000)
- `topic_similarity_threshold`: Segment grouping threshold (default: 0.6)
- `mid_term_heat_threshold`: Heat level triggering consolidation (default: 5.0)
- `knowledge_capacity`: LTM knowledge deque size (default: 100)
- `retrieval_page_topk`: Top-K results per tier (default: 7)

## Storage Structure

File storage creates:
```
storage_dir/
├── short_term/{user_id}.json
├── mid_term/{user_id}.json
└── long_term/{user_id}.json
```

## Dependencies

Core dependencies (infer from imports):
- `langchain-core` - LLM abstractions
- `langgraph` - BaseStore interface
- `langchain-chat-models` - Model initialization
- `langchain-huggingface` - Embeddings
- `faiss-cpu` - Vector similarity
- `numpy` - Numerical operations
- `pydantic` - Structured outputs

## Development Notes

- The project is not yet a proper Python package (no `__init__.py`, `setup.py`, or `requirements.txt`)
- Currently requires direct import paths: `from memory_core.memoryos import MemoryOS`
- LLM calls use structured outputs with Pydantic models defined in `multitask_llm.py`
- Heat calculation weights and thresholds are currently hardcoded in `mid_term.py`
- The system assumes embedding dimensions are consistent across all operations
