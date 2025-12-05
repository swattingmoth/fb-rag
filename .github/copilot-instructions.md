# AI Coding Agent Instructions for RAG Project

## Project Overview
This is a Retrieval-Augmented Generation (RAG) application that combines information retrieval with language models to provide context-aware responses. The project is currently in initialization phase.

## Architecture (To Be Defined)
Update this section as the project grows with:
- **Document Storage**: How documents are ingested and indexed
- **Retrieval Layer**: Vector embeddings and search mechanisms
- **Generation Layer**: LLM integration and prompt management
- **API Layer**: Request handling and response formatting

## Setup & Development Workflow

### Initial Setup
1. Clone the repository
2. Install dependencies (specify package manager and key packages)
3. Configure environment variables (add `.env.example` when needed)
4. Run initial tests to verify setup

### Building & Running
- **Build**: `[insert build command]`
- **Run**: `[insert run command]`
- **Tests**: `[insert test command]`
- **Linting**: `[insert lint command]`

### Key Commands (Add as Project Develops)
Document critical commands that aren't obvious from file inspection, such as:
- Database migrations or setup
- Vector index generation
- Document ingestion scripts
- Configuration validation

## Code Conventions & Patterns

### Project Structure
- **Source Code**: Specify main language/framework location (e.g., `src/`, `lib/`)
- **Tests**: Location and structure conventions
- **Configuration**: Where environment and app config lives
- **Documentation**: API docs, deployment guides, architectural diagrams

### Language/Framework Specifics
- **Primary Language**: Python 3.9+
- **Key Dependencies**: LangChain, LlamaIndex, OpenAI SDK, or similar
- **Document Format**: Handling of PDFs, markdown, plain text, etc.
- **Vector Store**: Pinecone, Weaviate, Qdrant, FAISS, or similar

### Python Development Guidelines

#### Code Style & Quality
- **Format**: Follow [PEP 8](https://pep8.org/) with 100-character line limit
- **Linting**: Use `ruff` for fast linting and `black` for code formatting
- **Type Hints**: Use type annotations throughout for better IDE support and maintainability
  ```python
  from typing import List, Optional
  
  def retrieve_documents(query: str, top_k: int = 5) -> List[Dict[str, str]]:
      """Retrieve top-k documents matching the query."""
  ```
- **Docstrings**: Use Google-style docstrings for modules, functions, and classes
  ```python
  def embed_text(text: str) -> List[float]:
      """Convert text to embeddings.
      
      Args:
          text: The input text to embed.
          
      Returns:
          A list of embedding values.
          
      Raises:
          ValueError: If text is empty.
      """
  ```

#### Documenting Complex Logic
- Use inline comments to explain the "why" behind complex algorithms, especially in:
  - Document chunking and overlap strategies
  - Retrieval ranking and scoring logic
  - Prompt engineering decisions
  - Token limit calculations
  ```python
  # Split documents into overlapping chunks to preserve context across boundaries
  # Overlap of 100 tokens ensures semantic continuity for retrieval
  chunk_size = 512
  overlap = 100
  chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size-overlap)]
  ```
- Document architectural decisions in module docstrings and README
- Use block comments for non-obvious design patterns or workarounds

#### Project Structure
```
src/
├── __init__.py
├── config.py           # Configuration and environment variables
├── retrieval/          # Document retrieval logic
│   ├── __init__.py
│   ├── loader.py       # Document loading/parsing
│   ├── embedder.py     # Embedding generation
│   └── retriever.py    # Vector search and retrieval
├── generation/         # LLM and response generation
│   ├── __init__.py
│   ├── prompts.py      # Prompt templates
│   └── llm.py          # LLM interactions
├── rag.py             # Main RAG pipeline orchestration
└── utils/             # Shared utilities
    ├── __init__.py
    ├── logging.py
    └── helpers.py

tests/
├── __init__.py
├── test_retrieval.py
├── test_generation.py
└── test_rag.py
```

#### Dependencies Management
- **Requirements**: Use `requirements.txt` or `pyproject.toml` with pinned versions for reproducibility
- **Development Dependencies**: Maintain separate dev requirements for testing, linting, and formatting
  ```
  # requirements.txt
  langchain==0.x.x
  openai==1.x.x
  numpy==1.x.x
  
  # requirements-dev.txt
  pytest==7.x.x
  ruff==0.x.x
  black==23.x.x
  ```

#### Virtual Environment
- Use Anaconda for environment management with the `llmdev` environment
- Create/activate with: `conda activate llmdev`
- If environment doesn't exist, create it: `conda create -n llmdev python=3.9`
- Install dependencies: `conda install -r requirements.txt` or `pip install -r requirements.txt` within the activated environment
- Document Anaconda setup in project README

#### Testing & Error Handling
- **Unit Tests**: Test individual components (retrieval, embedding, prompt formatting) using `unittest`
  ```python
  import unittest
  
  class TestEmbedder(unittest.TestCase):
      def test_embed_text_returns_vector(self):
          embedder = TextEmbedder()
          result = embedder.embed("test text")
          self.assertIsInstance(result, list)
          self.assertGreater(len(result), 0)
  ```
- **Integration Tests**: Test end-to-end RAG workflow with sample documents
- **Error Handling**: Catch and log specific exceptions; provide context in error messages
  ```python
  try:
      embeddings = model.embed(text)
  except ConnectionError as e:
      logger.error(f"Failed to connect to embedding service: {e}")
      raise
  ```

#### Logging
- Use Python's `logging` module; avoid `print()` for debug info
  ```python
  import logging
  logger = logging.getLogger(__name__)
  logger.info(f"Retrieved {len(docs)} documents for query: {query}")
  ```
- Configure logging levels in config module
- Log important operations: document loading, embedding requests, LLM calls

#### Async Patterns (if applicable)
- Use `asyncio` for concurrent operations (batch embedding, parallel retrievals)
- Prefer `async/await` syntax over callbacks
- Document which functions are async and why

### RAG-Specific Patterns
- **Prompt Templates**: Location and customization approach
- **Context Window Management**: How chunks are sized and retrieved
- **Source Attribution**: How original documents are referenced in responses
- **Error Handling**: Fallback behavior when retrieval fails

## Integration Points

### External Dependencies
- Document repositories/storage (S3, local filesystem, etc.)
- Vector databases or embedding services
- LLM providers (OpenAI, Anthropic, Hugging Face, etc.)
- Authentication mechanisms

### Cross-Component Communication
- How retrieval results feed into the LLM prompt
- Async/sync patterns for document processing
- Error propagation and recovery strategies

## Contributing Guidelines

### Before Making Changes
1. Check existing patterns in similar files
2. Understand the data flow from document ingestion to response generation
3. Verify any new dependencies are approved

### Testing Expectations
- Unit tests for retrieval logic
- Integration tests for end-to-end RAG workflows
- Validation of response quality (when applicable)

### Code Review Focus Areas
- Efficient retrieval without excessive token usage
- Proper source attribution in generated responses
- Security of API keys and sensitive configurations
- Performance of document processing pipeline

---

**Status**: Project structure under development. Add specific patterns and tools as they're implemented.
