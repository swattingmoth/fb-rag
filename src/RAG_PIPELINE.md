"""RAG Pipeline Implementation for Facebook Posts Q&A

This module implements a Retrieval-Augmented Generation (RAG) pipeline for
answering natural language questions about Facebook posts using LangChain,
Ollama, and hybrid vector search.

## Overview

The RAG pipeline combines three key components:

1. **Hybrid Retrieval**: Uses Qdrant with three embedding models:
   - Dense embeddings (Qwen3 8B): Semantic similarity
   - Sparse embeddings (BM25): Keyword matching
   - Late interaction embeddings (ColBERTv2.0): Token-level interactions

2. **LLM Generation**: Qwen3 14B via Ollama for natural, conversational answers

3. **JSON Response**: Structured output with sources for frontend consumption

## Key Functions

- `create_rag_prompt()`: Creates the LLM prompt template
- `format_retrieved_docs()`: Formats retrieved documents for LLM context
- `create_rag_response()`: Builds JSON response with citations
- `prompt_query()`: Interactive query loop with RAG pipeline

## Usage

Run the interactive Q&A loop:

```bash
conda activate llmdev
python src/query_docs.py
```

## Example Interaction

```
Question: what did I post about hiking?
Searching your posts...

{
  "question": "what did I post about hiking?",
  "answer": "You shared some great hiking memories. On 2023-07-15, you posted about 
    'Mountain Trail Adventure' where you talked about the scenic views. Then in 2024-03-20, 
    you shared 'Spring Hiking Season Begins' with photos from your local trails.",
  "sources": [
    {
      "date": "2023-07-15T14:30:00",
      "title": "Mountain Trail Adventure",
      "post_type": "post",
      "urls": ["http://example.com/trail"]
    },
    {
      "date": "2024-03-20T10:15:00",
      "title": "Spring Hiking Season Begins",
      "post_type": "photo",
      "urls": []
    }
  ],
  "num_sources": 2
}
```

## JSON Response Schema

```json
{
  "question": "string - the user's original question",
  "answer": "string - the LLM-generated answer with context",
  "sources": [
    {
      "date": "ISO 8601 timestamp",
      "title": "string - Facebook post title",
      "post_type": "link|photo|video|event|status|other",
      "urls": ["list of URLs in the post"]
    }
  ],
  "num_sources": "integer - count of retrieved documents"
}
```

## Error Handling

- If Ollama is not running or qwen3:14b is unavailable, the system raises an
  error with clear messaging and exits (as specified).
- Retrieval errors are raised immediately.
- If no documents are found, a response is still generated with an appropriate message.

## Configuration

Adjust the following in `prompt_query()`:

- `default_k`: Number of documents to retrieve (default: 7)
- `temperature`: LLM creativity (0.7 = balanced)
- `top_p`: LLM sampling parameter (0.9 = diverse)

## Design Decisions

### Why Three Embedding Models?

The hybrid search leverages complementary strengths:
- **Dense**: Captures semantic meaning and context
- **Sparse**: Excellent for exact phrase and keyword matching
- **Late Interaction**: Token-level interactions for nuanced relevance

This combination ensures both semantic and keyword-based queries work well.

### Why JSON Response Format?

JSON is ideal for frontend integration:
- Structured data easy to parse and display
- Includes sources for citation/verification
- Ready for web APIs and downstream processing

### Why Raise on Ollama Errors?

Making Ollama errors fatal ensures:
- Users know immediately if the LLM isn't available
- No silent degradation or partial responses
- Clear error messages guide troubleshooting
"""
