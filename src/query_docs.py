import json
from collections.abc import Iterable
from typing import Optional
from venv import logger

from fastembed import LateInteractionTextEmbedding, SparseTextEmbedding
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from qdrant_client import QdrantClient
from qdrant_client import QdrantClient, models
from src.store_docs import (
    OllamaTextEmbedding,
    create_client,
    load_and_store_documents,
)
from src import config as c


def create_rag_prompt() -> PromptTemplate:
    """Create a prompt template for RAG-based Facebook post Q&A.

    The prompt instructs the LLM to answer questions about Facebook posts
    using the provided context, maintaining an informal and factual tone
    while referencing dates and post titles.

    Returns:
        PromptTemplate: Configured prompt for RAG pipeline.
    """
    template = """You are a helpful assistant answering questions about someone's Facebook posts and memories.

Based on the following Facebook posts, answer the user's question in a casual, friendly, and factual tone.
Always mention the dates and post titles when relevant to give context about when things happened.

Facebook Posts Context:
{context}

Question: {question}

Answer:"""

    return PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )


def format_retrieved_docs(documents: list[Document]) -> str:
    """Format retrieved documents for inclusion in the RAG prompt.

    Structures document content and metadata for the LLM to reference,
    preserving dates, titles, and source information.

    Args:
        documents: List of retrieved Document objects from vector store.

    Returns:
        Formatted string representation of documents.
    """
    if not documents:
        return "No relevant posts found."

    formatted = []
    for i, doc in enumerate(documents, 1):
        meta = doc.metadata
        timestamp = meta.get("timestamp", "unknown date")
        title = meta.get("title", "(no title)")
        post_type = meta.get("post_type", "post")

        entry = f"Post {i}:\n"
        entry += f"  Date: {timestamp}\n"
        entry += f"  Title: {title}\n"
        entry += f"  Type: {post_type}\n"
        entry += f"  Content: {doc.page_content}\n"
        formatted.append(entry)

    return "\n".join(formatted)


def create_rag_response(
    question: str,
    retrieved_docs: list[Document],
    answer_text: str,
) -> dict:
    """Create a JSON response object suitable for frontend consumption.

    Structures the RAG response with the generated answer and source citations.

    Args:
        question: The original user question.
        retrieved_docs: Documents retrieved from vector store for context.
        answer_text: The LLM-generated answer text.

    Returns:
        Dictionary with answer, sources, and metadata for JSON serialization.
    """
    sources = []
    for doc in retrieved_docs:
        sources.append(
            {
                "date": doc.metadata.get("timestamp", "unknown"),
                "title": doc.metadata.get("title", "(no title)"),
                "post_type": doc.metadata.get("post_type", "post"),
                "urls": doc.metadata.get("urls", []),
            }
        )

    return {
        "question": question,
        "answer": answer_text.strip(),
        "sources": sources,
        "num_sources": len(sources),
    }


def prompt_query(
    client: QdrantClient,
    dense_model: OllamaTextEmbedding,
    sparse_model: SparseTextEmbedding,
    late_interaction_model: LateInteractionTextEmbedding,
    config: c.Config,
) -> None:
    """
    Interactive RAG query loop powered by hybrid search and Ollama LLM.

    Uses dense, sparse, and late-interaction embeddings to retrieve relevant
    Facebook posts, then generates context-aware answers using qwen3:14b.
    Responses are returned as JSON with source citations.

    Parameters:
        client (QdrantClient): Qdrant client for vector search.
        dense_model (OllamaTextEmbedding): Model for dense embeddings.
        sparse_model (SparseTextEmbedding): Model for sparse embeddings.
        late_interaction_model (LateInteractionTextEmbedding): Model for late interaction embeddings.
        config (Config): Configuration object with application settings.

    Returns:
        None
    """
    default_k = 7
    print("\nFacebook Posts RAG Q&A")
    print("  - Ask questions about your Facebook posts and memories")
    print("  - Type 'exit' to quit")
    print("  - Type 'help' for options\n")

    # Initialize LLM and prompt
    try:
        llm = ChatOllama(
            model=config.model,
            temperature=0.7,
            top_p=0.9,
        )
    except Exception as e:
        print(
            f"Error: Failed to connect to Ollama. Make sure {config.model} is running."
        )
        logger.error(f"Error connecting to {config.model}: {e}")
        raise

    rag_prompt = create_rag_prompt()
    parser = StrOutputParser()

    while True:
        user_input = input("\nEnter your query (or 'help' / 'exit'): ").strip()

        if user_input.lower() == "exit":
            print("Goodbye.")
            break

        if user_input.lower() == "help":
            print(
                """
Ask natural language questions about your Facebook posts. Examples:
  what did I post about hiking?
  show me my favorite pizza posts
  what did I share in 2024?
  tell me about my vacation memories
  
The RAG pipeline will find relevant posts and answer your question
with citations showing dates and post titles.
            """
            )
            continue

        # Parse query
        query = user_input

        print(f"\nQuestion: {query}")
        print("Searching your posts...")
        logger.info(f"Searching documents with query: {query}")

        try:
            # Perform hybrid search using all three embedding models
            dense_vectors = dense_model.query_embed(query)[0]  # type: ignore
            sparse_vectors = next(sparse_model.query_embed(query))  # type: ignore
            late_vectors = next(late_interaction_model.query_embed(query))  # type: ignore

            prefetch = [
                models.Prefetch(
                    query=dense_vectors,
                    using="dense",
                    limit=default_k,
                ),
                models.Prefetch(
                    query=models.SparseVector(**sparse_vectors.as_object()),
                    using="bm25",
                    limit=default_k,
                ),
            ]

            results = client.query_points(
                collection_name=config.collection_name,
                prefetch=prefetch,
                query=late_vectors,
                using="colbertv2.0",
                limit=default_k,
                with_payload=True,
            )
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            raise

        logger.info("Finished document retrieval")

        if not results or not results.points:  # type: ignore[index]
            # No results found, still try to generate a response
            response = create_rag_response(
                question=query,
                retrieved_docs=[],
                answer_text="I couldn't find any relevant Facebook posts matching your question.",
            )
            print(json.dumps(response, indent=2))
            continue

        # Convert Qdrant results to LangChain Documents for RAG
        retrieved_docs = []
        for group in results:
            for i, scored_point in enumerate(group[1]):
                payload = scored_point.payload
                logger.info(f"Retrieved doc {i+1}:\n {json.dumps(payload, indent=2)}")
                doc = Document(
                    page_content=payload.get("document", ""),
                    metadata={
                        "timestamp": payload.get("timestamp", ""),
                        "title": payload.get("title", ""),
                        "post_type": payload.get("post_type", "post"),
                        "urls": payload.get("urls", []),
                        "score": scored_point.score,
                    },
                )
                retrieved_docs.append(doc)

        # Format context for the RAG prompt
        context_text = format_retrieved_docs(retrieved_docs)

        # Generate answer using RAG pipeline
        try:
            print("Generating answer...")
            chain = rag_prompt | llm | parser
            answer = chain.invoke(
                {
                    "context": context_text,
                    "question": query,
                }
            )
        except Exception as e:
            print(f"Error: LLM generation failed. Make sure Ollama is running.")
            print(f"Details: {e}")
            raise

        # Create and display JSON response
        response = create_rag_response(
            question=query,
            retrieved_docs=retrieved_docs,
            answer_text=answer,
        )
        print(response["answer"])
        logger.info(f"Generated response:\n {json.dumps(response, indent=2)}")


if __name__ == "__main__":
    config = c.Config()
    c.configure_logging(config.log_folder + "/query_docs.log", log_to_console=False)
    client, dense_model, sparse_model, late_interaction_model = create_client(config)

    prompt_query(client, dense_model, sparse_model, late_interaction_model, config)

    client.close()
