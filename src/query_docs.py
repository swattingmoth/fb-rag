import json
from collections.abc import Iterable
from typing import Optional

from fastembed import LateInteractionTextEmbedding, SparseTextEmbedding
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from qdrant_client import QdrantClient
from qdrant_client import QdrantClient, models
from store_docs import (
    OllamaTextEmbedding,
    create_client,
    load_embed_store,
)


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
) -> None:
    """
    Interactive RAG query loop powered by hybrid search and Ollama LLM.

    Uses dense, sparse, and late-interaction embeddings to retrieve relevant
    Facebook posts, then generates context-aware answers using qwen3:14b.
    Responses are returned as JSON with source citations.
    """
    default_k = 7
    print("\nFacebook Posts RAG Q&A")
    print("  - Ask questions about your Facebook posts and memories")
    print("  - Type 'exit' to quit")
    print("  - Type 'help' for options\n")

    # Initialize LLM and prompt
    try:
        llm = ChatOllama(
            model="qwen3:14b",
            temperature=0.7,
            top_p=0.9,
        )
    except Exception as e:
        print(f"Error: Failed to connect to Ollama. Make sure qwen3:14b is running.")
        print(f"Details: {e}")
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
                collection_name="facebook_posts",
                prefetch=prefetch,
                query=late_vectors,
                using="colbertv2.0",
                limit=default_k,
                with_payload=True,
            )
        except Exception as e:
            print(f"Error during retrieval: {e}")
            raise

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
            for scored_point in group[1]:
                payload = scored_point.payload
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
        print(json.dumps(response, indent=2))


if __name__ == "__main__":
    # manual_sparse_search(
    #     db_folder_name=r"c:\users\jordan-dev\data\qdrant_db",
    #     collection_name="facebook_posts",
    #     query="pastor pat",
    #     k=10,
    # )
    # Use the Qdrant paths you defined earlier
    # retriever, client = create_document_retriever(
    #     db_folder_name=r"c:\users\jordan-dev\data\qdrant_db",
    #     collection_name="facebook_posts",
    #     parent_store=create_doc_store(r"c:\users\jordan-dev\data\parent_store"),
    # )

    # vector_store, doc_store, client = create_vector_store(
    #     db_folder_name=r"c:\users\jordan-dev\data\qdrant_db",
    #     collection_name="facebook_posts",
    #     parent_store_folder=r"c:\users\jordan-dev\data\parent_store",
    # )

    # # Optional: verify we have hybrid mode
    # if hasattr(vector_store, "retrieval_mode"):
    #     print(f"Vector store retrieval mode: {vector_store.retrieval_mode}")

    client, dense_model, sparse_model, late_interaction_model = create_client(
        collection_name="facebook_posts",
    )

    prompt_query(client, dense_model, sparse_model, late_interaction_model)

    client.close()
