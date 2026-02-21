from collections.abc import Iterable
from typing import Optional

from fastembed import LateInteractionTextEmbedding, SparseTextEmbedding
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client import QdrantClient, models
from store_docs import (
    OllamaTextEmbedding,
    load_embed_store,
)


def prompt_query(
    client: QdrantClient,
    dense_model: OllamaTextEmbedding,
    sparse_model: SparseTextEmbedding,
    late_interaction_model: LateInteractionTextEmbedding,
) -> None:
    """
    Interactive query loop with hybrid search tuning options.
    """
    default_k = 7
    print("\nHybrid Qdrant query tool")
    print("  - Type 'exit' to quit")
    print("  - Type 'help' for options\n")

    while True:
        user_input = input("\nEnter your query (or 'help' / 'exit'): ").strip()

        if user_input.lower() == "exit":
            print("Goodbye.")
            break

        if user_input.lower() == "help":
            print(
                """
Options you can add after your query (space separated):
  --k N              → number of final parent docs (default: 7)
  --post-type VALUE  → filter by metadata.post_type (e.g. status, reel)
  --year YYYY        → filter by metadata.timestamp year

Examples:
  what did we post about hiking
  best pizza in town --k 5 
  vacation photos --post-type photo --year 2024
            """
            )
            continue

        # Parse simple command-line style flags
        query = user_input

        k = default_k

        print(f"\nQuery: {query}")
        print(f"  → k={k}")

        try:
            dense_vectors = dense_model.query_embed(query)[0]
            sparse_vectors = next(sparse_model.query_embed(query))
            late_vectors = next(late_interaction_model.query_embed(query))
            prefetch = [
                models.Prefetch(
                    query=dense_vectors,
                    using="dense",
                    limit=k,
                ),
                models.Prefetch(
                    query=models.SparseVector(**sparse_vectors.as_object()),
                    using="bm25",
                    limit=k,
                ),
            ]

            results = client.query_points(
                collection_name="facebook_posts",
                prefetch=prefetch,
                query=late_vectors,
                using="colbertv2.0",
                limit=k,
                with_payload=True,
            )
        except Exception as e:
            print(f"Error during retrieval: {e}")
            continue

        if not results:
            print("No relevant documents found.")
            continue

        print("\nTop relevant parent documents:")
        # print(format_documents(results.))
        for i, result in enumerate(results):
            print("Result", i + 1)
            for scored in result[1]:
                print(scored.payload, "score", scored.score)


def format_documents(documents: Iterable[Document]) -> str:
    formatted_docs: list[str] = []
    for i, doc in enumerate(documents, start=1):
        formatted = f"──── Document {i} ────\n"
        formatted += "Metadata:\n"
        for key, value in sorted(doc.metadata.items()):
            formatted += f"  {key}: {value}\n"
        formatted += f"\nContent:\n{doc.page_content.strip()}\n"
        formatted_docs.append(formatted)
    return "\n".join(formatted_docs)


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

    client, dense_model, sparse_model, late_interaction_model = load_embed_store()

    # TODO: Update prompt_query to do retreivel as described here: https://qdrant.tech/documentation/tutorials-search-engineering/reranking-hybrid-search/
    prompt_query(client, dense_model, sparse_model, late_interaction_model)

    client.close()
