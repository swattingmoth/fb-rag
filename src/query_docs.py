from collections import defaultdict
from collections.abc import Iterable
from pydoc import doc
from typing import Any, List, Optional, Tuple

from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.documents import Document
from langchain_core.stores import BaseStore
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http.models import QueryRequest, NamedSparseVector, Filter


from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector, QueryRequest  # Import these
from store_docs import create_doc_store, create_document_retriever, create_vector_store
from qdrant_client.models import Prefetch, Query, Fusion


def prompt_query(
    vector_store: QdrantVectorStore,
    document_store: BaseStore[str, Document],
    default_k: int = 7,
    default_query_weight: float = 0.7,  # higher = more semantic importance
    default_sparse_weight: float = 0.3,  # higher = more keyword/BM25 importance
) -> None:
    """
    Interactive query loop with hybrid search tuning options.
    """
    print("\nHybrid Qdrant + ParentDocumentRetriever query tool")
    print("  - Semantic (Ollama nomic-embed) + BM25 keyword search")
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
  --qw FLOAT         → query (dense/semantic) weight 0.0–1.0 (default: 0.7)
  --sw FLOAT         → sparse (BM25/keyword) weight 0.0–1.0 (default: 0.3)
  --post-type VALUE  → filter by metadata.post_type (e.g. status, reel)
  --year YYYY        → filter by metadata.timestamp year

Examples:
  what did we post about hiking
  best pizza in town --k 5 --qw 0.6 --sw 0.4
  vacation photos --post-type photo --year 2024
            """
            )
            continue

        # Parse simple command-line style flags
        parts = user_input.split()
        query = parts[0]
        extra_args = parts[1:]

        k = default_k
        query_weight = default_query_weight
        sparse_weight = default_sparse_weight
        filter_dict: Optional[dict] = None

        i = 0
        while i < len(extra_args):
            arg = extra_args[i]
            if arg == "--k" and i + 1 < len(extra_args):
                try:
                    k = int(extra_args[i + 1])
                except ValueError:
                    print("Invalid --k value — using default")
            elif arg == "--qw" and i + 1 < len(extra_args):
                try:
                    query_weight = float(extra_args[i + 1])
                except ValueError:
                    print("Invalid --qw value — using default")
            elif arg == "--sw" and i + 1 < len(extra_args):
                try:
                    sparse_weight = float(extra_args[i + 1])
                except ValueError:
                    print("Invalid --sw value — using default")
            elif arg == "--post-type" and i + 1 < len(extra_args):
                if filter_dict is None:
                    filter_dict = {}
                filter_dict["post_type"] = extra_args[i + 1]
            elif arg == "--year" and i + 1 < len(extra_args):
                if filter_dict is None:
                    filter_dict = {}
                try:
                    year = int(extra_args[i + 1])
                    filter_dict["timestamp"] = {
                        "$gte": f"{year}-01-01",
                        "$lt": f"{year+1}-01-01",
                    }
                except ValueError:
                    print("Invalid --year — skipping filter")
            i += 2 if arg.startswith("--") else 1

        # Build search kwargs with hybrid weights
        search_kwargs: dict[str, Any] = {
            "k": k,
            # Qdrant hybrid-specific fusion weights (passed through search_kwargs)
            "query_weight": query_weight,
            "sparse_weight": sparse_weight,
        }

        if filter_dict:
            # Qdrant filter syntax → convert simple dict to Qdrant Filter
            from qdrant_client.http.models import (
                Filter,
                FieldCondition,
                MatchValue,
                Range,
            )

            qdrant_filter = Filter(
                must=[
                    (
                        FieldCondition(key=key, match=MatchValue(value=value))
                        if isinstance(value, (str, int, bool))
                        else FieldCondition(key=key, range=Range(**value))
                    )
                    for key, value in filter_dict.items()
                ]
            )
            search_kwargs["filter"] = qdrant_filter

        print(f"\nQuery: {query}")
        print(
            f"  → k={k}, semantic_weight={query_weight:.2f}, keyword_weight={sparse_weight:.2f}"
        )
        if filter_dict:
            print(f"  → filter: {filter_dict}")

        try:
            # results = retriever.invoke(query, **search_kwargs)
            results = custom_hybrid_retrieve(
                vector_store=vector_store,
                docstore=doc_store,
                query=query,
                dense_weight=query_weight,
                sparse_weight=sparse_weight,
                final_k=k,
            )
        except Exception as e:
            print(f"Error during retrieval: {e}")
            continue

        if not results:
            print("No relevant documents found.")
            continue

        print("\nTop relevant parent documents:")
        print(format_documents(results))


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


def manual_sparse_search(
    db_folder_name: str, collection_name: str, query: str, k: int = 7
):
    client = QdrantClient(path=db_folder_name)

    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

    # Embed the query sparsely
    sparse_query = sparse_embeddings.embed_query(query)
    if not sparse_query or len(sparse_query.indices) == 0:
        print("Query has no sparse tokens! Try a different query.")
        client.close()
        return

    print(
        f"Query sparse tokens (sample): indices {sparse_query.indices[:10]}, values {sparse_query.values[:10]}"
    )

    # Convert FastEmbed output to Qdrant's SparseVector model
    sparse_vector = SparseVector(
        indices=sparse_query.indices,  # must be list[int]
        values=sparse_query.values,  # must be list[float]
    )

    # Build the query request for the named sparse vector
    query_request = QueryRequest(
        query=sparse_vector,  # ← SparseVector, not NamedSparseVector
        using="sparse",  # ← your sparse_vector_name
        limit=k,
        with_payload=True,
        # Optional: add query_filter=Filter(...) here if needed
    )

    # Execute the query
    response = client.query_points(
        collection_name=collection_name,
        query=query_request.query,
        using=query_request.using,
        limit=query_request.limit,
        with_payload=query_request.with_payload,
        # Add other params like offset, score_threshold if needed
    )

    print(f"\nPure Sparse Results for '{query}' ({len(response.points)} hits):")
    for i, point in enumerate(response.points, 1):
        score = point.score
        text = point.payload.get("page_content", "No content")[:300]
        print(f"{i}. Point ID: {point.id}, Score: {score:.4f}")
        print(f"   Text: {text}...")
        print("---")

    client.close()


def custom_weighted_hybrid_retrieve(
    vectorstore: QdrantVectorStore,
    docstore: BaseStore[str, Document],
    query: str,
    dense_weight: float = 0.3,
    sparse_weight: float = 0.7,
    fetch_k: int = 20,  # Fetch more candidates before fusion
    final_k: int = 7,
) -> List[Document]:
    """Custom hybrid retrieval with tunable weights, returning unique parent docs."""

    # Dense-only search
    dense_results = vectorstore.similarity_search_with_score(
        query, k=fetch_k, retrieval_mode=RetrievalMode.DENSE
    )

    # Sparse-only search
    sparse_results = vectorstore.similarity_search_with_score(
        query, k=fetch_k, retrieval_mode=RetrievalMode.SPARSE
    )

    # Fuse with weighted RRF (simple implementation)
    def weighted_rrf(results: List[Tuple[Document, float]], weight: float) -> dict:
        ranked = defaultdict(float)
        for rank, (doc, score) in enumerate(results, 1):
            child_id = doc.metadata.get(
                "doc_id"
            )  # Assuming child has parent ID in metadata
            ranked[child_id] += weight / (
                rank + 60
            )  # Standard RRF formula, adjustable constant
        return ranked

    dense_rrf = weighted_rrf(dense_results, dense_weight)
    sparse_rrf = weighted_rrf(sparse_results, sparse_weight)

    # Combine scores
    combined_scores = defaultdict(float)
    for child_id in set(dense_rrf) | set(sparse_rrf):
        combined_scores[child_id] = dense_rrf.get(child_id, 0) + sparse_rrf.get(
            child_id, 0
        )

    # Sort by combined score descending
    sorted_child_ids = sorted(combined_scores, key=combined_scores.get, reverse=True)[
        : final_k * 2
    ]  # Buffer for dups

    # Fetch unique parents from docstore
    unique_parents = []
    seen = set()
    for child_id in sorted_child_ids:
        parent = docstore.get(child_id)
        if parent and parent.page_content not in seen:
            seen.add(parent.page_content)
            unique_parents.append(parent)
        if len(unique_parents) >= final_k:
            break

    return unique_parents


def custom_hybrid_retrieve(
    vector_store: QdrantVectorStore,
    docstore: BaseStore[str, Document],
    query: str,
    dense_weight: float = 0.4,
    sparse_weight: float = 0.6,
    fetch_k: int = 30,
    final_k: int = 7,
) -> List[Document]:
    client = vector_store.client  # Your QdrantClient
    collection = vector_store.collection_name

    dense_vec = vector_store.embedding.embed_query(query)
    sparse_vec = vector_store.sparse_embedding.embed_query(query)
    sparse_q = SparseVector(indices=sparse_vec.indices, values=sparse_vec.values)

    response = client.query_points(
        collection_name=collection,
        prefetch=[
            Prefetch(
                query=dense_vec, using="dense", limit=fetch_k, weight=dense_weight
            ),
            Prefetch(
                query=sparse_q, using="sparse", limit=fetch_k, weight=sparse_weight
            ),
        ],
        query=Query(fusion=Fusion.DBSF),  # or RRF; DBSF often better with weights
        limit=final_k,
        with_payload=True,
    )

    # Convert to LangChain Documents (adjust payload keys as needed)
    docs = []
    for point in response.points:
        payload = point.payload
        doc = Document(
            page_content=payload.get("page_content", ""),
            metadata=payload.get("metadata", {}),
        )
        docs.append(doc)

    # Optional: resolve to parents if using ParentDocumentRetriever style
    # (fetch from docstore if child IDs are in metadata)

    return docs


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

    vector_store, doc_store, client = create_vector_store(
        db_folder_name=r"c:\users\jordan-dev\data\qdrant_db",
        collection_name="facebook_posts",
        parent_store_folder=r"c:\users\jordan-dev\data\parent_store",
    )

    # Optional: verify we have hybrid mode
    if hasattr(vector_store, "retrieval_mode"):
        print(f"Vector store retrieval mode: {vector_store.retrieval_mode}")

    prompt_query(vector_store, doc_store)

    client.close()
