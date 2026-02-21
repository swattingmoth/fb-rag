from json import load
import shutil
from typing import Iterable, List, Tuple
from fastembed import (
    LateInteractionTextEmbedding,
    SparseEmbedding,
    SparseTextEmbedding,
    TextEmbedding,
)
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import (
    ParentDocumentRetriever,
)  # Updated import (langchain_classic → langchain)
from langchain_classic.storage import (
    LocalFileStore,
)  # Updated import (langchain_classic → langchain)
from langchain_classic.storage._lc_store import create_kv_docstore  # Updated import
from langchain_community.document_loaders import JSONLoader
from langchain_core.stores import BaseStore
from langchain_core.retrievers import BaseRetriever
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client import models
from qdrant_client.http.exceptions import UnexpectedResponse
import os


class OllamaTextEmbedding(TextEmbedding):
    def __init__(self, model: str):
        self.embeddings = OllamaEmbeddings(model=model)

    def query_embed(self, query: str | Iterable[str], **kwargs) -> Iterable[np.ndarray]:
        if not isinstance(query, str):
            raise ValueError(
                "OllamaTextEmbedding only supports single string queries for query_embed."
            )
        embedding = self.embeddings.embed_query(query)
        return self._list_to_numpy([embedding])

    def _list_to_numpy(self, embeddings: List[List[float]]) -> List[np.ndarray]:
        return [np.array(l) for l in embeddings]  # Convert each list to np.ndarray

    def embed(
        self,
        documents: str | Iterable[str],
        batch_size: int = 256,
        parallel: int | None = None,
        **kwargs,
    ) -> Iterable[np.ndarray]:
        if isinstance(documents, str):
            return self._list_to_numpy(self.embeddings.embed_documents([documents]))

        return self._list_to_numpy(self.embeddings.embed_documents(list(documents)))


def load_json_documents(file_path: str) -> List[Document]:
    """Load documents from a JSON file.

    Args:
        file_path: Path to the JSON file containing documents.

    Returns:
        A list of Document objects loaded from the JSON file.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """

    def metadata_func(record: dict, metadata: dict) -> dict:
        record_metadata = record.get("metadata") or {}
        metadata["source"] = record_metadata.get("source")
        metadata["timestamp"] = record_metadata.get("timestamp")
        metadata["title"] = record_metadata.get("title")
        metadata["post_type"] = record_metadata.get("post_type")
        return metadata

    loader = JSONLoader(
        file_path=file_path,
        jq_schema=".[]",
        content_key="content",
        metadata_func=metadata_func,
    )

    return loader.load()


def split_documents(documents: List[Document]) -> List[Document]:
    """Split documents into smaller chunks.

    Args:
        documents: List of Document objects to be split.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""],
    )

    return splitter.split_documents(documents)


def create_document_retriever(
    db_folder_name: str, collection_name: str, parent_store: BaseStore[str, Document]
) -> Tuple[ParentDocumentRetriever, QdrantClient]:
    """Create and configure a ParentDocumentRetriever backed by a local Qdrant vector store.

    This function builds a text splitter, dense and sparse embedding functions, and a Qdrant vectorstore
    and then returns a ParentDocumentRetriever that uses the vectorstore for child-document
    retrieval (with hybrid dense + sparse search) and the provided parent_store for parent-document lookups.

    Args:
        db_folder_name (str): Filesystem path to use as Qdrant's local storage directory. The
            directory will be used by Qdrant to store vector data for all collections.
        collection_name (str): Name of the Qdrant collection to use or create. Documents
            (child vectors) will be stored/retrieved under this collection.
        parent_store (BaseStore[str, Document]): A docstore implementing the expected interface
            required by ParentDocumentRetriever (used to fetch parent documents).
    Returns:
        ParentDocumentRetriever: A retriever configured to:
            - split incoming text into chunks using RecursiveCharacterTextSplitter
              (chunk_size=400, chunk_overlap=50, separators=["\n\n", "\n", ".", " ", ""])
            - embed chunks with OllamaEmbeddings(model="nomic-embed-text:latest") for dense vectors
            - embed chunks with FastEmbedSparse(model_name="Qdrant/bm25") for sparse (BM25 keyword) vectors
            - query a Qdrant vectorstore persisted at db_folder_name and scoped to collection_name using hybrid search
            - consult parent_store for parent-document access
    Notes:
        - This function creates the Qdrant collection if it doesn't exist, with configurations for both dense and sparse vectors.
        - parent_store must implement the methods ParentDocumentRetriever expects (e.g., fetch/lookup by id).
        - If you need different chunking, overlap, or embedding settings, modify the splitter and embeddings before creating the retriever.
        - Ensure Qdrant is running locally or use the embedded mode (as here with path=db_folder_name).
    Example:
        retriever = create_document_retriever("/path/to/db", "my_collection", my_parent_store)

    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    # Dense embeddings (semantic)
    dense_embeddings = OllamaTextEmbedding(model="nomic-embed-text:latest")

    # Sparse embeddings (BM25 for keyword matching)
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

    # Get the dimension of dense embeddings
    embedding_dim = len(
        dense_embeddings.embed("test")
    )  # Should be 768 for nomic-embed-text

    # Connect to local Qdrant (embedded mode with persistent storage)
    client = QdrantClient(path=db_folder_name)

    # Create collection if it doesn't exist
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=embedding_dim,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False)
                )
            },
        )

    # Vectorstore for child documents with hybrid search
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=dense_embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name="dense",
        sparse_vector_name="sparse",
    )

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=parent_store,
        child_splitter=splitter,
        search_kwargs={"k": 7},
    )

    return retriever, client


def create_client(
    collection_name: str,
) -> Tuple[
    QdrantClient, OllamaTextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding
]:
    """Create and configure a QdrantClient with dense, sparse, and late interaction embeddings.
    Initializes a Qdrant vector store with three embedding models:
    - Dense embeddings (semantic): Qwen3 8B model for semantic similarity
    - Sparse embeddings (keyword): BM25 model for keyword matching
    - Late interaction embeddings: ColBERTv2.0 model for token-level interactions
        Tuple[QdrantClient, OllamaTextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding]:
            A tuple containing:
            - QdrantClient: Connected in-memory Qdrant client
            - OllamaTextEmbedding: Dense embedding model instance (768 dimensions)
            - SparseTextEmbedding: BM25 sparse embedding model instance
            - LateInteractionTextEmbedding: ColBERT embedding model instance (128 dimensions)
    Note:
        Creates the collection if it doesn't already exist with configured vector spaces
        for dense, late interaction, and sparse embeddings.

    Parameters:
        collection_name (str): The name of the Qdrant collection to create or connect to.
    Returns:
        Tuple[QdrantClient, OllamaTextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding]: A tuple containing:
            - QdrantClient: Connected in-memory Qdrant client
            - OllamaTextEmbedding: Dense embedding model instance (768 dimensions)
            - SparseTextEmbedding: BM25 sparse embedding model instance
            - LateInteractionTextEmbedding: ColBERT embedding model instance (128 dimensions)
    """

    # Dense embeddings (semantic)
    dense_embeddings_model = OllamaTextEmbedding(model="qwen3-embedding:8b")

    # Sparse embeddings (BM25 for keyword matching)
    sparse_embeddings_model = SparseTextEmbedding(model_name="Qdrant/bm25")

    # late interaction embedding
    late_interaction_embeddings_model = LateInteractionTextEmbedding(
        "colbert-ir/colbertv2.0"
    )

    dense_embedding_dim = len(
        list(dense_embeddings_model.query_embed("test")[0])
    )  # Should be 768 for qwen3-embedding:8b
    late_interation_embedding_dim = len(
        list(late_interaction_embeddings_model.query_embed("test"))[0][0]
    )  # Should be 128 for colbertv2.0

    # Connect to local Qdrant (embedded mode with persistent storage)
    client = QdrantClient(":memory:")

    # Create collection if it doesn't exist
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=dense_embedding_dim,
                    distance=models.Distance.COSINE,
                ),
                "colbertv2.0": models.VectorParams(
                    size=late_interation_embedding_dim,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM,
                    ),
                    hnsw_config=models.HnswConfigDiff(m=0),
                ),
            },
            sparse_vectors_config={
                "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )

    return (
        client,
        dense_embeddings_model,
        sparse_embeddings_model,
        late_interaction_embeddings_model,
    )


def create_doc_store(folder_path: str) -> BaseStore[str, Document]:
    """Create a LocalFileStore docstore at the specified folder path.

    Args:
        folder_path: Filesystem path where the LocalFileStore will persist data.

    Returns:
        A LocalFileStore instance configured to use the specified folder path.
    """
    local_store = LocalFileStore(folder_path)
    return create_kv_docstore(local_store)


def clear_stores(db_folder_name: str, collection_name: str, parent_store_path: str):
    """
    Clear Qdrant collection and parent docstore folder to start fresh.
    """
    client = QdrantClient(path=db_folder_name)

    # 1. Delete Qdrant collection if it exists
    try:
        if client.collection_exists(collection_name):
            client.delete_collection(collection_name)
            print(f"Deleted existing Qdrant collection: {collection_name}")
        else:
            print(f"No existing Qdrant collection found: {collection_name}")
    except UnexpectedResponse as e:
        if e.status_code == 404:
            print(f"Collection {collection_name} not found (already clean)")
        else:
            raise

    # 2. Clear the parent docstore folder
    if os.path.exists(parent_store_path):
        shutil.rmtree(parent_store_path)
        print(f"Cleared parent docstore folder: {parent_store_path}")
    else:
        print(f"No parent docstore folder found: {parent_store_path}")

    # Optional: recreate empty folder so create_kv_docstore doesn't complain
    os.makedirs(parent_store_path, exist_ok=True)


def inspect_qdrant_collection(db_folder_name: str, collection_name: str):
    client = QdrantClient(path=db_folder_name)

    try:
        info = client.get_collection(collection_name)
    except Exception as e:
        print(f"Error fetching collection info: {e}")
        client.close()
        return

    print("Collection Info:")
    print(f"  Status: {info.status}")
    print(f"  Optimizer status: {info.optimizer_status}")
    print(f"  Points count: {info.points_count}")
    print(f"  Indexed vectors count: {info.indexed_vectors_count}")
    print(f"  Segments count: {info.segments_count}")

    # Vector configs
    print("\nDense Vectors:")
    if info.config.params.vectors:
        for name, vec_params in info.config.params.vectors.items():
            print(f"  '{name}': size={vec_params.size}, distance={vec_params.distance}")
    else:
        print("  No dense vectors configured.")

    # Sparse Vectors
    print("\nSparse Vectors:")
    sparse_config = info.config.params.sparse_vectors
    if sparse_config and "sparse" in sparse_config:
        sparse_params = sparse_config["sparse"]
        print(f"  'sparse' configured: {sparse_params}")
        print(
            f"    Index on_disk: {sparse_params.index.on_disk if sparse_params.index else 'N/A'}"
        )
    else:
        print("  WARNING: No sparse vectors configured in this collection!")

    client.close()


def view_sample_sparse_vector(db_folder_name: str, collection_name: str):
    client = QdrantClient(path=db_folder_name)

    # Scroll first point (or use offset=0, limit=1)
    points, _ = client.scroll(
        collection_name=collection_name,
        limit=1,
        with_payload=True,
        with_vectors={"include": ["sparse"]},
    )

    if points:
        point = points[0]
        print(f"Sample Point ID: {point.id}")
        print(f"Payload (metadata + text snippet): {point.payload}")

        if point.vector and "sparse" in point.vector:
            sparse_vec = point.vector["sparse"]
            print(
                f"Sparse Vector (BM25): Indices {sparse_vec.indices[:10]}..., Values {sparse_vec.values[:10]}..."
            )
            if len(sparse_vec.indices) > 0:
                print("Sparse vector populated correctly (non-empty).")
            else:
                print("WARNING: Sparse vector is empty! BM25 not working.")
        else:
            print("WARNING: No sparse vector found on point!")
    else:
        print("No points in collection.")

    client.close()


def store_documents(
    documents: List[Document],
    collection_name: str,
    client: QdrantClient,
    dense_embeddings_model: OllamaTextEmbedding,
    sparse_embeddings_model: SparseTextEmbedding,
    late_interaction_embeddings_model: LateInteractionTextEmbedding,
):
    points = []

    document_texts = [doc.page_content for doc in documents]
    print("Creating dense embeddings...")
    dense_embeddings = list(dense_embeddings_model.embed(document_texts))
    print("Creating sparse embeddings...")
    sparse_embeddings = list(sparse_embeddings_model.embed(document_texts))
    print("Creating late interaction embeddings...")
    late_interaction_embeddings = list(
        late_interaction_embeddings_model.embed(document_texts)
    )

    print("Upserting points into Qdrant...")
    for idx, (
        dense_embedding,
        sparse_embedding,
        late_interaction_embedding,
        doc,
    ) in enumerate(
        zip(dense_embeddings, sparse_embeddings, late_interaction_embeddings, documents)
    ):
        point = models.PointStruct(
            id=idx,
            vector={
                "dense": dense_embedding,  # type: ignore [dict-item]
                "bm25": sparse_embedding.as_object(),  # type: ignore [dict-item]
                "colbertv2.0": late_interaction_embedding,  # type: ignore [dict-item]
            },
            payload={
                "document": doc.page_content,
                **doc.metadata,
            },
        )
        points.append(point)

    operation_info = client.upsert(
        collection_name=collection_name,
        points=points,
    )
    print(f"Upserted {len(points)} points. Operation info: {operation_info}")


def load_embed_store() -> Tuple[
    QdrantClient,
    OllamaTextEmbedding,
    SparseTextEmbedding,
    LateInteractionTextEmbedding,
]:
    documents = load_json_documents(r"c:\Users\jordan-dev\data\processed_posts.json")

    client, dense_model, sparse_model, late_interaction_model = create_client(
        collection_name="facebook_posts"
    )

    store_documents(
        documents=documents,
        collection_name="facebook_posts",
        client=client,
        dense_embeddings_model=dense_model,
        sparse_embeddings_model=sparse_model,
        late_interaction_embeddings_model=late_interaction_model,
    )

    return client, dense_model, sparse_model, late_interaction_model


if __name__ == "__main__":
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"  # silence symlink nag
    os.environ["FASTEMBED_CACHE_PATH"] = (
        r"c:\users\jordan-dev\data\fastembed_cache"  # persistent cache
    )
    load_embed_store()

    # Clear existing stores for a fresh start
    # clear_stores(
    #     db_folder_name=r"c:\users\jordan-dev\data\qdrant_db",
    #     collection_name="facebook_posts",
    #     parent_store_path=r"c:\users\jordan-dev\data\parent_store",
    # )

    # retriever, client = create_document_retriever(
    #     db_folder_name=r"c:\users\jordan-dev\data\qdrant_db",
    #     collection_name="facebook_posts",
    #     parent_store=create_doc_store(r"c:\users\jordan-dev\data\parent_store"),
    # )

    # retriever.add_documents(documents)

    # client.close()

    # inspect_qdrant_collection(
    #     db_folder_name=r"c:\users\jordan-dev\data\qdrant_db",
    #     collection_name="facebook_posts",
    # )

    # view_sample_sparse_vector(
    #     db_folder_name=r"c:\users\jordan-dev\data\qdrant_db",
    #     collection_name="facebook_posts",
    # )
