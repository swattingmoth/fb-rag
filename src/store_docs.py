from typing import Iterable, List, Tuple
from venv import logger
from fastembed import (
    LateInteractionTextEmbedding,
    SparseTextEmbedding,
    TextEmbedding,
)
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import JSONLoader
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client import models

from src import config as c


class OllamaTextEmbedding(TextEmbedding):
    """OllamaTextEmbedding is a wrapper around the OllamaEmbeddings class that is compatible with the TextEmbedding interface."""

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

    Returns:
        A new list of Document objects where long documents have been
        subdivided according to the configured chunk size and overlap.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""],
    )

    return splitter.split_documents(documents)


def create_client(
    config: c.Config,
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
        config (Config): The configuration object containing settings for Qdrant connection and collection name.
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
        list(dense_embeddings_model.query_embed("test")[0])  # type: ignore [index]
    )  # Should be 768 for qwen3-embedding:8b
    late_interation_embedding_dim = len(
        list(late_interaction_embeddings_model.query_embed("test"))[0][0]
    )  # Should be 128 for colbertv2.0

    # Connect to local Qdrant (embedded mode with persistent storage)
    client = QdrantClient(host=config.qdrant_host, port=config.qdrant_port)

    # Create collection if it doesn't exist
    if not client.collection_exists(config.collection_name):
        client.create_collection(
            collection_name=config.collection_name,
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


def store_documents(
    documents: List[Document],
    collection_name: str,
    client: QdrantClient,
    dense_embeddings_model: OllamaTextEmbedding,
    sparse_embeddings_model: SparseTextEmbedding,
    late_interaction_embeddings_model: LateInteractionTextEmbedding,
):
    """Embed and upsert a batch of documents into a Qdrant collection.

    This function takes a list of ``Document`` objects, computes dense,
    sparse and late‑interaction embeddings for each document text, constructs
    the appropriate payloads, and writes them into the specified Qdrant
    collection in batches. It is intended to be called after the client and
    embedding models have been initialized via :func:`create_client`.

    Args:
        documents: Documents to embed and store.
        collection_name: Name of the Qdrant collection to upsert into.
        client: Active QdrantClient instance.
        dense_embeddings_model: Model used for dense semantic embeddings.
        sparse_embeddings_model: Model used for BM25 sparse embeddings.
        late_interaction_embeddings_model: Model used for token‑level
            interaction embeddings.
    """
    batch_size = 50
    for batch_num in range(0, len(documents), batch_size):
        batch_docs = documents[batch_num : batch_num + batch_size]
        batch_texts = [doc.page_content for doc in batch_docs]
        logger.info(f"Creating embeddings for batch {batch_num // batch_size + 1}...")
        dense_embeddings = list(dense_embeddings_model.embed(batch_texts))
        sparse_embeddings = list(sparse_embeddings_model.embed(batch_texts))
        late_interaction_embeddings = list(
            late_interaction_embeddings_model.embed(batch_texts)
        )
        points = []
        for i, (
            dense_embedding,
            sparse_embedding,
            late_interaction_embedding,
            doc,
        ) in enumerate(
            zip(
                dense_embeddings,
                sparse_embeddings,
                late_interaction_embeddings,
                batch_docs,
            )
        ):
            global_id = batch_num + i
            point = models.PointStruct(
                id=global_id,
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
        logger.info(
            f"Upserting batch {batch_num // batch_size + 1} ({len(points)} points)..."
        )
        info = client.upsert(
            collection_name=collection_name,
            points=points,
        )
        logger.info(
            f"Upserted batch {batch_num // batch_size + 1} - operation info: {info}"
        )


def load_and_store_documents(config: c.Config) -> Tuple[
    QdrantClient,
    OllamaTextEmbedding,
    SparseTextEmbedding,
    LateInteractionTextEmbedding,
]:
    """
    Store documents in a vector store.

    Parameters:
        config (Config): The configuration object containing settings for document loading and storage.

    Returns:
        Tuple[QdrantClient, OllamaTextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding]: A tuple containing:
            - QdrantClient: The Qdrant client connected to the collection
            - OllamaTextEmbedding: The dense embedding model instance
            - SparseTextEmbedding: The sparse embedding model instance
            - LateInteractionTextEmbedding: The late interaction embedding model instance

    """
    documents = load_json_documents(config.document_path)

    client, dense_model, sparse_model, late_interaction_model = create_client(config)

    store_documents(
        documents=documents,
        collection_name=config.collection_name,
        client=client,
        dense_embeddings_model=dense_model,
        sparse_embeddings_model=sparse_model,
        late_interaction_embeddings_model=late_interaction_model,
    )

    return client, dense_model, sparse_model, late_interaction_model


if __name__ == "__main__":
    config = c.Config()
    c.configure_logging(config.log_folder + "/store_docs.log")
    load_and_store_documents(config)
