import re
from threading import local
from langchain_chroma import Chroma
from langchain_classic.storage import LocalFileStore
from langchain_core.stores import ByteStore
from langchain_community.document_loaders import JSONLoader
from typing import List
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage._lc_store import create_kv_docstore
from langchain_core.stores import BaseStore
from langchain_core.retrievers import BaseRetriever


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
) -> ParentDocumentRetriever:
    """Create and configure a ParentDocumentRetriever backed by a local Chroma vector store.

    This function builds a text splitter, an embedding function, and a Chroma vectorstore
    and then returns a ParentDocumentRetriever that uses the vectorstore for child-document
    retrieval and the provided parent_store for parent-document lookups.

    Args:
        db_folder_name (str): Filesystem path to use as Chroma's persist_directory. The
            directory will be used by Chroma to store vector data for the specified collection.
        collection_name (str): Name of the Chroma collection to use or create. Documents
            (child vectors) will be stored/retrieved under this collection.
        parent_store (ByteStore): A docstore implementing the expected ByteStore/docstore
            interface required by ParentDocumentRetriever (used to fetch parent documents).
    Returns:
        ParentDocumentRetriever: A retriever configured to:
            - split incoming text into chunks using RecursiveCharacterTextSplitter
              (chunk_size=400, chunk_overlap=50, separators=["\n\n", "\n", ".", " ", ""])
            - embed chunks with OllamaEmbeddings(model="nomic-embed-text:latest")
            - query a Chroma vectorstore persisted at db_folder_name and scoped to collection_name
            - consult parent_store for parent-document access
    Notes:
        - This function constructs and returns the retriever but does not itself persist
          any documents or call vectorstore.persist(); persistence behavior depends on
          the Chroma client and how it's used elsewhere in your application.
        - parent_store must implement the methods ParentDocumentRetriever expects (e.g.,
          fetch/lookup by id). Mismatched interfaces will lead to runtime errors.
        - If you need different chunking, overlap, or embedding settings, modify the
          splitter and embeddings before creating the retriever.
    Example:
        retriever = create_document_retriever("/path/to/db", "my_collection", my_parent_store)

    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
    # vectorstore for child documents
    vectorstore = Chroma(
        persist_directory=db_folder_name,
        collection_name=collection_name,
        embedding_function=embeddings,
    )

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=parent_store,
        child_splitter=splitter,
        search_kwargs={"k": 7},
    )

    return retriever


def create_doc_store(folder_path: str) -> BaseStore[str, Document]:
    """Create a LocalFileStore docstore at the specified folder path.

    Args:
        folder_path: Filesystem path where the LocalFileStore will persist data.

    Returns:
        A LocalFileStore instance configured to use the specified folder path.
    """
    local_store = LocalFileStore(folder_path)
    return create_kv_docstore(local_store)


if __name__ == "__main__":
    documents = load_json_documents(r"c:\Users\jordan-dev\data\processed_posts.json")

    retriever = create_document_retriever(
        db_folder_name=r"c:\users\jordan-dev\data\chroma_db",
        collection_name="facebook_posts",
        parent_store=create_doc_store(r"c:\users\jordan-dev\data\parent_store"),
    )

    retriever.add_documents(documents)
