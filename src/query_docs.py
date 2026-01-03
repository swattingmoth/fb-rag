from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_classic.retrievers import ParentDocumentRetriever

from store_docs import create_doc_store, create_document_retriever


def load_vectorstore(db_folder_name: str, collection_name: str = "documents") -> Chroma:
    """
    Load an existing persisted Chroma vector store.

    Args:
        db_folder_name: The folder where the database was persisted.
        collection_name: The name of the collection used when creating the store.
                         Must match the one used in persist_documents().

    Returns:
        Chroma vectorstore instance ready for querying or adding more documents.
    """
    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

    vectorstore = Chroma(
        persist_directory=db_folder_name,
        embedding_function=embeddings,
        collection_name=collection_name,
    )

    return vectorstore


def prompt_query(retriever: ParentDocumentRetriever) -> None:
    """
    Prompt the user for queries and return results from the retriever.

    Args:
        retriever: The ParentDocumentRetriever instance to query.
    """
    while True:
        query = input("Enter your query (or 'exit' to quit): ")
        if query.lower() == "exit":
            break

        results = retriever.invoke(query, top_k=5)

        if not results:
            print("No relevant documents found.")
            continue

        print("\nTop relevant documents:")
        for i, doc in enumerate(results, start=1):
            print(f"\nDocument {i}:\n{doc.page_content}\n")


if __name__ == "__main__":
    retriever = create_document_retriever(
        db_folder_name=r"c:\users\jordan-dev\data\chroma_db",
        collection_name="facebook_posts",
        parent_store=create_doc_store(r"c:\users\jordan-dev\data\parent_store"),
    )
    prompt_query(retriever)
