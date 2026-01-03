from collections.abc import Iterable
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.documents import Document

from store_docs import create_doc_store, create_document_retriever


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

        results = retriever.invoke(query)

        if not results:
            print("No relevant documents found.")
            continue

        print("\nTop relevant documents:")
        print(format_documents(results))


def format_documents(documents: Iterable[Document]) -> str:
    """
    Format a document for display.

    Args:
        doc: The Document object to format.

    Returns:
        A formatted string representation of the document.
    """
    formatted_docs: list[str] = []
    for i, doc in enumerate(documents, start=1):
        formatted = f"---------------Document {i}\nMetadata:"
        for key, value in doc.metadata.items():
            formatted += f"{key}: {value}\n"
        formatted += f"Document Content:\n{doc.page_content}\n"
        formatted_docs.append(formatted)

    return "\n".join(formatted_docs)


if __name__ == "__main__":
    retriever = create_document_retriever(
        db_folder_name=r"c:\users\jordan-dev\data\chroma_db",
        collection_name="facebook_posts",
        parent_store=create_doc_store(r"c:\users\jordan-dev\data\parent_store"),
    )
    prompt_query(retriever)
