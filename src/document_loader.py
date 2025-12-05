#TODO: Create a parameterized function to load documents
from importlib import metadata
from langchain_community.document_loaders import JSONLoader
def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["source"] = record.get("metadata").get("source")
    metadata["timestamp"] = record.get("metadata").get("timestamp")    
    metadata["title"] = record.get("metadata").get("title")
    # metadata["urls"] = record.get("metadata").get("urls")
    metadata["post_type"] = record.get("metadata").get("post_type")

    return metadata

loader = JSONLoader(
    file_path=r"c:\Users\jordan-dev\repos\rag\data\processed_posts.json",
    jq_schema=".[]",
    content_key="content",
    metadata_func=metadata_func,
)

documents = loader.load()