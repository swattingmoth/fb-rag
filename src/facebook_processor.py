"""Process Facebook posts JSON into RAG-suitable format.

This module handles ingestion and normalization of Facebook post exported data
into a document format optimized for retrieval-augmented generation (RAG).
The output preserves metadata for source attribution while extracting content
for semantic embedding and retrieval.
"""

import json
import logging
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

from src import config as c

logger = logging.getLogger(__name__)


def normalize_timestamp(unix_timestamp: int) -> str:
    """Convert Unix timestamp to ISO 8601 format.

    Args:
        unix_timestamp: Unix timestamp (seconds since epoch).

    Returns:
        ISO 8601 formatted datetime string.

    Raises:
        ValueError: If timestamp is invalid.
    """
    try:
        return datetime.fromtimestamp(unix_timestamp, tz=timezone.utc).isoformat()
    except (ValueError, OSError) as e:
        logger.warning(f"Invalid timestamp {unix_timestamp}: {e}")
        return ""


def clean_text(text: str) -> str:
    """Clean and normalize text for downstream RAG processing.

    - Normalizes unicode using NFKC so visually-similar characters are unified.
    - Removes control and formatting characters (categories starting with 'C'),
        e.g. zero-width spaces, bidi marks, and other invisible characters that
        break tokenization or embedding consistency.
    - Collapses repeated whitespace into single spaces and strips ends.

    This keeps the content stable for embedding and retrieval while preserving
    human-readable text and URLs.
    """
    if not text:
        return ""

    # Normalize unicode to a canonical form
    normalized = unicodedata.normalize("NFKC", text)

    # Remove control/formatting characters (category 'C*')
    cleaned_chars = [
        ch for ch in normalized if not unicodedata.category(ch).startswith("C")
    ]
    cleaned = "".join(cleaned_chars)

    # Collapse whitespace (including newlines, tabs) to single spaces
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


def extract_post_text(post: Dict[str, Any]) -> str:
    """Extract primary text content from a post.

    Attempts to extract post text from multiple possible locations in the
    Facebook data structure, with fallback to title if no post text found.

    Args:
        post: A single post dictionary from the Facebook export.

    Returns:
        Extracted post text, or empty string if none found.
    """
    # Try to get post text from data array
    if "data" in post and isinstance(post["data"], list):
        for item in post["data"]:
            if isinstance(item, dict) and "post" in item:
                return item["post"].strip()

    # Fallback to title
    if "title" in post:
        return post["title"].strip()

    return ""


def extract_urls(post: Dict[str, Any]) -> List[str]:
    """Extract all URLs from a post (post text and attachments).

    Preserves URLs for source attribution and context verification in RAG output.

    Args:
        post: A single post dictionary from the Facebook export.

    Returns:
        List of unique URLs found in the post.
    """
    urls = set()

    # Extract from post text
    post_text = extract_post_text(post)
    if post_text:
        import re

        # Simple URL pattern matching
        url_pattern = r"https?://[^\s]+"
        urls.update(re.findall(url_pattern, post_text))

    # Extract from attachments
    if "attachments" in post and isinstance(post["attachments"], list):
        for attachment in post["attachments"]:
            if isinstance(attachment, dict) and "data" in attachment:
                for data_item in attachment["data"]:
                    if isinstance(data_item, dict):
                        # Check external_context
                        if "external_context" in data_item:
                            ext_ctx = data_item["external_context"]
                            if isinstance(ext_ctx, dict) and "url" in ext_ctx:
                                urls.add(ext_ctx["url"])
                        # Check media (for photos/videos)
                        if "media" in data_item:
                            media = data_item["media"]
                            if isinstance(media, dict) and "image" in media:
                                img = media["image"]
                                if isinstance(img, dict) and "src" in img:
                                    urls.add(img["src"])

    return sorted(list(urls))


def process_facebook_post(post: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Transform a raw Facebook post into RAG document format.

    The output format is designed for:
    - Semantic embedding: 'content' field contains normalized text
    - Source attribution: 'metadata' preserves original post info
    - Retrieval context: structured data enables filtering by date, URLs, etc.

    Args:
        post: A single post dictionary from the Facebook export.

    Returns:
        Processed document dict or None if post has no extractable content.
    """
    content = extract_post_text(post)

    # Clean content to remove non-printable/special characters that hurt
    # tokenization and embeddings. This ensures consistent RAG inputs.
    content = clean_text(content)

    # Skip posts with no content
    if not content:
        logger.debug("Skipping post with no extractable content")
        return None

    timestamp = post.get("timestamp")
    urls = extract_urls(post)

    # Normalize timestamp to ISO 8601 for consistency
    normalized_timestamp = normalize_timestamp(timestamp) if timestamp else ""

    # Create document suitable for RAG embedding and retrieval
    document = {
        "content": content,
        "metadata": {
            "source": "facebook_posts",
            "timestamp": normalized_timestamp,
            "unix_timestamp": timestamp,
            # Clean title as well to avoid invisible chars in metadata
            "title": clean_text(post.get("title", "")),
            "urls": urls,
            "post_type": _infer_post_type(post),
        },
    }

    return document


def _infer_post_type(post: Dict[str, Any]) -> str:
    """Infer the type of post (link, photo, event, status, etc.).

    Uses the title field which typically contains Facebook's description
    of the action (e.g., "shared a link", "added a photo").

    Args:
        post: A single post dictionary from the Facebook export.

    Returns:
        Inferred post type string.
    """
    title = post.get("title", "").lower()

    if "link" in title:
        return "link"
    elif "photo" in title:
        return "photo"
    elif "video" in title:
        return "video"
    elif "event" in title:
        return "event"
    elif "status" in title or "updated" in title:
        return "status"
    else:
        return "other"


def fix_mojibake(text: str) -> str:
    """Repair common mojibake sequences that appear in Facebook exports.

    Facebook sometimes double‑encodes certain characters when exporting
    posts, resulting in sequences like ``\u00e2\u0080\u0093`` instead of a
    simple hyphen. This helper performs a handful of known substitutions to
    restore the intended characters before the JSON is parsed.

    Args:
        text: Raw text that may contain encoded mojibake sequences.

    Returns:
        The cleaned text with replacements applied.
    """
    replacements = {
        r"\u00e2\u0080\u0093": "-",
        r"\u00e2\u0080\u0094": "—",
        r"\u00e2\u0080\u009c": r"\"",
        r"\u00e2\u0080\u009d": r"\"",
        r"\u00e2\u0080\u0099": "'",
        r"\u00c3\u00a2\u0080\u0093": "-",  # double-encoded variant
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text


def process_facebook_posts(input_file: str, output_file: str) -> int:
    """Process all Facebook posts from input JSON file into RAG format.

    Reads the Facebook export JSON, processes each post into a standardized
    document format, and writes results to output file. Logs processing
    statistics for debugging and validation.

    Args:
        input_file: Path to input Facebook posts JSON file.
        output_file: Path to write processed documents JSON.

    Returns:
        Number of successfully processed posts.

    Raises:
        FileNotFoundError: If input file does not exist.
        json.JSONDecodeError: If input JSON is malformed.
    """
    logger.info(f"Loading Facebook posts from {input_file}")

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            raw_text = f.read()
            clean_text = fix_mojibake(raw_text)
            posts = json.loads(clean_text)
    except FileNotFoundError:
        logger.error(f"Input file not found: {input_file}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in input file: {e}")
        raise

    if not isinstance(posts, list):
        logger.error("Input JSON must be an array of posts")
        raise ValueError("Input JSON must be an array of posts")

    logger.info(f"Processing {len(posts)} posts")

    processed_documents = []
    skipped_count = 0

    for idx, post in enumerate(posts):
        try:
            document = process_facebook_post(post)
            if document:
                processed_documents.append(document)
            else:
                skipped_count += 1
        except Exception as e:
            logger.warning(f"Error processing post {idx}: {e}")
            skipped_count += 1

    # Write processed documents
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(processed_documents, f, indent=2, ensure_ascii=False)

    logger.info(
        f"Processed {len(processed_documents)} posts successfully, "
        f"skipped {skipped_count}"
    )
    logger.info(f"Output written to {output_file}")

    return len(processed_documents)


if __name__ == "__main__":
    config = c.Config()
    c.configure_logging(config.log_folder + "/facebook_processor.log")

    try:
        count = process_facebook_posts(config.raw_data_path, config.document_path)
        logger.info(f"Successfully processed {count} posts")
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        exit(1)
