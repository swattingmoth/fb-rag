"""Unit tests for Facebook posts processor.

Tests cover individual components of the processing pipeline to ensure
correct extraction, transformation, and error handling.
"""

import unittest
from datetime import datetime
from src.facebook_processor import (
    normalize_timestamp,
    extract_post_text,
    extract_urls,
    process_facebook_post,
    _infer_post_type,
)


class TestNormalizeTimestamp(unittest.TestCase):
    """Test Unix timestamp normalization."""
    
    def test_normalize_timestamp_valid(self):
        """Should convert valid Unix timestamp to ISO format."""
        # 2011-10-05 00:59:23 UTC
        result = normalize_timestamp(1317781963)
        self.assertIn("2011-10", result)
        self.assertIn("T", result)
    
    def test_normalize_timestamp_zero(self):
        """Should handle epoch timestamp."""
        result = normalize_timestamp(0)
        self.assertIn("1970", result)


class TestExtractPostText(unittest.TestCase):
    """Test post text extraction from various formats."""
    
    def test_extract_from_post_field(self):
        """Should extract text from 'post' field in data array."""
        post = {
            "data": [
                {"post": "Test post content"}
            ]
        }
        result = extract_post_text(post)
        self.assertEqual(result, "Test post content")
    
    def test_extract_from_title_fallback(self):
        """Should fallback to title if no post field."""
        post = {
            "data": [{}],
            "title": "Christopher Jordan shared a link."
        }
        result = extract_post_text(post)
        self.assertEqual(result, "Christopher Jordan shared a link.")
    
    def test_extract_empty_post(self):
        """Should return empty string for posts with no content."""
        post = {"data": []}
        result = extract_post_text(post)
        self.assertEqual(result, "")


class TestExtractUrls(unittest.TestCase):
    """Test URL extraction from posts and attachments."""
    
    def test_extract_urls_from_post_text(self):
        """Should extract URLs from post text."""
        post = {
            "data": [
                {"post": "Check this out: http://example.com/page"}
            ]
        }
        result = extract_urls(post)
        self.assertIn("http://example.com/page", result)
    
    def test_extract_urls_from_external_context(self):
        """Should extract URLs from attachment external_context."""
        post = {
            "attachments": [
                {
                    "data": [
                        {
                            "external_context": {
                                "url": "https://example.com/shared"
                            }
                        }
                    ]
                }
            ],
            "data": [{}]
        }
        result = extract_urls(post)
        self.assertIn("https://example.com/shared", result)
    
    def test_extract_urls_returns_unique(self):
        """Should return unique URLs without duplicates."""
        post = {
            "data": [
                {"post": "http://example.com http://example.com"}
            ]
        }
        result = extract_urls(post)
        # Count occurrences of example.com
        count = sum(1 for url in result if "example.com" in url)
        self.assertEqual(count, 1)


class TestInferPostType(unittest.TestCase):
    """Test post type inference from title."""
    
    def test_infer_link_type(self):
        """Should identify shared links."""
        post = {"title": "Christopher Jordan shared a link."}
        result = _infer_post_type(post)
        self.assertEqual(result, "link")
    
    def test_infer_photo_type(self):
        """Should identify photos."""
        post = {"title": "Christopher Jordan added a new photo."}
        result = _infer_post_type(post)
        self.assertEqual(result, "photo")
    
    def test_infer_event_type(self):
        """Should identify events."""
        post = {"title": "Christopher Jordan shared an event."}
        result = _infer_post_type(post)
        self.assertEqual(result, "event")
    
    def test_infer_status_type(self):
        """Should identify status updates."""
        post = {"title": "Christopher Jordan updated his status."}
        result = _infer_post_type(post)
        self.assertEqual(result, "status")


class TestProcessFacebookPost(unittest.TestCase):
    """Test full post processing pipeline."""
    
    def test_process_complete_post(self):
        """Should process a complete post with all fields."""
        post = {
            "timestamp": 1317781963,
            "title": "Christopher Jordan shared a link.",
            "data": [
                {
                    "post": "A different way to think about Christmas: "
                            "http://forgottenchristmas.org/resources/"
                }
            ],
            "attachments": [
                {
                    "data": [
                        {
                            "external_context": {
                                "url": "http://www.forgottenchristmas.org/resources/"
                            }
                        }
                    ]
                }
            ]
        }
        
        result = process_facebook_post(post)
        
        self.assertIsNotNone(result)
        self.assertIn("content", result)
        self.assertIn("metadata", result)
        self.assertIn("timestamp", result["metadata"])
        self.assertIn("urls", result["metadata"])
        self.assertEqual(result["metadata"]["post_type"], "link")
    
    def test_process_empty_post_returns_none(self):
        """Should return None for posts with no extractable content."""
        post = {"data": [{}], "timestamp": 1317781963}
        result = process_facebook_post(post)
        self.assertIsNone(result)
    
    def test_process_preserves_urls(self):
        """Should preserve URLs for source attribution."""
        post = {
            "timestamp": 1317781963,
            "data": [
                {"post": "Link: http://example.com"}
            ]
        }
        result = process_facebook_post(post)
        self.assertIn("http://example.com", result["metadata"]["urls"])


if __name__ == "__main__":
    unittest.main()
