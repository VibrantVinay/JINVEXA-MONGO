# D:\Jinvexa\DataHandle\Utils\DataExtractor.py

import os
from urllib.parse import urlparse
from typing import Optional, Any

from DataHandle.Utils.DocumentParser import DocumentParser
from DataHandle.Utils.ImageToText import ImageToText
from DataHandle.Utils.WebsiteParser import WebsiteParser
from DataHandle.Utils.YouTubeTranscript import YouTubeTranscript


class DataExtract:

    def __init__(self, llm_client: Optional[Any] = None):
        """
        Initialize DataExtractor with an optional LLM client.
        
        Args:
            llm_client: The OllamaLLMClient instance from app.py
                       Used for image OCR and other AI-powered extractions
        """
        self.llm_client = llm_client
        
        # Initialize parsers with llm_client if needed
        self.youtube = YouTubeTranscript()
        self.document = DocumentParser()
        self.image = ImageToText(llm_client=llm_client)
        self.website = WebsiteParser()

    def _is_url(self, value):
        try:
            p = urlparse(value)
            return p.scheme in ("http", "https")
        except Exception:
            return False

    def _is_youtube(self, url):
        hosts = [
            "youtube.com",
            "www.youtube.com",
            "youtu.be",
            "m.youtube.com"
        ]
        return any(host in url for host in hosts)

    def extract(self, source):
        """
        Extract data from the given source.
        
        Args:
            source: URL, file path, or YouTube link
            
        Returns:
            Dictionary with extracted data
        """
        # ---------------- URL ---------------- #
        if self._is_url(source):
            if self._is_youtube(source):
                result = self.youtube.transcribe(source)
                # Extract the full text from the result
                full_text = result.get('full_text', '')
                if not full_text or len(full_text.strip()) < 50:
                    # If no transcript, return error
                    return {
                        "type": "youtube",
                        "data": {
                            "content": f"No transcript available for this video.\nVideo ID: {result.get('video_id', '')}\n\nPlease try a different video with captions available.",
                            "video_id": result.get('video_id', ''),
                            "error": "No transcript available"
                        },
                        "metadata": {"source": source}
                    }
                return {
                    "type": "youtube",
                    "data": {
                        "content": full_text,
                        "video_id": result.get('video_id', ''),
                        "language": result.get('language', ''),
                        "segment_count": result.get('segment_count', 0),
                        "full_text": full_text
                    },
                    "metadata": {"source": source}
                }

            result = self.website.parse(source)
            # Website parser already returns structured data
            return {
                "type": "website",
                "data": self._process_website_result(result),
                "metadata": {"source": source}
            }

        # ---------------- FILE ---------------- #
        if not os.path.exists(source):
            raise FileNotFoundError(f"File not found: {source}")

        ext = os.path.splitext(source)[1].lower()

        # Document files
        if ext in [".pdf", ".docx", ".txt", ".md", ".py"]:
            result = self.document.parse(source)
            # Extract full text from document
            full_text = result.get('full_text', '')
            if not full_text or len(full_text.strip()) < 10:
                return {
                    "type": "document",
                    "data": {
                        "content": f"File: {result.get('file_name', source)}\nType: {result.get('file_type', 'document')}\n\nCould not extract meaningful text from this file.",
                        "file_name": result.get('file_name', ''),
                        "file_type": result.get('file_type', ''),
                        "error": "No text content extracted"
                    },
                    "metadata": {"source": source}
                }
            return {
                "type": "document",
                "data": {
                    "content": full_text,
                    "file_name": result.get('file_name', ''),
                    "file_type": result.get('file_type', ''),
                    "page_count": result.get('page_count', 0),
                    "paragraph_count": result.get('paragraph_count', 0),
                    "full_text": full_text
                },
                "metadata": {"source": source}
            }

        # Image files
        elif ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]:
            text = self.image.extract(source)
            if text == "NO_TEXT_FOUND" or not text.strip():
                return {
                    "type": "image",
                    "data": {
                        "content": f"Image: {os.path.basename(source)}\n\nNo text was found in this image.",
                        "error": "No text found in image"
                    },
                    "metadata": {"source": source}
                }
            return {
                "type": "image",
                "data": {"content": text},
                "metadata": {"source": source}
            }

        # Unsupported
        else:
            raise Exception(f"Unsupported input type: {ext}")

    def _process_website_result(self, result: dict) -> dict:
        """
        Process website parser result and extract meaningful content.
        """
        if result.get("status") == "error":
            return {
                "content": f"Error: {result.get('message', 'Unknown error')}",
                "error": result.get('message', 'Unknown error')
            }

        if result.get("status") == "verification_required":
            return {
                "content": f"Verification required: {result.get('message', 'CAPTCHA detected')}",
                "error": "verification_required"
            }

        # Build content from structured data
        content_parts = []

        if result.get("title"):
            content_parts.append(f"Title: {result['title']}")

        if result.get("description"):
            content_parts.append(f"Description: {result['description']}")

        # Use full_text from trafilatura if available (cleanest text)
        if result.get("full_text"):
            content_parts.append("\n" + result["full_text"])
        else:
            # Fallback to paragraphs
            if result.get("paragraphs"):
                content_parts.append("\nContent:")
                for p in result["paragraphs"][:30]:
                    if len(p) > 50:
                        content_parts.append(p)

        full_content = "\n".join(content_parts)

        if not full_content.strip():
            full_content = f"Website: No content could be extracted."

        return {
            "content": full_content,
            "title": result.get("title", ""),
            "description": result.get("description", ""),
            "full_text": result.get("full_text", ""),
            "headings": result.get("headings", [])
        }