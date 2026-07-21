from __future__ import annotations

import logging
import time
from io import BytesIO
from typing import List, Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from requests.adapters import HTTPAdapter
from requests.exceptions import (
    ConnectionError as RequestsConnectionError,
    RequestException,
    Timeout,
)

from module_3.discovery.models import FetchedDocument


logger = logging.getLogger(__name__)

NON_RETRYABLE_STATUS_CODES = {400, 401, 403, 404, 405, 410, 422}
RETRYABLE_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}
HTML_CONTENT_TYPES = {"text/html", "application/xhtml+xml"}
PDF_CONTENT_TYPES = {"application/pdf", "application/x-pdf"}

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/pdf,"
        "application/xml;q=0.9,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
}


class DocumentFetcher:
    def __init__(
        self,
        timeout: int = 30,
        max_size: int = 10 * 1024 * 1024,
        retries: int = 3,
        retry_backoff_seconds: float = 1.5,
        max_redirects: int = 5,
    ):
        self.timeout = timeout
        self.max_size = max_size
        self.retries = max(1, retries)
        self.retry_backoff_seconds = max(0.0, retry_backoff_seconds)

        self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)
        self.session.max_redirects = max_redirects

        adapter = HTTPAdapter(max_retries=0)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        self.allowed_schemes = {"http", "https"}

    def _validate_url(self, url: str) -> bool:
        parsed = urlparse(url)
        return parsed.scheme.lower() in self.allowed_schemes and bool(parsed.netloc)

    def _error_document(
        self,
        *,
        url: str,
        error: str,
        final_url: str | None = None,
        content_type: str = "",
    ) -> FetchedDocument:
        return FetchedDocument(
            final_url=final_url or url,
            content_type=content_type,
            text="",
            text_chunks=[],
            fetch_status="error",
            fetch_error=error,
        )

    @staticmethod
    def _base_content_type(value: str | None) -> str:
        return (value or "").split(";", 1)[0].strip().lower()

    @staticmethod
    def _retry_after_seconds(response: requests.Response) -> float | None:
        value = response.headers.get("Retry-After")
        if not value:
            return None
        try:
            return max(0.0, float(value.strip()))
        except ValueError:
            return None

    def _retry_delay(
        self,
        *,
        attempt: int,
        response: requests.Response | None = None,
    ) -> float:
        if response is not None and response.status_code == 429:
            retry_after = self._retry_after_seconds(response)
            if retry_after is not None:
                return retry_after
        return self.retry_backoff_seconds * (2 ** attempt)

    def _read_streamed_content(self, response: requests.Response) -> bytes:
        content_length = response.headers.get("Content-Length")
        if content_length:
            try:
                declared_size = int(content_length)
            except ValueError:
                declared_size = None
            if declared_size is not None and declared_size > self.max_size:
                raise ValueError(
                    f"Content too large: declared size {declared_size} bytes "
                    f"exceeds limit {self.max_size} bytes"
                )

        chunks: list[bytes] = []
        total_size = 0
        for chunk in response.iter_content(chunk_size=64 * 1024):
            if not chunk:
                continue
            total_size += len(chunk)
            if total_size > self.max_size:
                raise ValueError(
                    f"Content too large: downloaded more than {self.max_size} bytes"
                )
            chunks.append(chunk)
        return b"".join(chunks)

    @staticmethod
    def _looks_like_pdf(content: bytes) -> bool:
        return content.lstrip().startswith(b"%PDF-")

    @staticmethod
    def _looks_like_html(content: bytes) -> bool:
        sample = content[:2048].lstrip().lower()
        return (
            sample.startswith(b"<!doctype html")
            or sample.startswith(b"<html")
            or b"<html" in sample
            or b"<body" in sample
        )

    def _detect_content_kind(
        self,
        *,
        content_type: str,
        content: bytes,
        final_url: str,
    ) -> str:
        if content_type in PDF_CONTENT_TYPES:
            return "pdf"
        if content_type in HTML_CONTENT_TYPES:
            return "html"
        if self._looks_like_pdf(content):
            return "pdf"
        if self._looks_like_html(content):
            return "html"
        if urlparse(final_url).path.lower().endswith(".pdf"):
            return "pdf"
        return "unsupported"

    def fetch(self, url: str) -> FetchedDocument:
        if not self._validate_url(url):
            return self._error_document(url=url, error="Invalid URL or SSRF blocked")

        last_error = "Unknown fetch error"

        for attempt in range(self.retries):
            response: requests.Response | None = None
            try:
                response = self.session.get(
                    url,
                    timeout=self.timeout,
                    stream=True,
                    allow_redirects=True,
                )

                final_url = response.url
                if not self._validate_url(final_url):
                    return self._error_document(
                        url=url,
                        final_url=final_url,
                        error="Redirected to an invalid or blocked URL",
                    )

                status_code = response.status_code

                if status_code in NON_RETRYABLE_STATUS_CODES:
                    error = f"HTTP {status_code}: non-retryable response"
                    logger.warning(
                        "Fetch failed without retry | URL: %s | Final URL: %s | Status: %s",
                        url,
                        final_url,
                        status_code,
                    )
                    return self._error_document(
                        url=url,
                        final_url=final_url,
                        content_type=self._base_content_type(
                            response.headers.get("Content-Type")
                        ),
                        error=error,
                    )

                if status_code in RETRYABLE_STATUS_CODES:
                    last_error = f"HTTP {status_code}: retryable response"
                    if attempt >= self.retries - 1:
                        logger.warning(
                            "Fetch failed after final retry | URL: %s | Final URL: %s | Status: %s",
                            url,
                            final_url,
                            status_code,
                        )
                        return self._error_document(
                            url=url,
                            final_url=final_url,
                            content_type=self._base_content_type(
                                response.headers.get("Content-Type")
                            ),
                            error=last_error,
                        )

                    delay = self._retry_delay(attempt=attempt, response=response)
                    logger.warning(
                        "Retryable HTTP response | URL: %s | Status: %s | Attempt: %s/%s | Retrying in %.1fs",
                        url,
                        status_code,
                        attempt + 1,
                        self.retries,
                        delay,
                    )
                    time.sleep(delay)
                    continue

                if status_code < 200 or status_code >= 300:
                    error = f"HTTP {status_code}: unsupported response status"
                    logger.warning(
                        "Fetch failed without retry | URL: %s | Status: %s",
                        url,
                        status_code,
                    )
                    return self._error_document(
                        url=url,
                        final_url=final_url,
                        content_type=self._base_content_type(
                            response.headers.get("Content-Type")
                        ),
                        error=error,
                    )

                content_type = self._base_content_type(
                    response.headers.get("Content-Type")
                )
                content = self._read_streamed_content(response)
                content_kind = self._detect_content_kind(
                    content_type=content_type,
                    content=content,
                    final_url=final_url,
                )

                if content_kind == "html":
                    encoding = response.encoding or response.apparent_encoding or "utf-8"
                    decoded_html = content.decode(encoding, errors="replace")
                    from app.collector.text_extractor import extract_text
                    text = extract_text(decoded_html)
                    chunks = self._chunk_text(text)
                    return FetchedDocument(
                        final_url=final_url,
                        canonical_url=final_url,
                        content_type="text/html",
                        title=self._extract_title(decoded_html),
                        text=text,
                        text_chunks=chunks,
                        fetch_status="success",
                        fetch_error=None,
                    )

                if content_kind == "pdf":
                    text = self._extract_pdf(content)
                    chunks = self._chunk_text(text)
                    return FetchedDocument(
                        final_url=final_url,
                        canonical_url=final_url,
                        content_type="application/pdf",
                        title=None,
                        text=text,
                        text_chunks=chunks,
                        fetch_status="success",
                        fetch_error=None,
                    )

                error = f"Unsupported content type: {content_type or 'unknown'}"
                logger.info(
                    "Unsupported fetched content | URL: %s | Final URL: %s | Content-Type: %s",
                    url,
                    final_url,
                    content_type or "unknown",
                )
                return self._error_document(
                    url=url,
                    final_url=final_url,
                    content_type=content_type,
                    error=error,
                )

            except (Timeout, RequestsConnectionError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt >= self.retries - 1:
                    logger.warning(
                        "Network fetch failed after final retry | URL: %s | Error: %s",
                        url,
                        last_error,
                    )
                    break
                delay = self._retry_delay(attempt=attempt)
                logger.warning(
                    "Transient network failure | URL: %s | Attempt: %s/%s | Retrying in %.1fs | Error: %s",
                    url,
                    attempt + 1,
                    self.retries,
                    delay,
                    last_error,
                )
                time.sleep(delay)

            except requests.TooManyRedirects as exc:
                last_error = f"Too many redirects: {exc}"
                logger.warning(
                    "Fetch failed without retry | URL: %s | Error: %s",
                    url,
                    last_error,
                )
                break

            except ValueError as exc:
                last_error = str(exc)
                logger.warning(
                    "Fetch failed without retry | URL: %s | Error: %s",
                    url,
                    last_error,
                )
                break

            except RequestException as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "Non-retryable request failure | URL: %s | Error: %s",
                    url,
                    last_error,
                )
                break

            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                logger.exception("Unexpected fetch error | URL: %s", url)
                break

            finally:
                if response is not None:
                    response.close()

        return self._error_document(url=url, error=last_error)

    def _extract_title(self, html: str) -> Optional[str]:
        soup = BeautifulSoup(html, "html.parser")
        title_tag = soup.find("title")
        return title_tag.get_text(strip=True) if title_tag else None

    def _extract_pdf(self, content: bytes) -> str:
        with BytesIO(content) as file:
            reader = PdfReader(file)
            page_text: list[str] = []
            for page_number, page in enumerate(reader.pages, start=1):
                try:
                    text = page.extract_text() or ""
                except Exception as exc:
                    logger.warning(
                        "Failed to extract PDF page %s: %s",
                        page_number,
                        exc,
                    )
                    continue
                if text.strip():
                    page_text.append(text)
            return "\n\n".join(page_text)

    def _chunk_text(self, text: str, max_chars: int = 4000) -> List[str]:
        import re

        cleaned_text = re.sub(r"\s+", " ", text or "").strip()
        if not cleaned_text:
            return []

        sentences = re.split(r"(?<=[.!?])\s+", cleaned_text)
        chunks: list[str] = []
        current_parts: list[str] = []
        current_length = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(sentence) > max_chars:
                if current_parts:
                    chunks.append(" ".join(current_parts))
                    current_parts = []
                    current_length = 0
                for index in range(0, len(sentence), max_chars):
                    chunks.append(sentence[index:index + max_chars].strip())
                continue

            projected_length = current_length + len(sentence) + (1 if current_parts else 0)
            if projected_length <= max_chars:
                current_parts.append(sentence)
                current_length = projected_length
            else:
                chunks.append(" ".join(current_parts))
                current_parts = [sentence]
                current_length = len(sentence)

        if current_parts:
            chunks.append(" ".join(current_parts))

        return chunks