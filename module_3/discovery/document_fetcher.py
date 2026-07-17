import requests
import logging
from urllib.parse import urlparse
from typing import Optional, List
from io import BytesIO
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from module_3.discovery.models import FetchedDocument

logger = logging.getLogger(__name__)

class DocumentFetcher:
    def __init__(self, timeout: int = 30, max_size: int = 10 * 1024 * 1024, retries: int = 3):
        self.timeout = timeout
        self.max_size = max_size
        self.retries = retries
        self.session = requests.Session()
        self.allowed_schemes = {'http', 'https'}

    def _validate_url(self, url: str) -> bool:
        parsed = urlparse(url)
        if parsed.scheme not in self.allowed_schemes:
            return False
        # Add IP blocking (simplified; use ipaddress in production)
        return True

    def fetch(self, url: str) -> FetchedDocument:
        if not self._validate_url(url):
            return FetchedDocument(
                final_url=url,
                content_type="",
                text="",
                text_chunks=[],
                fetch_status="error",
                fetch_error="Invalid URL or SSRF blocked"
            )
        for attempt in range(self.retries):
            try:
                response = self.session.get(url, timeout=self.timeout, stream=True)
                response.raise_for_status()
                # Check size
                content_length = response.headers.get('content-length')
                if content_length and int(content_length) > self.max_size:
                    raise ValueError("Content too large")
                final_url = response.url
                if not self._validate_url(final_url):
                    raise ValueError("Redirect to blocked URL")
                content_type = response.headers.get('content-type', '').lower()
                if 'html' in content_type:
                    from app.collector.text_extractor import extract_text
                    html_text = response.text
                    text = extract_text(html_text)
                    chunks = self._chunk_text(text)
                    return FetchedDocument(
                        final_url=final_url,
                        canonical_url=final_url,
                        content_type='text/html',
                        title=self._extract_title(html_text),
                        text=text,
                        text_chunks=chunks,
                        fetch_status="success",
                        fetch_error=None
                    )
                elif 'pdf' in content_type:
                    text = self._extract_pdf(response.content)
                    chunks = self._chunk_text(text)
                    return FetchedDocument(
                        final_url=final_url,
                        canonical_url=final_url,
                        content_type='application/pdf',
                        title=None,
                        text=text,
                        text_chunks=chunks,
                        fetch_status="success",
                        fetch_error=None
                    )
                else:
                    raise ValueError(f"Unsupported content type: {content_type}")
            except Exception as e:
                logger.warning(f"Fetch attempt {attempt+1} failed: {e}")
                if attempt == self.retries - 1:
                    return FetchedDocument(
                        final_url=url,
                        content_type="",
                        text="",
                        text_chunks=[],
                        fetch_status="error",
                        fetch_error=str(e)
                    )

    def _extract_title(self, html: str) -> Optional[str]:
        soup = BeautifulSoup(html, 'html.parser')
        title_tag = soup.find('title')
        return title_tag.get_text(strip=True) if title_tag else None

    def _extract_pdf(self, content: bytes) -> str:
        with BytesIO(content) as f:
            reader = PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text

    def _chunk_text(self, text: str, max_chars: int = 4000) -> List[str]:
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current = ""
        for sent in sentences:
            if len(current) + len(sent) < max_chars:
                current += sent + " "
            else:
                if current:
                    chunks.append(current.strip())
                current = sent + " "
        if current:
            chunks.append(current.strip())
        return chunks