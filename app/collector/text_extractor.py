from __future__ import annotations

import trafilatura
from bs4 import BeautifulSoup


def extract_text(
    html: str,
    max_characters: int = 12000,
) -> str:
    """
    Extract main webpage content.

    Trafilatura is used first. BeautifulSoup is the fallback.
    """

    extracted = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=False,
        favor_precision=True,
    )

    if extracted:
        cleaned = " ".join(
            extracted.strip().split()
        )

        return cleaned[:max_characters]

    soup = BeautifulSoup(
        html,
        "html.parser",
    )

    for tag in soup(
        [
            "script",
            "style",
            "nav",
            "footer",
            "header",
            "aside",
            "noscript",
        ]
    ):
        tag.decompose()

    text = soup.get_text(
        separator=" ",
        strip=True,
    )

    cleaned = " ".join(
        text.split()
    )

    return cleaned[:max_characters]