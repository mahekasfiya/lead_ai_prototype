from dataclasses import dataclass

@dataclass
class Document:
    url: str
    title: str
    text: str
    