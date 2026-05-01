"""
loaders.py — File loaders for KB articles.

Supported formats: .txt, .md, .html / .htm, .json.
Each loader returns a RawDocument with stripped text + structured metadata
(article_id, title, category, ...). HTML is parsed with BeautifulSoup so
boilerplate, scripts, and tags are removed before chunking.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

from src.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class RawDocument:
    """A loaded source document, before chunking."""

    source: str
    text: str
    metadata: dict = field(default_factory=dict)


class BaseLoader:
    """Subclasses implement `load(path)`."""

    def load(self, path: Path) -> RawDocument:
        raise NotImplementedError


class TextLoader(BaseLoader):
    """.txt files — read as UTF-8, no transformation."""

    def load(self, path: Path) -> RawDocument:
        text = path.read_text(encoding="utf-8", errors="replace")
        return RawDocument(
            source=path.name,
            text=text,
            metadata={"path": str(path), "format": "txt"},
        )


class MarkdownLoader(BaseLoader):
    """.md files — keep markdown structure (headings, lists) so the semantic
    chunker can split on headings."""

    def load(self, path: Path) -> RawDocument:
        text = path.read_text(encoding="utf-8", errors="replace")
        title = self._extract_title(text) or path.stem
        return RawDocument(
            source=path.name,
            text=text,
            metadata={"path": str(path), "format": "md", "title": title},
        )

    @staticmethod
    def _extract_title(text: str) -> str | None:
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("# "):
                return line[2:].strip()
        return None


class HTMLLoader(BaseLoader):
    """.html / .htm — strip tags, scripts, styles; keep visible text."""

    def load(self, path: Path) -> RawDocument:
        from bs4 import BeautifulSoup

        raw = path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        title = soup.title.string.strip() if soup.title and soup.title.string else path.stem
        text = soup.get_text(separator="\n")
        return RawDocument(
            source=path.name,
            text=text,
            metadata={"path": str(path), "format": "html", "title": title},
        )


class JSONLoader(BaseLoader):
    """.json — expects either a single article object or an array of articles.

    Recognized keys: title, body / content / text, category, tags, id.
    The whole loader-pass yields one RawDocument per article.
    """

    def load(self, path: Path) -> list[RawDocument]:
        data = json.loads(path.read_text(encoding="utf-8"))
        items = data if isinstance(data, list) else [data]

        docs: list[RawDocument] = []
        for i, item in enumerate(items):
            body = item.get("body") or item.get("content") or item.get("text", "")
            title = item.get("title", path.stem)
            article_id = item.get("id", f"{path.stem}#{i}")
            text = f"# {title}\n\n{body}"
            docs.append(
                RawDocument(
                    source=f"{path.name}#{article_id}",
                    text=text,
                    metadata={
                        "path": str(path),
                        "format": "json",
                        "article_id": article_id,
                        "title": title,
                        "category": item.get("category", ""),
                        "tags": ",".join(item.get("tags", [])),
                    },
                )
            )
        return docs


# Dispatch

_LOADERS: dict[str, BaseLoader] = {
    ".txt": TextLoader(),
    ".md": MarkdownLoader(),
    ".markdown": MarkdownLoader(),
    ".html": HTMLLoader(),
    ".htm": HTMLLoader(),
    ".json": JSONLoader(),
}


def load_path(path: Path) -> list[RawDocument]:
    """Load one file (or every supported file under a directory).

    JSON loaders may produce multiple docs per file; everything else returns one.
    Unsupported extensions are skipped with a warning.
    """
    if path.is_dir():
        out: list[RawDocument] = []
        for child in sorted(path.rglob("*")):
            if child.is_file():
                out.extend(load_path(child))
        return out

    suffix = path.suffix.lower()
    loader = _LOADERS.get(suffix)
    if loader is None:
        log.warning(f"Unsupported file extension {suffix!r}, skipping {path.name}")
        return []

    try:
        result = loader.load(path)
    except Exception as exc:
        log.error(f"Failed to load {path}: {exc}")
        return []

    return result if isinstance(result, list) else [result]
