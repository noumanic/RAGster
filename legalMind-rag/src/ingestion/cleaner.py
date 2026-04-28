"""
cleaner.py — Text normalization and cleaning pipeline.
Removes noise, normalizes whitespace, strips PII optionally.
"""
import re
import unicodedata
from dataclasses import dataclass, replace

from src.ingestion.loader import RawDocument
from src.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class CleanerConfig:
    normalize_unicode: bool = True
    remove_extra_whitespace: bool = True
    remove_page_headers_footers: bool = True
    min_line_length: int = 15          # discard lines shorter than this
    max_consecutive_newlines: int = 2


class TextCleaner:
    """Cleans raw extracted text before chunking.

    Handles:
    - Unicode normalization (NFC)
    - Whitespace normalization
    - Removal of PDF header/footer artifacts
    - Removal of boilerplate legal lines (configurable)
    - Line noise filtering
    """

    BOILERPLATE_PATTERNS = [
        r"(?i)confidential\s*[-–]\s*for\s+internal\s+use\s+only",
        r"(?i)page\s+\d+\s+of\s+\d+",
        r"(?i)^\s*\d+\s*$",            # standalone page numbers
        r"DRAFT\s+v?\d+\.\d+",
        r"(?i)all\s+rights\s+reserved",
    ]

    def __init__(self, config: CleanerConfig | None = None) -> None:
        self.config = config or CleanerConfig()
        self._boilerplate_re = [re.compile(p) for p in self.BOILERPLATE_PATTERNS]

    def clean(self, doc: RawDocument) -> RawDocument:
        """Clean text in a RawDocument and return a new RawDocument."""
        text = doc.text

        if self.config.normalize_unicode:
            text = self._normalize_unicode(text)

        text = self._strip_boilerplate(text)

        if self.config.remove_page_headers_footers:
            text = self._strip_page_artifacts(text)

        text = self._filter_short_lines(text, self.config.min_line_length)

        if self.config.remove_extra_whitespace:
            text = self._normalize_whitespace(text, self.config.max_consecutive_newlines)

        original_len = len(doc.text)
        cleaned_len = len(text)
        log.debug(
            "Cleaned document",
            source=doc.source,
            original_chars=original_len,
            cleaned_chars=cleaned_len,
            reduction_pct=round((1 - cleaned_len / max(original_len, 1)) * 100, 1),
        )

        return replace(
            doc,
            text=text.strip(),
            metadata={
                **doc.metadata,
                "cleaned": True,
                "original_char_count": original_len,
                "cleaned_char_count": cleaned_len,
            },
        )

    # Private helpers

    @staticmethod
    def _normalize_unicode(text: str) -> str:
        """NFC normalize and replace common bad chars."""
        text = unicodedata.normalize("NFC", text)
        # Replace common PDF extraction artifacts
        replacements = {
            "\x00": "",       # null bytes
            "\uf8ff": "",     # private use area glyph
            "\u00a0": " ",    # non-breaking space
            "\u2019": "'",    # smart apostrophe
            "\u2018": "'",
            "\u201c": '"',    # smart quotes
            "\u201d": '"',
            "\u2013": "-",    # en-dash
            "\u2014": "-",    # em-dash
            "\u2026": "...",  # ellipsis
        }
        for bad, good in replacements.items():
            text = text.replace(bad, good)
        return text

    def _strip_boilerplate(self, text: str) -> str:
        for pattern in self._boilerplate_re:
            text = pattern.sub("", text)
        return text

    @staticmethod
    def _strip_page_artifacts(text: str) -> str:
        """Remove [Page N] headers injected by the PDF loader."""
        text = re.sub(r"\[Page \d+\]\n?", "", text)
        return text

    @staticmethod
    def _filter_short_lines(text: str, min_len: int) -> str:
        """Remove lines that are too short to carry meaning."""
        lines = text.splitlines()
        kept = [
            line for line in lines
            if len(line.strip()) >= min_len or line.strip() == ""
        ]
        return "\n".join(kept)

    @staticmethod
    def _normalize_whitespace(text: str, max_newlines: int) -> str:
        """Collapse runs of spaces and limit consecutive newlines."""
        # Collapse multiple spaces
        text = re.sub(r" {2,}", " ", text)
        # Collapse runs of newlines
        pattern = r"\n{" + str(max_newlines + 1) + r",}"
        text = re.sub(pattern, "\n" * max_newlines, text)
        # Strip trailing whitespace from each line
        text = "\n".join(line.rstrip() for line in text.splitlines())
        return text
