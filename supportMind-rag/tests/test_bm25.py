"""Test the BM25 keyword index — tokenizer behavior + retrieval ordering."""
from pathlib import Path

from src.ingestion.chunker import Chunk
from src.retrieval.bm25_index import BM25Index, tokenize


def test_tokenizer_preserves_technical_tokens():
    toks = tokenize("Error E_47 in v2.3 over wi-fi/USB-C — see ssh-keygen output.")
    # rare tech tokens must survive intact
    assert "e_47" in toks
    assert "v2.3" in toks
    assert "wi-fi/usb-c" in toks
    assert "ssh-keygen" in toks
    # stopwords gone
    assert "in" not in toks
    assert "the" not in toks


def test_tokenizer_drops_short_tokens_and_stopwords():
    toks = tokenize("a the of for to charging device port works ok")
    # all the stopwords are filtered
    for word in ("a", "the", "of", "for", "to"):
        assert word not in toks
    # content words remain
    assert "charging" in toks
    assert "device" in toks


def test_bm25_index_ranks_exact_terms_first(tmp_path: Path, monkeypatch):
    # redirect persistence to a tmp file
    from src.utils import config

    config.get_settings.cache_clear()  # type: ignore
    monkeypatch.setenv("BM25_INDEX_PATH", str(tmp_path / "bm25.pkl"))
    config.get_settings.cache_clear()  # type: ignore

    idx = BM25Index()
    # BM25's IDF needs a corpus larger than 2 docs to produce non-zero scores
    # for half-frequency terms — that's why we add several distractors.
    chunks = [
        Chunk(
            chunk_id="c1",
            text="The Helios X4 returns error code E_FW_017 on signature verification failure.",
            source="firmware.md",
            chunk_index=0,
            total_chunks=1,
            metadata={},
        ),
        Chunk(
            chunk_id="c2",
            text="To pair a new device, open the companion app and scan the QR code.",
            source="setup.md",
            chunk_index=0,
            total_chunks=1,
            metadata={},
        ),
        Chunk(
            chunk_id="c3",
            text="Hold the power button for 15 seconds to hard-reset the charger.",
            source="charging.md",
            chunk_index=0,
            total_chunks=1,
            metadata={},
        ),
        Chunk(
            chunk_id="c4",
            text="Use a wooden toothpick to clear lint from the USB-C port.",
            source="charging.md",
            chunk_index=1,
            total_chunks=1,
            metadata={},
        ),
        Chunk(
            chunk_id="c5",
            text="Subscriptions can be cancelled at any time from the billing portal.",
            source="billing.md",
            chunk_index=0,
            total_chunks=1,
            metadata={},
        ),
    ]
    idx.add(chunks)
    hits = idx.search("E_FW_017 firmware signature", top_k=5)
    assert hits, "expected BM25 hits for the exact error code"
    assert hits[0].chunk_id == "c1"
    assert hits[0].score > 0
