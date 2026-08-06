"""Testes de serialização segura do ``BM25Index`` (formato tsbr-index-v2)."""

from __future__ import annotations

import json
import pickle
import tempfile
import warnings
from pathlib import Path

import joblib
import numpy as np
import pytest

from text_similarity.core._serialization import (
    INDEX_FORMAT_MAGIC,
    INDEX_FORMAT_VERSION,
    SecurityWarning,
)
from text_similarity.core.bm25 import BM25Index

HMAC_KEY = b"chave-de-teste-super-secreta-32b!"


# --- Round-trip -----------------------------------------------------------


def test_bm25_roundtrip_save_and_load_with_hmac():
    """Round-trip: save + load com HMAC produz índice idêntico."""
    corpus = ["notebook dell inspiron", "mouse logitech", "monitor samsung"]
    idx = BM25Index(k1=1.5, b=0.5).fit(corpus)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "idx.tsbr-index"
        idx.save(path, hmac_key=HMAC_KEY)
        loaded = BM25Index.load(path, hmac_key=HMAC_KEY)

    original = idx.get_scores("notebook dell")
    restored = loaded.get_scores("notebook dell")
    np.testing.assert_array_almost_equal(original, restored)
    assert loaded.k1 == pytest.approx(1.5)
    assert loaded.b == pytest.approx(0.5)


def test_bm25_file_starts_with_json_header():
    """O arquivo novo começa com header JSON tsbr-index-v2."""
    idx = BM25Index().fit(["um dois", "tres quatro"])
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "idx.bin"
        idx.save(path, hmac_key=HMAC_KEY)
        raw = path.read_bytes()
    assert raw.startswith(b"{")
    header_line, _, _ = raw.partition(b"\n")
    header = json.loads(header_line)
    assert header["format"] == INDEX_FORMAT_MAGIC
    assert header["version"] == INDEX_FORMAT_VERSION
    assert header["type"] == "BM25Index"
    assert isinstance(header["hmac"], str) and len(header["hmac"]) == 64


# --- HMAC inválido rejeita antes do parse ---------------------------------


def test_bm25_load_rejects_invalid_hmac_before_parse(monkeypatch):
    """HMAC inválido → ValueError; ``json.loads`` do payload NÃO é invocado."""
    idx = BM25Index().fit(["a b c", "d e f"])
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "idx.bin"
        idx.save(path, hmac_key=HMAC_KEY)

        # Adultera 1 byte do payload (após o header + '\n')
        raw = bytearray(path.read_bytes())
        nl = raw.index(b"\n")
        # Vira o primeiro byte do payload para outro valor válido
        raw[nl + 1] = ord(b"X") if raw[nl + 1] != ord(b"X") else ord(b"Y")
        path.write_bytes(bytes(raw))

        # Espiona json.loads para provar que não foi invocado
        # sobre o payload adulterado
        original_loads = json.loads
        calls: list = []

        def spy_loads(s, *a, **kw):
            calls.append(s)
            return original_loads(s, *a, **kw)

        monkeypatch.setattr(json, "loads", spy_loads)

        with pytest.raises(ValueError, match="HMAC"):
            BM25Index.load(path, hmac_key=HMAC_KEY)

    # O único parse JSON esperado é do header (curto, começa com '{')
    for call in calls:
        if isinstance(call, (bytes, bytearray)):
            call_str = bytes(call).decode("utf-8", errors="replace")
        else:
            call_str = call
        assert call_str.startswith("{"), (
            "Somente header JSON deve ser parseado antes do HMAC"
        )
        # Header tem chaves fixas
        parsed = original_loads(call_str)
        assert set(parsed.keys()) == {"format", "type", "version", "hmac"}


def test_bm25_load_rejects_wrong_hmac_key():
    """Chave HMAC errada é detectada e levanta ValueError."""
    idx = BM25Index().fit(["a b c"])
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "idx.bin"
        idx.save(path, hmac_key=HMAC_KEY)

        with pytest.raises(ValueError, match="HMAC"):
            BM25Index.load(path, hmac_key=b"chave-errada")


# --- Arquivo legado (pickle) só carrega com flag explícita ----------------


def _write_legacy_bm25(path: Path) -> None:
    corpus = ["notebook dell inspiron"]
    tmp = BM25Index().fit(corpus)
    legacy_payload = {
        "version": "1.0",
        "type": "BM25Index",
        "data": {
            "k1": tmp.k1,
            "b": tmp.b,
            "corpus_size": tmp._corpus_size,
            "avgdl": tmp._avgdl,
            "doc_freqs": tmp._doc_freqs,
            "doc_lens": tmp._doc_lens,
            "term_freqs": tmp._term_freqs,
        },
        "integrity_hash": "fake",
    }
    joblib.dump(legacy_payload, path)


def test_bm25_legacy_pickle_rejected_without_optin():
    """Arquivo legado (joblib) sem allow_legacy_pickle levanta ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "legacy.pkl"
        _write_legacy_bm25(path)
        with pytest.raises(ValueError, match="legado"):
            BM25Index.load(path)


def test_bm25_legacy_pickle_via_raw_pickle_dump_rejected():
    """Um pickle 'puro' (não joblib) também é rejeitado por padrão."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "raw.pkl"
        with open(path, "wb") as fh:
            pickle.dump({"anything": 1}, fh)
        with pytest.raises(ValueError, match="legado"):
            BM25Index.load(path)


def test_bm25_legacy_pickle_optin_emits_security_warning():
    """Com allow_legacy_pickle=True, emite SecurityWarning e carrega."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "legacy.pkl"
        _write_legacy_bm25(path)

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            # O integrity_hash "fake" fará load falhar, mas o SecurityWarning
            # deve ter sido emitido ANTES.
            with pytest.raises(ValueError):
                BM25Index.load(path, allow_legacy_pickle=True)

        assert any(issubclass(w.category, SecurityWarning) for w in record), (
            "SecurityWarning deveria ter sido emitido para arquivo legado."
        )


def test_bm25_save_without_hmac_emits_warning():
    """Salvar sem hmac_key emite warnings.warn (arquivo fica sem autenticação)."""
    idx = BM25Index().fit(["a b"])
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "noauth.bin"
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            idx.save(path)
        assert any("HMAC" in str(w.message) for w in record)


def test_bm25_load_without_hmac_key_warns_but_loads():
    """Sem chave HMAC no load, warn é emitido mas o índice é carregado."""
    idx = BM25Index().fit(["cadeira escritorio"])
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "idx.bin"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            idx.save(path)  # sem hmac
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            loaded = BM25Index.load(path)  # sem hmac
        assert loaded._corpus_size == 1
        assert any("HMAC" in str(w.message) for w in record)


# --- INDEX_FORMAT_VERSION importado do módulo central ---------------------


def test_bm25_uses_central_index_format_version():
    """bm25.INDEX_FORMAT_VERSION vem de _serialization.py."""
    from text_similarity.core import _serialization, bm25

    assert bm25.INDEX_FORMAT_VERSION is _serialization.INDEX_FORMAT_VERSION
    assert not hasattr(bm25, "_INDEX_VERSION"), (
        "Declaração local _INDEX_VERSION deve ter sido removida."
    )
