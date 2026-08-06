"""Testes de serialização segura do ``DenseIndex`` (NPZ + HMAC)."""

from __future__ import annotations

import json
import tempfile
import warnings
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pytest

from text_similarity.core._serialization import (
    INDEX_FORMAT_MAGIC,
    INDEX_FORMAT_VERSION,
    SecurityWarning,
)
from text_similarity.core.dense import DenseIndex

HMAC_KEY = b"chave-dense-super-secreta-32byte!"


def _make_index_with_fake_embeddings(
    n: int = 3, dim: int = 8, precision: str = "float32"
) -> DenseIndex:
    """Cria um DenseIndex sem depender de sentence-transformers."""
    idx = DenseIndex(precision=precision)
    rng = np.random.default_rng(seed=42)
    emb = rng.standard_normal((n, dim)).astype(np.float32)
    if precision == "float32":
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / np.maximum(norms, 1e-10)
    idx._embeddings = emb
    idx.n_documents = n
    idx.embedding_dim = dim
    return idx


# --- Round-trip -----------------------------------------------------------


def test_dense_roundtrip_save_and_load_with_hmac():
    """Round-trip: save + load recupera embeddings e metadados."""
    idx = _make_index_with_fake_embeddings()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "dense.tsbr-index"
        idx.save(path, hmac_key=HMAC_KEY)
        loaded = DenseIndex.load(path, hmac_key=HMAC_KEY)

    assert loaded.model_name == idx.model_name
    assert loaded.precision == idx.precision
    assert loaded.n_documents == idx.n_documents
    assert loaded.embedding_dim == idx.embedding_dim
    assert loaded._embeddings is not None
    np.testing.assert_array_equal(loaded._embeddings, idx._embeddings)


def test_dense_file_header_is_tsbr_v2():
    """Arquivo Dense começa com header JSON no formato tsbr-index-v2."""
    idx = _make_index_with_fake_embeddings()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "d.bin"
        idx.save(path, hmac_key=HMAC_KEY)
        raw = path.read_bytes()
    header_line, _, _ = raw.partition(b"\n")
    header = json.loads(header_line)
    assert header["format"] == INDEX_FORMAT_MAGIC
    assert header["type"] == "DenseIndex"
    assert header["version"] == INDEX_FORMAT_VERSION


# --- HMAC inválido rejeita antes de np.load -------------------------------


def test_dense_invalid_hmac_rejected_before_np_load():
    """HMAC ruim → ValueError; np.load NÃO é chamado."""
    idx = _make_index_with_fake_embeddings()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "d.bin"
        idx.save(path, hmac_key=HMAC_KEY)

        # Adultera um byte no meio do payload (após o header)
        raw = bytearray(path.read_bytes())
        nl = raw.index(b"\n")
        target = nl + 200 if nl + 200 < len(raw) else len(raw) - 1
        raw[target] ^= 0x01
        path.write_bytes(bytes(raw))

        with patch("text_similarity.core.dense.np.load") as np_load_spy:
            with pytest.raises(ValueError, match="HMAC"):
                DenseIndex.load(path, hmac_key=HMAC_KEY)
            np_load_spy.assert_not_called()


def test_dense_bit_flip_fails_at_hmac_not_at_parse():
    """Bit-flip no NPZ é detectado pelo HMAC, não pelo parser NPZ."""
    idx = _make_index_with_fake_embeddings()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "d.bin"
        idx.save(path, hmac_key=HMAC_KEY)

        raw = bytearray(path.read_bytes())
        # vira o último byte do arquivo (dentro do NPZ)
        raw[-1] ^= 0xFF
        path.write_bytes(bytes(raw))

        with pytest.raises(ValueError) as excinfo:
            DenseIndex.load(path, hmac_key=HMAC_KEY)
        # A mensagem deve mencionar HMAC e não "NPZ" ou "load"
        assert "HMAC" in str(excinfo.value)


def test_dense_wrong_hmac_key_rejected():
    idx = _make_index_with_fake_embeddings()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "d.bin"
        idx.save(path, hmac_key=HMAC_KEY)
        with pytest.raises(ValueError, match="HMAC"):
            DenseIndex.load(path, hmac_key=b"chave-errada")


# --- Arquivo legado (pickle) só carrega com opt-in ------------------------


def _write_legacy_dense(path: Path) -> None:
    idx = _make_index_with_fake_embeddings()
    legacy_payload = {
        "version": "1.0",
        "type": "DenseIndex",
        "data": {
            "model_name": idx.model_name,
            "device": idx.device,
            "precision": idx.precision,
            "n_documents": idx.n_documents,
            "embedding_dim": idx.embedding_dim,
            "embeddings": idx._embeddings,
        },
        "integrity_hash": "fake",
    }
    joblib.dump(legacy_payload, path)


def test_dense_legacy_pickle_rejected_by_default():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "legacy.pkl"
        _write_legacy_dense(path)
        with pytest.raises(ValueError, match="legado"):
            DenseIndex.load(path, hmac_key=HMAC_KEY)


def test_dense_legacy_pickle_emits_security_warning_on_optin():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "legacy.pkl"
        _write_legacy_dense(path)

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            loaded = DenseIndex.load(path, allow_legacy_pickle=True)

        assert loaded.n_documents == 3
        assert any(issubclass(w.category, SecurityWarning) for w in record)


# --- Metadados e uso central de INDEX_FORMAT_VERSION ----------------------


def test_dense_uses_central_index_format_version():
    from text_similarity.core import _serialization, dense

    assert dense.INDEX_FORMAT_VERSION is _serialization.INDEX_FORMAT_VERSION
    assert not hasattr(dense, "_INDEX_VERSION"), (
        "Declaração local _INDEX_VERSION deve ter sido removida."
    )


def test_dense_np_load_uses_allow_pickle_false():
    """np.load é sempre invocado com allow_pickle=False."""
    idx = _make_index_with_fake_embeddings()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "d.bin"
        idx.save(path, hmac_key=HMAC_KEY)

        real_np_load = np.load
        captured_kwargs: list = []

        def spy(*args, **kwargs):
            captured_kwargs.append(kwargs.copy())
            return real_np_load(*args, **kwargs)

        with patch("text_similarity.core.dense.np.load", side_effect=spy):
            DenseIndex.load(path, hmac_key=HMAC_KEY)

    assert captured_kwargs, "np.load deveria ter sido chamado"
    for kw in captured_kwargs:
        assert kw.get("allow_pickle") is False, (
            "np.load DEVE ser chamado com allow_pickle=False"
        )
