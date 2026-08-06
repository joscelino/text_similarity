"""Testes do utilitário CLI ``text_similarity.tools.migrate_index``."""

from __future__ import annotations

import tempfile
import warnings
from pathlib import Path

import joblib
import numpy as np
import pytest

from text_similarity.core._serialization import (
    INDEX_FORMAT_MAGIC,
    load_index,
)
from text_similarity.core.bm25 import BM25Index
from text_similarity.core.dense import DenseIndex
from text_similarity.tools import migrate_index as migrate_mod

HMAC_KEY_STR = "chave-migracao-teste"
HMAC_KEY = HMAC_KEY_STR.encode("utf-8")


# --- Fixtures -------------------------------------------------------------


def _write_legacy_bm25(path: Path) -> None:
    tmp = BM25Index(k1=1.3, b=0.6).fit(
        ["notebook dell inspiron", "mouse logitech", "monitor samsung"]
    )
    legacy = {
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
        "integrity_hash": "unused",
    }
    joblib.dump(legacy, path)


def _write_legacy_dense(path: Path) -> None:
    rng = np.random.default_rng(seed=7)
    emb = rng.standard_normal((4, 6)).astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.maximum(norms, 1e-10)
    legacy = {
        "version": "1.0",
        "type": "DenseIndex",
        "data": {
            "model_name": "test-model",
            "device": None,
            "precision": "float32",
            "n_documents": 4,
            "embedding_dim": 6,
            "embeddings": emb,
        },
        "integrity_hash": "unused",
    }
    joblib.dump(legacy, path)


# --- (a) BM25 legacy -> v2 -----------------------------------------------


def test_migrate_bm25_legacy_to_v2_via_cli():
    with tempfile.TemporaryDirectory() as tmpdir:
        legacy = Path(tmpdir) / "old.pkl"
        new = Path(tmpdir) / "new.tsbr-index"
        _write_legacy_bm25(legacy)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exit_code = migrate_mod.main(
                [
                    str(legacy),
                    str(new),
                    "--index-type",
                    "bm25",
                    "--hmac-key",
                    HMAC_KEY_STR,
                    "--i-accept-pickle-risk",
                ]
            )
        assert exit_code == 0
        assert new.exists()
        # Header contém magic string
        head = new.read_bytes()[:2048]
        assert INDEX_FORMAT_MAGIC.encode("utf-8") in head.split(b"\n", 1)[0]


# --- (b) Dense legacy -> v2 ----------------------------------------------


def test_migrate_dense_legacy_to_v2_via_cli():
    with tempfile.TemporaryDirectory() as tmpdir:
        legacy = Path(tmpdir) / "old.pkl"
        new = Path(tmpdir) / "new.tsbr-index"
        _write_legacy_dense(legacy)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exit_code = migrate_mod.main(
                [
                    str(legacy),
                    str(new),
                    "--index-type",
                    "dense",
                    "--hmac-key",
                    HMAC_KEY_STR,
                    "--i-accept-pickle-risk",
                ]
            )
        assert exit_code == 0
        assert new.exists()
        head = new.read_bytes()[:2048]
        assert INDEX_FORMAT_MAGIC.encode("utf-8") in head.split(b"\n", 1)[0]


# --- (c) Recusa sem --i-accept-pickle-risk --------------------------------


def test_migrate_aborts_without_pickle_risk_acceptance(capsys):
    with tempfile.TemporaryDirectory() as tmpdir:
        legacy = Path(tmpdir) / "old.pkl"
        new = Path(tmpdir) / "new.tsbr-index"
        _write_legacy_bm25(legacy)

        exit_code = migrate_mod.main(
            [
                str(legacy),
                str(new),
                "--index-type",
                "bm25",
                "--hmac-key",
                HMAC_KEY_STR,
            ]
        )
        assert exit_code == 2
        assert not new.exists()
        stderr = capsys.readouterr().err.lower()
        assert (
            "i-accept-pickle-risk" in stderr or "aceite" in stderr or "risco" in stderr
        )


# --- (d) Round-trip do arquivo migrado via load_index ---------------------


def test_migrated_bm25_roundtrip_load_index():
    with tempfile.TemporaryDirectory() as tmpdir:
        legacy = Path(tmpdir) / "old.pkl"
        new = Path(tmpdir) / "new.tsbr-index"
        _write_legacy_bm25(legacy)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            migrate_mod.migrate(
                legacy,
                new,
                index_type="bm25",
                hmac_key=HMAC_KEY,
                accepted_pickle_risk=True,
            )

        # 1) load_index bruto funciona com HMAC correto
        payload = load_index(new, hmac_key=HMAC_KEY)
        assert payload["type"] == "BM25Index"

        # 2) BM25Index.load também funciona
        loaded = BM25Index.load(new, hmac_key=HMAC_KEY)
        assert loaded.k1 == pytest.approx(1.3)
        assert loaded.b == pytest.approx(0.6)


def test_migrated_dense_roundtrip_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        legacy = Path(tmpdir) / "old.pkl"
        new = Path(tmpdir) / "new.tsbr-index"
        _write_legacy_dense(legacy)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            migrate_mod.migrate(
                legacy,
                new,
                index_type="dense",
                hmac_key=HMAC_KEY,
                accepted_pickle_risk=True,
            )

        loaded = DenseIndex.load(new, hmac_key=HMAC_KEY)
        assert loaded.n_documents == 4
        assert loaded.embedding_dim == 6
        assert loaded._embeddings is not None
        assert loaded._embeddings.shape == (4, 6)


# --- Extras --------------------------------------------------------------


def test_migrate_refuses_missing_legacy_file(capsys):
    with tempfile.TemporaryDirectory() as tmpdir:
        new = Path(tmpdir) / "new.tsbr-index"
        code = migrate_mod.main(
            [
                str(Path(tmpdir) / "missing.pkl"),
                str(new),
                "--index-type",
                "bm25",
                "--i-accept-pickle-risk",
            ]
        )
        assert code == 2
        assert not new.exists()


def test_migrate_refuses_to_overwrite_without_force(capsys):
    with tempfile.TemporaryDirectory() as tmpdir:
        legacy = Path(tmpdir) / "old.pkl"
        new = Path(tmpdir) / "new.tsbr-index"
        _write_legacy_bm25(legacy)
        new.write_bytes(b"{}")

        code = migrate_mod.main(
            [
                str(legacy),
                str(new),
                "--index-type",
                "bm25",
                "--i-accept-pickle-risk",
            ]
        )
        assert code == 2
        # Não sobrescreveu
        assert new.read_bytes() == b"{}"


def test_migrate_reads_hmac_from_env(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        legacy = Path(tmpdir) / "old.pkl"
        new = Path(tmpdir) / "new.tsbr-index"
        _write_legacy_bm25(legacy)

        monkeypatch.setenv("MY_CUSTOM_KEY", "env-secret-key")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            code = migrate_mod.main(
                [
                    str(legacy),
                    str(new),
                    "--index-type",
                    "bm25",
                    "--hmac-env",
                    "MY_CUSTOM_KEY",
                    "--i-accept-pickle-risk",
                ]
            )
        assert code == 0
        # Load com a chave correta funciona
        loaded = BM25Index.load(new, hmac_key=b"env-secret-key")
        assert loaded.k1 == pytest.approx(1.3)
