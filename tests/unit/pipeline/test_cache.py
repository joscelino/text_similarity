"""Testes do ``PipelineCache`` em formato JSON UTF-8 (não mais pickle)."""

from __future__ import annotations

import json
import pickle
import tempfile
import warnings
from pathlib import Path

import pytest

from text_similarity.core._serialization import SecurityWarning
from text_similarity.pipeline.cache import PipelineCache


def test_pipeline_cache_roundtrip_json():
    """save_catalog + load_catalog: round-trip em JSON."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "catalog.json"
        cache = PipelineCache(cache_dir=tmpdir)
        candidates = ["produto um", "produto dois"]
        processed = ["produto 1 preprocessado", "produto 2 preprocessado"]
        cache.save_catalog(candidates, processed, str(cache_path))
        loaded = cache.load_catalog(candidates, str(cache_path))
    assert loaded == processed


def test_pipeline_cache_file_is_valid_json():
    """O arquivo gravado é JSON válido (sem pickle)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "catalog.json"
        cache = PipelineCache(cache_dir=tmpdir)
        cache.save_catalog(["a", "b"], ["a1", "b1"], str(cache_path))
        raw = cache_path.read_bytes()
    # JSON começa com '{'
    assert raw.lstrip().startswith(b"{")
    obj = json.loads(raw)
    assert obj["processed"] == ["a1", "b1"]
    assert "catalog_hash" in obj
    assert "version" in obj


def test_pipeline_cache_invalidates_on_different_catalog():
    """Se o catálogo muda, load_catalog retorna None."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "catalog.json"
        cache = PipelineCache(cache_dir=tmpdir)
        cache.save_catalog(["a", "b"], ["a1", "b1"], str(cache_path))
        loaded = cache.load_catalog(["a", "c"], str(cache_path))
    assert loaded is None


def test_pipeline_cache_missing_file_returns_none():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = PipelineCache(cache_dir=tmpdir)
        loaded = cache.load_catalog(["a"], str(Path(tmpdir) / "missing.json"))
    assert loaded is None


def test_pipeline_cache_rejects_legacy_pickle_file():
    """Arquivo pickle legado emite SecurityWarning e é ignorado."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "legacy.pkl"
        legacy_data = {
            "version": "1.0",
            "catalog_hash": "irrelevant",
            "processed": ["should", "not", "load"],
        }
        with open(cache_path, "wb") as fh:
            pickle.dump(legacy_data, fh, protocol=pickle.HIGHEST_PROTOCOL)

        cache = PipelineCache(cache_dir=tmpdir)
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            loaded = cache.load_catalog(["a"], str(cache_path))

        assert loaded is None
        assert any(issubclass(w.category, SecurityWarning) for w in record)


def test_pipeline_cache_malformed_json_raises_value_error():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "bad.json"
        cache_path.write_text("{ not really json", encoding="utf-8")
        cache = PipelineCache(cache_dir=tmpdir)
        with pytest.raises(ValueError, match="JSON"):
            cache.load_catalog(["a"], str(cache_path))


def test_pipeline_cache_no_noqa_s301_in_source():
    """O comentário ``# noqa: S301`` foi removido do módulo."""
    import inspect

    from text_similarity.pipeline import cache as cache_module

    source = inspect.getsource(cache_module)
    assert "# noqa: S301" not in source
    # E não há mais chamadas executáveis a pickle.load/pickle.dump
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith('"'):
            continue
        assert "pickle.load(" not in stripped
        assert "pickle.dump(" not in stripped
