"""Testes unitários para o módulo DenseIndex."""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest

from text_similarity.core.dense import DenseIndex

# --- Testes básicos do DenseIndex ---


def test_dense_index_fit_returns_self():
    """fit() retorna a própria instância para encadeamento."""
    idx = DenseIndex()
    result = idx.fit(["texto um", "texto dois"])
    assert result is idx


def test_dense_index_fit_and_query():
    """fit() codifica corpus e query retorna scores com shape e range corretos."""
    corpus = [
        "notebook dell inspiron 15",
        "mouse logitech wireless",
        "monitor samsung 27 4k",
    ]
    idx = DenseIndex().fit(corpus)
    scores = idx.get_scores_normalized("notebook dell")

    assert scores.shape == (3,)
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0


def test_dense_index_empty_corpus():
    """DenseIndex com corpus vazio retorna array vazio."""
    idx = DenseIndex()
    scores = idx.get_scores_normalized("qualquer busca")
    assert len(scores) == 0


def test_dense_index_exact_match_highest_score():
    """Documento idêntico à query deve ter o maior score."""
    corpus = [
        "notebook dell inspiron 15",
        "mouse logitech wireless",
        "monitor samsung 27 4k",
    ]
    idx = DenseIndex().fit(corpus)
    scores = idx.get_scores_normalized("notebook dell inspiron 15")

    best_idx = np.argmax(scores)
    assert best_idx == 0


def test_dense_index_pickle_safe():
    """DenseIndex é serializado/desserializado via pickle."""
    corpus = [
        "notebook dell inspiron",
        "mouse logitech",
        "monitor samsung",
    ]
    idx = DenseIndex().fit(corpus)

    data = pickle.dumps(idx)
    idx_restored = pickle.loads(data)  # noqa: S301

    query = "notebook dell"
    original_scores = idx.get_scores_normalized(query)
    restored_scores = idx_restored.get_scores_normalized(query)

    np.testing.assert_array_almost_equal(original_scores, restored_scores)


def test_dense_semantic_recall():
    """Embeddings densos capturam sinônimos sem overlap lexical.

    Este é o caso de uso principal: 'veículo flex' deve ter
    alta similaridade com 'carro bicombustível' mesmo sem
    nenhuma palavra em comum.
    """
    corpus = [
        "carro bicombustível",
        "notebook dell inspiron",
        "cadeira escritório ergonômica",
        "mesa de jantar madeira",
    ]
    idx = DenseIndex().fit(corpus)
    scores = idx.get_scores_normalized("veículo flex")

    # "carro bicombustível" deve ser o melhor resultado
    best_idx = int(np.argmax(scores))
    assert best_idx == 0, (
        f"Esperava índice 0 ('carro bicombustível'), "
        f"obteve {best_idx} com scores {scores}"
    )
    # Score deve ser significativamente maior que os demais
    assert scores[0] > scores[1]
    assert scores[0] > scores[2]
    assert scores[0] > scores[3]


def test_comparator_dense_strategy():
    """Comparator com indexing_strategy='dense' funciona no batch."""
    from text_similarity.api import Comparator

    comp = Comparator.smart(indexing_strategy="dense")

    candidates = [
        "carro bicombustível",
        "notebook dell inspiron",
        "cadeira escritório ergonômica",
        "mesa de jantar madeira",
        "smartphone samsung galaxy",
    ]
    results = comp.compare_batch(
        "veículo flex",
        candidates,
        top_n=5,
        min_cosine=0.0,
    )

    assert len(results) > 0
    for r in results:
        assert 0.0 <= r["score"] <= 1.0
        assert "candidate" in r
        assert "details" in r


def test_comparator_dense_many_to_many():
    """compare_many_to_many com dense retorna resultados coerentes."""
    from text_similarity.api import Comparator

    comp = Comparator.smart(indexing_strategy="dense")

    queries = ["veículo flex", "computador portátil"]
    candidates = [
        "carro bicombustível",
        "notebook dell inspiron",
        "cadeira escritório",
    ]
    results = comp.compare_many_to_many(queries, candidates, top_n=3, min_cosine=0.0)

    assert len(results) == 2
    for query_results in results:
        assert isinstance(query_results, list)
        for r in query_results:
            assert 0.0 <= r["score"] <= 1.0


# --- Testes de serialização ---


def test_should_save_and_load_dense_index_with_same_scores():
    """Should save and load dense index with same scores."""
    corpus = ["notebook dell inspiron", "mouse logitech", "monitor samsung"]
    idx = DenseIndex().fit(corpus)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "dense_idx.pkl"
        idx.save(path)
        idx_loaded = DenseIndex.load(path)

    query = "notebook dell"
    original_scores = idx.get_scores_normalized(query)
    loaded_scores = idx_loaded.get_scores_normalized(query)

    np.testing.assert_array_almost_equal(original_scores, loaded_scores, decimal=5)
    assert idx_loaded.model_name == idx.model_name
    assert idx_loaded.precision == "float32"


def test_should_reject_corrupted_index_file():
    """Should reject corrupted dense index file."""
    import joblib

    corpus = ["notebook dell"]
    idx = DenseIndex().fit(corpus)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "dense_idx.pkl"
        idx.save(path)

        payload = joblib.load(path)
        payload["integrity_hash"] = "hash_invalido"
        joblib.dump(payload, path)

        with pytest.raises(ValueError, match="corrompido"):
            DenseIndex.load(path)


def test_should_reject_version_mismatch():
    """Should reject dense index version mismatch."""
    import joblib

    corpus = ["notebook dell"]
    idx = DenseIndex().fit(corpus)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "dense_idx.pkl"
        idx.save(path)

        payload = joblib.load(path)
        payload["version"] = "99.0"
        joblib.dump(payload, path)

        with pytest.raises(ValueError, match="Versão incompatível"):
            DenseIndex.load(path)


# --- Testes de quantização ---


def test_should_fit_and_score_with_int8_precision():
    """Should fit and score with int8 precision."""
    corpus = ["notebook dell inspiron", "mouse logitech", "monitor samsung"]
    idx = DenseIndex(precision="int8").fit(corpus)

    assert idx._embeddings is not None
    assert idx._embeddings.dtype == np.int8
    scores = idx.get_scores_normalized("notebook dell")
    assert scores.shape == (3,)
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0


def test_should_fit_and_score_with_binary_precision():
    """Should fit and score with binary precision."""
    corpus = ["notebook dell inspiron", "mouse logitech", "monitor samsung"]
    idx = DenseIndex(precision="binary").fit(corpus)

    assert idx._embeddings is not None
    assert idx._embeddings.dtype in (np.uint8, np.int8)
    scores = idx.get_scores_normalized("notebook dell")
    assert scores.shape == (3,)
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0


def test_should_produce_smaller_memory_footprint_with_int8():
    """Should produce smaller memory footprint with int8."""
    corpus = ["notebook dell inspiron " * 5, "mouse logitech " * 5]
    idx_f32 = DenseIndex(precision="float32").fit(corpus)
    idx_i8 = DenseIndex(precision="int8").fit(corpus)

    assert idx_f32._embeddings is not None
    assert idx_i8._embeddings is not None
    size_f32 = idx_f32._embeddings.nbytes
    size_i8 = idx_i8._embeddings.nbytes
    assert size_i8 < size_f32


def test_should_save_and_load_int8_index_correctly():
    """Should save and load int8 index correctly."""
    corpus = ["notebook dell inspiron", "mouse logitech", "monitor samsung"]
    idx = DenseIndex(precision="int8").fit(corpus)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "dense_i8.pkl"
        idx.save(path)
        idx_loaded = DenseIndex.load(path)

    assert idx_loaded.precision == "int8"
    assert idx_loaded._embeddings is not None
    assert idx_loaded._embeddings.dtype == np.int8

    scores_orig = idx.get_scores_normalized("notebook dell")
    scores_loaded = idx_loaded.get_scores_normalized("notebook dell")
    np.testing.assert_array_almost_equal(scores_orig, scores_loaded, decimal=5)
