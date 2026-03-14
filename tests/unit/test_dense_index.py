"""Testes unitários para o módulo DenseIndex."""

from __future__ import annotations

import pickle

import numpy as np

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
