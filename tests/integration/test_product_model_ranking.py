# -*- coding: utf-8 -*-
"""Testes de integração para ranking de modelos no formato XX.NNN.NN.

Cobre Bugs 1, 2, 3, 4 do plano v0.7.1:
- Pipeline completo com entities=["product_model"]
- Verificação de que o candidato exato é top-1 com TF-IDF e BM25
"""

from __future__ import annotations

import pytest

from text_similarity.api import Comparator

# ---------------------------------------------------------------------------
# Grupo C — Pipeline integrado com TF-IDF (padrão)
# ---------------------------------------------------------------------------

QS_CANDIDATES = [
    "QS.250.08",
    "250.080.612",
    "250.080.588",
    "250.080.342",
    "QS.250.12",
    "QS.300.08",
]

QG_CANDIDATES = [
    "QG.418.17",
    "12.5.0418.17OE",
    "QG.418.20",
    "QG.500.17",
    "418.170.001",
]


@pytest.fixture
def tfidf_comp():
    return Comparator.smart(entities=["product_model"], indexing_strategy="tfidf")


@pytest.fixture
def bm25_comp():
    return Comparator.smart(entities=["product_model"], indexing_strategy="bm25")


def test_tfidf_exact_model_triggers_short_circuit(tfidf_comp):
    """QS.250.08 vs QS.250.08 deve gerar short-circuit com product_model."""
    score = tfidf_comp.compare("QS.250.08", "QS.250.08")
    assert score >= 0.90, f"Score esperado >= 0.90, obtido {score}"


def test_tfidf_partial_number_does_not_trigger_short_circuit(tfidf_comp):
    """QS.250.08 vs 250.080.612 NÃO deve gerar short-circuit."""
    score = tfidf_comp.compare("QS.250.08", "250.080.612")
    # Score deve ser baixo — sem short-circuit, sem entidade em comum
    assert score < 0.90, f"Score não deveria ser alto, obtido {score}"


def test_tfidf_compare_batch_ranks_exact_model_first_qs250(tfidf_comp):
    """Comparator.smart(entities=["product_model"]): QS.250.08 exato deve ser top-1."""
    results = tfidf_comp.compare_batch(
        "QS.250.08", QS_CANDIDATES, top_n=6, min_cosine=0.0
    )
    assert len(results) > 0
    assert results[0]["candidate"] == "QS.250.08", (
        f"Esperado top-1='QS.250.08', obtido '{results[0]['candidate']}'. "
        f"Ranking completo: {[r['candidate'] for r in results]}"
    )


def test_tfidf_compare_batch_ranks_exact_model_first_qg418(tfidf_comp):
    """Comparator.smart(entities=["product_model"]): QG.418.17 exato deve ser top-1."""
    results = tfidf_comp.compare_batch(
        "QG.418.17", QG_CANDIDATES, top_n=5, min_cosine=0.0
    )
    assert len(results) > 0
    assert results[0]["candidate"] == "QG.418.17", (
        f"Esperado top-1='QG.418.17', obtido '{results[0]['candidate']}'. "
        f"Ranking completo: {[r['candidate'] for r in results]}"
    )


# ---------------------------------------------------------------------------
# Grupo D — Pipeline integrado com BM25
# ---------------------------------------------------------------------------


def test_bm25_compare_batch_ranks_exact_model_first_qs250(bm25_comp):
    """BM25: QS.250.08 → candidato exato deve ser top-1."""
    results = bm25_comp.compare_batch(
        "QS.250.08", QS_CANDIDATES, top_n=6, min_cosine=0.0
    )
    assert len(results) > 0
    assert results[0]["candidate"] == "QS.250.08", (
        f"Esperado top-1='QS.250.08', obtido '{results[0]['candidate']}'. "
        f"Ranking completo: {[r['candidate'] for r in results]}"
    )


def test_bm25_compare_batch_ranks_exact_model_first_qg418(bm25_comp):
    """BM25: QG.418.17 → candidato exato deve ser top-1."""
    results = bm25_comp.compare_batch(
        "QG.418.17", QG_CANDIDATES, top_n=5, min_cosine=0.0
    )
    assert len(results) > 0
    assert results[0]["candidate"] == "QG.418.17", (
        f"Esperado top-1='QG.418.17', obtido '{results[0]['candidate']}'. "
        f"Ranking completo: {[r['candidate'] for r in results]}"
    )


def test_bm25_wrong_candidate_is_not_top_when_exact_exists(bm25_comp):
    """BM25: candidato errado (250.080.612) NÃO deve ser top-1 quando o exato existe."""
    results = bm25_comp.compare_batch(
        "QS.250.08", QS_CANDIDATES, top_n=6, min_cosine=0.0
    )
    assert len(results) > 0
    top_candidate = results[0]["candidate"]
    assert top_candidate != "250.080.612", (
        f"Falso positivo '250.080.612' foi top-1. "
        f"Ranking: {[r['candidate'] for r in results]}"
    )


def test_bm25_exact_model_score_above_threshold(bm25_comp):
    """BM25: score do candidato exato deve ser >= 0.90."""
    results = bm25_comp.compare_batch(
        "QS.250.08", QS_CANDIDATES, top_n=6, min_cosine=0.0
    )
    exact = next((r for r in results if r["candidate"] == "QS.250.08"), None)
    assert exact is not None, "Candidato exato 'QS.250.08' não apareceu nos resultados"
    assert exact["score"] >= 0.90, (
        f"Score do exato esperado >= 0.90, obtido {exact['score']}"
    )
