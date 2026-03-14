"""Testes unitários para o módulo BM25Index."""

import pickle

import numpy as np
import pytest

from text_similarity.core.bm25 import BM25Index


# --- Testes básicos do BM25Index ---


def test_bm25_fit_empty_corpus():
    """BM25 com corpus vazio retorna array vazio para qualquer query."""
    idx = BM25Index()
    idx.fit([])
    scores = idx.get_scores("qualquer busca")
    assert len(scores) == 0


def test_bm25_fit_returns_self():
    """fit() retorna a própria instância para encadeamento."""
    idx = BM25Index()
    result = idx.fit(["texto um", "texto dois"])
    assert result is idx


def test_bm25_scores_exact_match_highest():
    """Documento com match exato deve ter o maior score."""
    corpus = [
        "notebook dell inspiron 15",
        "mouse logitech wireless",
        "monitor samsung 27 4k",
    ]
    idx = BM25Index().fit(corpus)
    scores = idx.get_scores("notebook dell inspiron 15")

    assert scores[0] > scores[1]
    assert scores[0] > scores[2]


def test_bm25_scores_partial_match():
    """Documento com match parcial deve ter score intermediário."""
    corpus = [
        "notebook dell inspiron 15 i5",
        "notebook lenovo thinkpad",
        "cadeira escritorio",
    ]
    idx = BM25Index().fit(corpus)
    scores = idx.get_scores("notebook dell")

    # Ambos notebooks devem ter score > cadeira
    assert scores[0] > scores[2]
    assert scores[1] > scores[2]
    # Dell deve ter score maior que Lenovo (mais termos em comum)
    assert scores[0] > scores[1]


def test_bm25_scores_no_match_zero():
    """Query sem termos em comum retorna scores zero."""
    corpus = ["notebook dell", "mouse logitech"]
    idx = BM25Index().fit(corpus)
    scores = idx.get_scores("cadeira escritorio")

    np.testing.assert_array_equal(scores, np.zeros(2))


def test_bm25_normalized_range_0_to_1():
    """Scores normalizados devem estar no intervalo [0, 1]."""
    corpus = [
        "notebook dell inspiron 15",
        "mouse logitech wireless",
        "monitor samsung 27 4k",
        "teclado mecanico rgb",
    ]
    idx = BM25Index().fit(corpus)
    scores = idx.get_scores_normalized("notebook dell inspiron")

    assert scores.min() >= 0.0
    assert scores.max() <= 1.0
    # O melhor resultado deve ter score normalizado = 1.0
    assert scores.max() == pytest.approx(1.0)


def test_bm25_normalized_no_match_all_zeros():
    """Quando nenhum termo é encontrado, normalização retorna zeros."""
    corpus = ["notebook dell", "mouse logitech"]
    idx = BM25Index().fit(corpus)
    scores = idx.get_scores_normalized("cadeira escritorio")

    np.testing.assert_array_equal(scores, np.zeros(2))


def test_bm25_short_texts_ranking():
    """BM25 rankeia corretamente textos curtos de e-commerce."""
    corpus = [
        "samsung galaxy s22 ultra",
        "samsung galaxy s21",
        "iphone 15 pro max",
        "capa samsung galaxy",
        "pelicula s22",
    ]
    idx = BM25Index().fit(corpus)
    scores = idx.get_scores("samsung galaxy s22")

    # s22 ultra deve ser o melhor (3 termos em comum)
    best_idx = np.argmax(scores)
    assert best_idx == 0


def test_bm25_custom_parameters():
    """Parâmetros k1 e b personalizados são aplicados corretamente."""
    corpus = ["texto curto", "texto medio com mais palavras", "texto longo " * 10]
    idx_default = BM25Index(k1=1.2, b=0.75).fit(corpus)
    idx_custom = BM25Index(k1=1.5, b=0.3).fit(corpus)

    scores_default = idx_default.get_scores("texto")
    scores_custom = idx_custom.get_scores("texto")

    # Com b=0.3, a penalização por comprimento é menor
    # Os scores devem ser diferentes
    assert not np.allclose(scores_default, scores_custom)


def test_bm25_pickle_safe():
    """BM25Index é serializado/desserializado corretamente via pickle."""
    corpus = ["notebook dell inspiron", "mouse logitech", "monitor samsung"]
    idx = BM25Index(k1=1.5, b=0.5).fit(corpus)

    # Serializar e desserializar
    data = pickle.dumps(idx)
    idx_restored = pickle.loads(data)  # noqa: S301

    # Scores devem ser idênticos
    query = "notebook dell"
    original_scores = idx.get_scores(query)
    restored_scores = idx_restored.get_scores(query)

    np.testing.assert_array_equal(original_scores, restored_scores)


def test_bm25_single_doc_corpus():
    """BM25 funciona com corpus de um único documento."""
    idx = BM25Index().fit(["notebook dell inspiron 15"])
    scores = idx.get_scores("notebook dell")

    assert len(scores) == 1
    assert scores[0] > 0


def test_bm25_empty_query():
    """Query vazia retorna scores zero."""
    idx = BM25Index().fit(["notebook dell", "mouse logitech"])
    scores = idx.get_scores("")

    np.testing.assert_array_equal(scores, np.zeros(2))


def test_bm25_empty_documents_in_corpus():
    """Documentos vazios no corpus não causam erro."""
    corpus = ["", "notebook dell", "", "mouse logitech"]
    idx = BM25Index().fit(corpus)
    scores = idx.get_scores("notebook dell")

    assert len(scores) == 4
    assert scores[1] > 0  # "notebook dell" deve ter score positivo
    assert scores[0] == 0  # documento vazio deve ter score zero
