"""Testes para o short-circuit semântico quando indexing_strategy='dense'."""

from __future__ import annotations

from unittest.mock import patch

from text_similarity.api import Comparator

DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
OTHER_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

CANDIDATES = [
    "carro bicombustível",
    "notebook dell inspiron",
    "cadeira escritório ergonômica",
]


# --- _reuse_semantic_from_dense ---


def test_should_reuse_semantic_when_dense_and_same_model() -> None:
    """_reuse_semantic_from_dense é True quando dense + use_embeddings + mesmo modelo."""  # noqa: E501
    comp = Comparator.smart(
        indexing_strategy="dense",
        use_embeddings=True,
        dense_model_name=DEFAULT_MODEL,
    )
    assert comp._reuse_semantic_from_dense is True


def test_should_not_reuse_semantic_when_strategy_is_tfidf() -> None:
    """_reuse_semantic_from_dense é False quando indexing_strategy='tfidf'."""
    comp = Comparator.smart(
        indexing_strategy="tfidf",
        use_embeddings=True,
    )
    assert comp._reuse_semantic_from_dense is False


def test_should_not_reuse_semantic_when_strategy_is_bm25() -> None:
    """_reuse_semantic_from_dense é False quando indexing_strategy='bm25'."""
    comp = Comparator.smart(
        indexing_strategy="bm25",
        use_embeddings=True,
    )
    assert comp._reuse_semantic_from_dense is False


def test_should_not_reuse_semantic_when_models_differ() -> None:
    """_reuse_semantic_from_dense é False quando modelos são diferentes."""
    comp = Comparator.smart(
        indexing_strategy="dense",
        use_embeddings=True,
        dense_model_name=OTHER_MODEL,
    )
    # SemanticSimilarity usa DEFAULT_MODEL, DenseIndex usa OTHER_MODEL
    assert comp._reuse_semantic_from_dense is False


def test_should_not_reuse_semantic_when_embeddings_disabled() -> None:
    """_reuse_semantic_from_dense é False quando use_embeddings=False."""
    comp = Comparator.smart(
        indexing_strategy="dense",
        use_embeddings=False,
    )
    assert comp._reuse_semantic_from_dense is False


# --- Verificação de que SemanticSimilarity.compare NÃO é chamado ---


def test_should_not_call_semantic_compare_when_reuse_is_active() -> None:
    """SemanticSimilarity.compare não deve ser chamado quando reuse_semantic=True."""
    comp = Comparator.smart(
        indexing_strategy="dense",
        use_embeddings=True,
        dense_model_name=DEFAULT_MODEL,
    )

    semantic_alg = comp.algorithm.algorithms.get("semantic")
    assert semantic_alg is not None, "SemanticSimilarity não foi instanciado"

    with patch.object(semantic_alg, "compare", wraps=semantic_alg.compare) as mock_cmp:
        comp.compare_batch("veículo flex", CANDIDATES, top_n=3, min_cosine=0.0)
        mock_cmp.assert_not_called()


def test_should_call_semantic_compare_when_reuse_is_inactive() -> None:
    """SemanticSimilarity.compare deve ser chamado quando reuse_semantic=False."""
    comp = Comparator.smart(
        indexing_strategy="tfidf",
        use_embeddings=True,
    )

    semantic_alg = comp.algorithm.algorithms.get("semantic")
    assert semantic_alg is not None

    with patch.object(semantic_alg, "compare", wraps=semantic_alg.compare) as mock_cmp:
        comp.compare_batch("veículo flex", CANDIDATES, top_n=3, min_cosine=0.0)
        assert mock_cmp.call_count > 0


# --- RRF ---


def test_should_not_call_semantic_compare_in_rrf_when_reuse_is_active() -> None:
    """SemanticSimilarity.compare não é chamado no RRF quando reuse_semantic=True."""
    comp = Comparator.smart(
        indexing_strategy="dense",
        use_embeddings=True,
        dense_model_name=DEFAULT_MODEL,
        fusion_strategy="rrf",
    )

    semantic_alg = comp.algorithm.algorithms.get("semantic")
    assert semantic_alg is not None

    with patch.object(semantic_alg, "compare", wraps=semantic_alg.compare) as mock_cmp:
        comp.compare_batch("veículo flex", CANDIDATES, top_n=3, min_cosine=0.0)
        mock_cmp.assert_not_called()
