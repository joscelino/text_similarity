"""Testes para SEC-LOGIC-002: guard de precisão em ``_reuse_semantic_from_dense``.

Cobre o bug em que, com ``dense_precision`` quantizada (``int8``/``binary``),
o ``cos_score`` do DenseIndex — computado sobre embeddings quantizados —
era reutilizado como se fosse o score semântico full-precision. O
comportamento correto é retornar ``False`` na property de reuso,
forçando o :class:`SemanticSimilarity.compare` a recalcular explicitamente
sobre embeddings ``float32``.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from text_similarity.api import Comparator

DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


# --- Guard direto da property ---


def test_reuse_semantic_returns_false_for_int8_precision() -> None:
    """Precisão int8 desabilita o reuso mesmo com mesmo modelo/estratégia."""
    comp = Comparator.smart(
        indexing_strategy="dense",
        use_embeddings=True,
        dense_model_name=DEFAULT_MODEL,
        dense_precision="int8",
    )
    assert comp._reuse_semantic_from_dense is False


def test_reuse_semantic_returns_false_for_binary_precision() -> None:
    """Precisão binary desabilita o reuso mesmo com mesmo modelo/estratégia."""
    comp = Comparator.smart(
        indexing_strategy="dense",
        use_embeddings=True,
        dense_model_name=DEFAULT_MODEL,
        dense_precision="binary",
    )
    assert comp._reuse_semantic_from_dense is False


def test_reuse_semantic_still_true_for_float32_precision() -> None:
    """Precisão float32 mantém o reuso — comportamento pretendido preservado."""
    comp = Comparator.smart(
        indexing_strategy="dense",
        use_embeddings=True,
        dense_model_name=DEFAULT_MODEL,
        dense_precision="float32",
    )
    assert comp._reuse_semantic_from_dense is True


# --- Comportamento observável: score correto vs. score bugado ---


@pytest.mark.parametrize("bad_precision", ["int8", "binary"])
def test_binary_precision_forces_semantic_recompute_not_cos_score(
    bad_precision: str,
) -> None:
    """Em precisão quantizada, o score semântico deve DIFERIR do cos_score.

    O bug antigo reutilizava ``cos_score`` (quantizado) como score
    semântico; o comportamento correto recomputa via
    :meth:`SemanticSimilarity.compare`, que retorna um valor distinto
    (mockado abaixo). O teste falharia se a property voltasse a
    permitir o reuso indevido.
    """
    comp = Comparator.smart(
        indexing_strategy="dense",
        use_embeddings=True,
        dense_model_name=DEFAULT_MODEL,
        dense_precision=bad_precision,
    )
    # Sanity check: property já bloqueou o reuso
    assert comp._reuse_semantic_from_dense is False

    # Simula _score_candidates_linear: com reuse=False, o algoritmo
    # semântico DEVE ser invocado explicitamente com valor diferente
    # do cos_score.
    cos_score_quantized = 0.42  # valor "sujo" vindo do índice quantizado
    semantic_true_score = 0.91  # valor real full-precision

    from text_similarity.core.hybrid import HybridSimilarity

    assert isinstance(comp.algorithm, HybridSimilarity)
    semantic_alg = comp.algorithm.algorithms["semantic"]

    with patch.object(
        semantic_alg, "compare", return_value=semantic_true_score
    ) as mock_compare:
        top_candidates = [
            {
                "candidate": "carro",
                "p_candidate": "carro",
                "cos_score": cos_score_quantized,
            }
        ]
        results = comp._score_candidates_linear("carro esporte", top_candidates)

    # SemanticSimilarity.compare foi chamado (não houve short-circuit
    # via reuse) — prova de que o guard está ativo.
    mock_compare.assert_called_once()

    # O detalhamento do resultado deve conter o score semântico
    # RECALCULADO — não o cos_score quantizado.
    assert results, "esperava ao menos um candidato no resultado"
    details = results[0]["details"]
    assert "semantic" in details
    assert details["semantic"]["score"] == pytest.approx(semantic_true_score)
    # Contra-prova explícita do bug antigo:
    assert details["semantic"]["score"] != pytest.approx(cos_score_quantized)


def test_float32_precision_preserves_reuse_optimization() -> None:
    """Em float32 o reuso segue ativo — otimização não é derrubada."""
    comp = Comparator.smart(
        indexing_strategy="dense",
        use_embeddings=True,
        dense_model_name=DEFAULT_MODEL,
        dense_precision="float32",
    )
    assert comp._reuse_semantic_from_dense is True

    from text_similarity.core.hybrid import HybridSimilarity

    assert isinstance(comp.algorithm, HybridSimilarity)
    semantic_alg = comp.algorithm.algorithms["semantic"]

    cos_score = 0.77
    with patch.object(semantic_alg, "compare", return_value=0.10) as mock_compare:
        top_candidates = [
            {
                "candidate": "carro",
                "p_candidate": "carro",
                "cos_score": cos_score,
            }
        ]
        results = comp._score_candidates_linear("carro esporte", top_candidates)

    # Reuso ativo → SemanticSimilarity.compare NÃO deve ser chamado.
    mock_compare.assert_not_called()

    details = results[0]["details"]
    assert "semantic" in details
    assert details["semantic"]["score"] == pytest.approx(cos_score)
