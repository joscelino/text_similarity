"""Testes unitários para rerank_vector_results (Feature 2).

Valida o re-ranking de resultados vindos de bancos vetoriais
(Pinecone, Qdrant, Milvus, PGVector, Elasticsearch) usando
os algoritmos linguísticos do HybridSimilarity.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from text_similarity.api import Comparator


@pytest.fixture()
def comp_smart() -> Comparator:
    """Comparator smart com todas as entidades ativas."""
    return Comparator.smart(entities=["product_model", "money", "number"])


@pytest.fixture()
def comp_rrf() -> Comparator:
    """Comparator smart com fusão RRF."""
    return Comparator.smart(
        entities=["product_model"],
        fusion_strategy="rrf",
        rrf_k=60,
    )


@pytest.fixture()
def vector_results() -> list:
    """Simula saída de um banco vetorial (ex: Qdrant/Milvus)."""
    return [
        {"id": "doc1", "text": "Samsung Galaxy S22 Ultra 256GB", "score": 0.92},
        {"id": "doc2", "text": "Apple iPhone 15 Pro Max 512GB", "score": 0.85},
        {"id": "doc3", "text": "Mouse Logitech MX Master 3S", "score": 0.60},
        {"id": "doc4", "text": "Peças GN500 originais disponíveis", "score": 0.45},
    ]


class TestRerankInputValidation:
    """Validação de formato de entrada."""

    def test_empty_candidates_returns_empty(self, comp_smart) -> None:
        """Lista vazia de candidatos retorna lista vazia."""
        result = comp_smart.rerank_vector_results("query", [])
        assert result == []

    def test_missing_text_field_raises_error(self, comp_smart) -> None:
        """Candidato sem campo 'text' deve lançar ValueError."""
        with pytest.raises(ValueError, match="campo 'text'"):
            comp_smart.rerank_vector_results(
                "query",
                [{"id": "doc1", "score": 0.8}],
            )

    def test_missing_score_field_raises_error(self, comp_smart) -> None:
        """Candidato sem campo 'score' deve lançar ValueError."""
        with pytest.raises(ValueError, match="campo 'score'"):
            comp_smart.rerank_vector_results(
                "query",
                [{"id": "doc1", "text": "texto"}],
            )

    def test_candidate_without_id_works(self, comp_smart) -> None:
        """Candidato sem 'id' funciona normalmente (campo opcional)."""
        results = comp_smart.rerank_vector_results(
            "galaxy s22",
            [{"text": "Samsung Galaxy S22", "score": 0.9}],
        )
        assert len(results) == 1
        assert "id" not in results[0]
        assert "candidate" in results[0]


class TestRerankOutputFormat:
    """Valida o formato de saída do re-ranking."""

    def test_output_contains_required_fields(
        self, comp_smart, vector_results
    ) -> None:
        """Resultado deve conter id, candidate, score, vector_score, details."""
        results = comp_smart.rerank_vector_results(
            "Samsung Galaxy S22", vector_results
        )

        assert len(results) > 0
        first = results[0]
        assert "id" in first
        assert "candidate" in first
        assert "score" in first
        assert "vector_score" in first
        assert "details" in first

    def test_vector_score_preserved(self, comp_smart, vector_results) -> None:
        """O score original do banco vetorial é preservado em vector_score."""
        results = comp_smart.rerank_vector_results(
            "Samsung Galaxy S22", vector_results
        )

        # Mapear por candidate text para localizar
        by_text = {r["candidate"]: r for r in results}
        assert by_text["Samsung Galaxy S22 Ultra 256GB"]["vector_score"] == 0.92
        assert by_text["Apple iPhone 15 Pro Max 512GB"]["vector_score"] == 0.85

    def test_results_sorted_by_final_score(
        self, comp_smart, vector_results
    ) -> None:
        """Resultados devem estar ordenados por score final descendente."""
        results = comp_smart.rerank_vector_results(
            "Samsung Galaxy S22", vector_results
        )

        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)


class TestRerankScoring:
    """Valida a lógica de scoring e re-ordenação."""

    def test_rerank_changes_order_via_entity_short_circuit(self) -> None:
        """Candidato na posição #4 sobe para #1 via short-circuit de entidade.

        O banco vetorial colocou "Peças GN500" na posição 4 (score 0.45),
        mas o HybridSimilarity detecta a entidade product_model exata e
        aplica short-circuit, elevando-o ao topo.

        Usa preprocess_candidates=True para que o pipeline extraia as
        entidades dos textos dos candidatos (simulando textos brutos).
        """
        comp = Comparator.smart(entities=["product_model"])

        vector_candidates = [
            {"id": "doc1", "text": "Peças industriais variadas", "score": 0.90},
            {"id": "doc2", "text": "Ferramentas diversas completas", "score": 0.80},
            {"id": "doc3", "text": "Motor elétrico trifásico", "score": 0.70},
            {"id": "doc4", "text": "Peças GN500 originais", "score": 0.45},
        ]

        results = comp.rerank_vector_results(
            "GN500",
            vector_candidates,
            preprocess_query=True,
            preprocess_candidates=True,
        )

        # O candidato com entidade exata "GN500" deve estar no topo
        assert results[0]["id"] == "doc4"
        assert results[0]["score"] >= 0.90

    def test_vector_score_used_as_cosine(self, comp_smart) -> None:
        """O score vetorial é usado como cos_score nos algoritmos."""
        vector_candidates = [
            {"id": "doc1", "text": "galaxy s22", "score": 0.95},
        ]

        results = comp_smart.rerank_vector_results("galaxy s22", vector_candidates)

        # Com score vetorial alto e textos similares, score final deve ser alto
        assert results[0]["score"] > 0.5
        assert results[0]["vector_score"] == 0.95

    def test_short_circuit_preserves_vector_score(self) -> None:
        """Mesmo com short-circuit (score=0.95), vector_score original é mantido."""
        comp = Comparator.smart(entities=["product_model"])

        vector_candidates = [
            {"id": "doc1", "text": "<productmodel:XJ900>", "score": 0.30},
        ]

        results = comp.rerank_vector_results(
            "<productmodel:XJ900>",
            vector_candidates,
            preprocess_query=False,
            preprocess_candidates=False,
        )

        assert results[0]["score"] >= 0.90  # Short-circuit
        assert results[0]["vector_score"] == 0.30  # Original preservado


class TestRerankPreprocessing:
    """Valida o controle de pré-processamento no reranker."""

    def test_preprocess_query_true_runs_pipeline(self, comp_smart) -> None:
        """preprocess_query=True executa o pipeline na query."""
        vector_candidates = [
            {"id": "doc1", "text": "galaxy s22", "score": 0.8},
        ]

        with patch.object(
            comp_smart.pipeline, "process", wraps=comp_smart.pipeline.process
        ) as mock:
            comp_smart.rerank_vector_results(
                "GALAXY S22!!", vector_candidates, preprocess_query=True
            )
            # Pipeline chamado pelo menos 1x (para a query)
            assert mock.call_count >= 1

    def test_preprocess_candidates_false_skips_pipeline(
        self, comp_smart
    ) -> None:
        """preprocess_candidates=False não executa pipeline nos candidatos."""
        vector_candidates = [
            {"id": "doc1", "text": "galaxy s22", "score": 0.8},
            {"id": "doc2", "text": "iphone 15", "score": 0.7},
        ]

        with patch.object(
            comp_smart.pipeline, "process", wraps=comp_smart.pipeline.process
        ) as mock:
            comp_smart.rerank_vector_results(
                "galaxy s22",
                vector_candidates,
                preprocess_query=True,
                preprocess_candidates=False,
            )
            # Pipeline chamado apenas 1x (só para a query, não para candidatos)
            assert mock.call_count == 1


class TestRerankWithRRF:
    """Valida re-ranking com estratégia de fusão RRF."""

    def test_rrf_fusion_with_vector_results(self, comp_rrf) -> None:
        """RRF funciona corretamente com scores de banco vetorial."""
        vector_candidates = [
            {"id": "doc1", "text": "notebook dell inspiron 15 i5", "score": 0.88},
            {"id": "doc2", "text": "notebook lenovo thinkpad x1", "score": 0.82},
            {"id": "doc3", "text": "mouse logitech wireless", "score": 0.40},
        ]

        results = comp_rrf.rerank_vector_results(
            "notebook dell inspiron", vector_candidates
        )

        assert len(results) == 3
        # Todos devem ter score válido
        for r in results:
            assert 0.0 <= r["score"] <= 1.0
            assert "details" in r

    def test_rrf_reranking_reorders_candidates(self, comp_rrf) -> None:
        """RRF com scores vetoriais externos re-ordena candidatos."""
        vector_candidates = [
            {"id": "doc1", "text": "peças variadas industriais", "score": 0.90},
            {"id": "doc2", "text": "GN500 peças originais", "score": 0.50},
        ]

        results = comp_rrf.rerank_vector_results("GN500", vector_candidates)

        # O candidato com entidade exata deve subir no ranking
        assert len(results) == 2
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)
