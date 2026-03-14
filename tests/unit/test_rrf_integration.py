"""Testes de integração do RRF com o Comparator."""

from __future__ import annotations

import pytest

from text_similarity import Comparator, RRFusion


class TestRRFComparatorIntegration:
    """Testes da integração RRF com Comparator.smart()."""

    @pytest.fixture()
    def rrf_comp(self) -> Comparator:
        return Comparator.smart(fusion_strategy="rrf")

    @pytest.fixture()
    def linear_comp(self) -> Comparator:
        return Comparator.smart(fusion_strategy="linear")

    def test_smart_factory_accepts_rrf_strategy(self) -> None:
        comp = Comparator.smart(fusion_strategy="rrf")
        assert comp.fusion_strategy == "rrf"
        assert comp._rrf_fusion is not None
        assert isinstance(comp._rrf_fusion, RRFusion)

    def test_smart_factory_default_is_linear(self) -> None:
        comp = Comparator.smart()
        assert comp.fusion_strategy == "linear"
        assert comp._rrf_fusion is None

    def test_rrf_k_parameter_forwarded(self) -> None:
        comp = Comparator.smart(fusion_strategy="rrf", rrf_k=30)
        assert comp._rrf_fusion is not None
        assert comp._rrf_fusion.k == 30

    def test_pairwise_compare_unaffected_by_rrf(
        self, rrf_comp: Comparator, linear_comp: Comparator
    ) -> None:
        """Pairwise deve ser idêntico independente da estratégia."""
        text1 = "comprei dois quilos de arroz"
        text2 = "adquiri 2 kg de arroz branco"

        score_rrf = rrf_comp.compare(text1, text2)
        score_linear = linear_comp.compare(text1, text2)

        assert score_rrf == pytest.approx(score_linear, abs=1e-6)

    def test_pairwise_explain_unaffected_by_rrf(
        self, rrf_comp: Comparator, linear_comp: Comparator
    ) -> None:
        """explain() deve produzir resultados idênticos independente da estratégia."""
        text1 = "mesa de escritorio"
        text2 = "mesa para escritorio"

        explain_rrf = rrf_comp.explain(text1, text2)
        explain_linear = linear_comp.explain(text1, text2)

        assert explain_rrf["score"] == pytest.approx(explain_linear["score"], abs=1e-6)

    def test_compare_batch_rrf_returns_results(self, rrf_comp: Comparator) -> None:
        """compare_batch com RRF deve retornar resultados válidos."""
        query = "arroz branco tipo 1"
        candidates = [
            "arroz branco 5kg",
            "feijao preto tipo 1",
            "arroz integral organico",
            "macarrao espaguete",
        ]
        results = rrf_comp.compare_batch(query, candidates, top_n=10, min_cosine=0.0)

        assert len(results) > 0
        for r in results:
            assert "candidate" in r
            assert "score" in r
            assert "fusion" in r
            assert r["fusion"] == "rrf"
            assert 0.0 <= r["score"] <= 1.0

    def test_compare_batch_rrf_results_sorted_descending(
        self, rrf_comp: Comparator
    ) -> None:
        """Resultados do RRF devem estar ordenados por score descendente."""
        query = "televisao samsung 50 polegadas"
        candidates = [
            "tv samsung 50 pol led",
            "televisao lg 55 polegadas",
            "monitor samsung 27 pol",
            "controle remoto tv",
        ]
        results = rrf_comp.compare_batch(query, candidates, top_n=10, min_cosine=0.0)

        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_compare_batch_rrf_details_contain_ranks(
        self, rrf_comp: Comparator
    ) -> None:
        """Details no modo RRF devem conter rank e rrf_contribution."""
        query = "cadeira de escritorio"
        candidates = [
            "cadeira giratoria para escritorio",
            "mesa de escritorio",
        ]
        results = rrf_comp.compare_batch(query, candidates, top_n=10, min_cosine=0.0)

        if results:
            details = results[0]["details"]
            for algo_details in details.values():
                assert "rank" in algo_details
                assert "raw_score" in algo_details
                assert "rrf_contribution" in algo_details

    def test_compare_many_to_many_rrf(self, rrf_comp: Comparator) -> None:
        """compare_many_to_many com RRF deve funcionar para múltiplas queries."""
        queries = ["arroz branco", "feijao preto"]
        candidates = [
            "arroz branco tipo 1 5kg",
            "feijao preto tipo 1 1kg",
            "macarrao espaguete 500g",
        ]
        results = rrf_comp.compare_many_to_many(
            queries, candidates, top_n=10, min_cosine=0.0
        )

        assert len(results) == 2
        for query_results in results:
            for r in query_results:
                assert r["fusion"] == "rrf"
                assert 0.0 <= r["score"] <= 1.0


class TestRRFBackwardCompatibility:
    """Garante que o comportamento padrão (linear) permanece inalterado."""

    def test_default_comparator_is_linear(self) -> None:
        comp = Comparator()
        assert comp.fusion_strategy == "linear"

    def test_basic_mode_is_linear(self) -> None:
        comp = Comparator.basic()
        assert comp.fusion_strategy == "linear"

    def test_linear_batch_results_unchanged(self) -> None:
        """Batch linear mantém formato existente (sem 'fusion')."""
        comp = Comparator.smart()
        results = comp.compare_batch(
            "arroz branco", ["arroz integral", "feijao"], top_n=5, min_cosine=0.0
        )

        if results:
            assert "fusion" not in results[0]
            assert "details" in results[0]

    def test_rrf_import_from_package(self) -> None:
        """RRFusion deve ser importável do pacote raiz."""
        from text_similarity import RRFusion as RRF

        assert RRF is not None
        instance = RRF(k=30)
        assert instance.k == 30


class TestRRFWeightsIntegration:
    """Testes de integração dos pesos por algoritmo no RRF via Comparator."""

    def test_smart_factory_accepts_rrf_weights(self) -> None:
        """Comparator.smart() deve aceitar rrf_weights."""
        weights = {"cosine": 0.7, "edit": 0.3}
        comp = Comparator.smart(fusion_strategy="rrf", rrf_weights=weights)
        assert comp.rrf_weights == weights
        assert comp._rrf_fusion is not None
        assert comp._rrf_fusion.weights == weights

    def test_default_rrf_weights_is_none(self) -> None:
        """Sem rrf_weights, o padrão é None (pesos iguais)."""
        comp = Comparator.smart(fusion_strategy="rrf")
        assert comp.rrf_weights is None
        assert comp._rrf_fusion is not None
        assert comp._rrf_fusion.weights is None

    def test_rrf_weights_ignored_on_linear(self) -> None:
        """rrf_weights é ignorado quando fusion_strategy='linear'."""
        comp = Comparator.smart(
            fusion_strategy="linear",
            rrf_weights={"cosine": 0.8, "edit": 0.2},
        )
        assert comp._rrf_fusion is None

    def test_weighted_rrf_batch_returns_valid_results(self) -> None:
        """compare_batch com RRF ponderado deve retornar resultados válidos."""
        comp = Comparator.smart(
            fusion_strategy="rrf",
            rrf_weights={"cosine": 0.7, "edit": 0.2, "phonetic": 0.1},
        )
        results = comp.compare_batch(
            "arroz branco tipo 1",
            ["arroz branco 5kg", "feijao preto", "macarrao"],
            top_n=10,
            min_cosine=0.0,
        )

        assert len(results) > 0
        for r in results:
            assert r["fusion"] == "rrf"
            assert 0.0 <= r["score"] <= 1.0
            for algo_details in r["details"].values():
                assert "weight" in algo_details
