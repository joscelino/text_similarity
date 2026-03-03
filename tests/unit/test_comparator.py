"""Testes unitários da classe Comparator (api.py)."""

from __future__ import annotations

import pytest

from text_similarity.api import Comparator
from text_similarity.core.hybrid import HybridSimilarity


class TestComparatorBasicMode:
    """Testes para o modo básico do Comparator."""

    def test_factory_method_returns_comparator(self) -> None:
        """Comparator.basic() deve retornar uma instância do modo 'basic'."""
        comp = Comparator.basic()
        assert comp.mode == "basic"

    def test_basic_mode_phonetic_weight_is_zero(self) -> None:
        """Modo básico não deve usar fonética (peso 0.0)."""
        comp = Comparator.basic()
        assert isinstance(comp.algorithm, HybridSimilarity)
        assert comp.algorithm.weights.get("phonetic", 0.0) == pytest.approx(0.0)

    def test_basic_mode_uses_cosine_and_edit(self) -> None:
        """Modo básico deve usar cosseno e distância de edição com pesos positivos."""
        comp = Comparator.basic()
        assert isinstance(comp.algorithm, HybridSimilarity)
        assert comp.algorithm.weights["cosine"] > 0.0
        assert comp.algorithm.weights["edit"] > 0.0


class TestComparatorSmartMode:
    """Testes para o modo inteligente do Comparator."""

    def test_factory_method_returns_smart_comparator(self) -> None:
        """Comparator.smart() deve retornar uma instância no modo 'smart'."""
        comp = Comparator.smart()
        assert comp.mode == "smart"

    def test_smart_mode_phonetic_weight_is_positive(self) -> None:
        """Modo smart deve usar fonética com peso positivo."""
        comp = Comparator.smart()
        assert isinstance(comp.algorithm, HybridSimilarity)
        assert comp.algorithm.weights.get("phonetic", 0.0) > 0.0

    def test_smart_mode_accepts_custom_entities(self) -> None:
        """Modo smart deve aceitar entidades personalizadas."""
        comp = Comparator.smart(entities=["money", "date"])
        assert comp.entities == ["money", "date"]


class TestComparatorCache:
    """Testes para o comportamento do cache in-memory."""

    def test_cache_enabled_by_default(self) -> None:
        """Cache deve estar habilitado por padrão."""
        comp = Comparator.basic()
        assert comp.cache is not None
        assert comp.use_cache is True

    def test_cache_disabled_with_flag(self) -> None:
        """use_cache=False deve desabilitar o cache."""
        comp = Comparator(use_cache=False)
        assert comp.cache is None
        assert comp._cache_store == {}

    def test_cache_hit_returns_same_result(self) -> None:
        """Texto já processado deve ser retornado do cache sem reprocessar."""
        comp = Comparator.basic()
        text = "produto samsung galaxy"

        result1 = comp._process(text)
        result2 = comp._process(text)  # deve vir do cache

        assert result1 == result2
        # O cache deve ter uma entrada para este texto
        assert len(comp._cache_store) >= 1

    def test_cache_stores_different_texts_separately(self) -> None:
        """Textos diferentes devem gerar entradas de cache distintas."""
        comp = Comparator.basic()
        comp._process("geladeira frost free")
        comp._process("fogão 4 bocas")

        assert len(comp._cache_store) == 2

    def test_clear_cache_empties_store(self) -> None:
        """clear_cache() deve zerar o dicionário in-memory."""
        comp = Comparator.basic()
        comp._process("produto qualquer")
        assert len(comp._cache_store) >= 1

        comp.clear_cache()
        assert comp._cache_store == {}


class TestComparatorCompare:
    """Testes para o método compare()."""

    def test_compare_returns_float(self) -> None:
        """compare() deve retornar um float."""
        comp = Comparator.basic()
        score = comp.compare("televisão", "televisão")
        assert isinstance(score, float)

    def test_compare_identical_texts_score_near_one(self) -> None:
        """Textos idênticos devem ter score próximo de 1.0."""
        comp = Comparator.basic()
        score = comp.compare("iphone 13 pro max", "iphone 13 pro max")
        assert score >= 0.9

    def test_compare_completely_different_texts_score_near_zero(self) -> None:
        """Textos completamente diferentes devem ter score baixo."""
        comp = Comparator.basic()
        score = comp.compare("geladeira electrolux", "foguete espacial lua")
        assert score < 0.3

    def test_compare_empty_strings_returns_zero(self) -> None:
        """Strings vazias devem retornar score 0.0."""
        comp = Comparator.basic()
        score = comp.compare("", "")
        assert score == pytest.approx(0.0)

    def test_compare_one_empty_string_returns_zero(self) -> None:
        """Uma string vazia deve retornar score 0.0."""
        comp = Comparator.basic()
        assert comp.compare("produto", "") == pytest.approx(0.0)
        assert comp.compare("", "produto") == pytest.approx(0.0)

    def test_compare_score_is_between_zero_and_one(self) -> None:
        """Score deve estar sempre no intervalo [0.0, 1.0]."""
        comp = Comparator.smart()
        pairs = [
            ("notebook dell inspiron", "laptop dell i5"),
            ("R$ 50,00", "cinquenta reais"),
            ("rato", "gato"),
        ]
        for t1, t2 in pairs:
            score = comp.compare(t1, t2)
            assert 0.0 <= score <= 1.0, f"Score fora do range para: '{t1}' x '{t2}'"


class TestComparatorExplain:
    """Testes para o método explain()."""

    def test_explain_returns_score_and_details(self) -> None:
        """explain() deve retornar dict com 'score' e 'details'."""
        comp = Comparator.smart()
        result = comp.explain("notebook samsung", "laptop samsung")
        assert "score" in result
        assert "details" in result

    def test_explain_details_contains_all_algorithms(self) -> None:
        """explain() no modo smart deve detalhar os 3 algoritmos (sem short-circuit)."""
        comp = Comparator.smart()
        # Par sem entidades detectáveis para garantir que não ativa o short-circuit
        result = comp.explain("cadeira de madeira", "mesa de madeira rustica")
        assert "cosine" in result["details"]
        assert "edit" in result["details"]
        assert "phonetic" in result["details"]

    def test_explain_score_is_consistent_with_compare(self) -> None:
        """Score do explain() deve ser igual ao score do compare()."""
        comp = Comparator.smart()
        t1, t2 = "arroz 5kg", "cinco quilos de arroz"
        score_compare = comp.compare(t1, t2)
        score_explain = comp.explain(t1, t2)["score"]
        assert score_compare == pytest.approx(score_explain, abs=1e-6)
