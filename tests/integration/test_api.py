"""Testes de integração da classe Comparator — cobertura expandida."""

from __future__ import annotations

import pytest

from text_similarity.api import Comparator


def test_comparator_basic() -> None:
    comp = Comparator.basic()

    score1 = comp.compare("iphone 13 pro", "iphone pro 13")
    # Bag of words idêntica, levenshtein alterado (modo basic: 0.5 coseno + 0.5 edit)
    assert 0.5 < score1 <= 1.0

    score2 = comp.compare("geladeira electrolux frost free", "foguete espacial da nasa")
    # Completamente diferente
    assert score2 < 0.2


def test_comparator_smart() -> None:
    comp = Comparator.smart()

    # "30 reais" vira <money:30.0>
    # "R$ 30,00" vira <money:30.0>
    score = comp.compare("Custa 30 reais", "O preço é R$ 30,00")

    # Semanticamente parecido devido à normalização de money
    assert score > 0.4


def test_comparator_explain() -> None:
    comp = Comparator.smart()

    result = comp.explain("televisão samsung 55 polegadas", 'tv samsung 55"')

    assert "score" in result
    assert "details" in result
    assert "cosine" in result["details"]
    assert "edit" in result["details"]
    assert "phonetic" in result["details"]

    assert result["score"] >= 0.0


def test_comparator_cache_hit() -> None:
    """O mesmo texto processado duas vezes deve ser retornado do cache."""
    comp = Comparator.basic()
    text = "notebook dell inspiron 15"

    result1 = comp._process(text)
    cache_size_after_first = len(comp._cache_store)

    result2 = comp._process(text)  # deve vir do cache
    cache_size_after_second = len(comp._cache_store)

    assert result1 == result2
    assert cache_size_after_first == cache_size_after_second  # nenhuma entrada nova


def test_comparator_clear_cache() -> None:
    """clear_cache() deve esvaziar o cache in-memory."""
    comp = Comparator.basic()
    comp._process("geladeira frost free")
    comp._process("fogão 4 bocas")
    assert len(comp._cache_store) == 2

    comp.clear_cache()
    assert comp._cache_store == {}


def test_comparator_no_cache() -> None:
    """use_cache=False deve manter o _cache_store sempre vazio."""
    comp = Comparator(use_cache=False)
    comp._process("produto qualquer")
    assert comp.cache is None
    assert comp._cache_store == {}


def test_comparator_empty_strings() -> None:
    """Strings vazias devem retornar 0.0 sem lançar exceção."""
    comp = Comparator.basic()
    assert comp.compare("", "") == pytest.approx(0.0)


def test_comparator_identical_texts() -> None:
    """Textos idênticos devem ter score >= 0.9."""
    comp = Comparator.basic()
    score = comp.compare("samsung galaxy s22 ultra", "samsung galaxy s22 ultra")
    assert score >= 0.9
