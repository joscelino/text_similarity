# -*- coding: utf-8 -*-
"""Testes unitários para extração de modelos no formato XX.NNN.NN (com ponto).

Cobre Bugs 1, 2 do plano v0.7.1:
- Bug 1: regex do ProductModelExtractor não reconhece ponto como separador
- Bug 2: EntityIntersectionSimilarity usa substring em vez de match exato
"""

from __future__ import annotations

from text_similarity.core.entity_overlap import EntityIntersectionSimilarity
from text_similarity.entities.extractors.product_model import ProductModelExtractor

# ---------------------------------------------------------------------------
# Grupo A — Extração de entidade com ponto (Bug 1)
# ---------------------------------------------------------------------------


def test_should_extract_dot_separated_model_qs():
    """Deve extrair QS.250.08 como UMA ÚNICA entidade productmodel."""
    extractor = ProductModelExtractor()
    matches = extractor.extract("QS.250.08")
    assert len(matches) == 1
    assert matches[0].entity_type == "productmodel"


def test_should_extract_dot_separated_model_qg():
    """Deve extrair QG.418.17 como UMA ÚNICA entidade productmodel."""
    extractor = ProductModelExtractor()
    matches = extractor.extract("QG.418.17")
    assert len(matches) == 1
    assert matches[0].entity_type == "productmodel"


def test_should_extract_model_with_two_dot_separators():
    """Deve extrair modelos no padrão letras.digitos.digitos corretamente."""
    extractor = ProductModelExtractor()
    matches = extractor.extract("AB.123.45")
    assert len(matches) == 1
    assert matches[0].entity_type == "productmodel"


def test_should_not_extract_pure_decimal_as_model():
    """Deve NÃO extrair números decimais puros como modelo (ex: '30.00', '1.5')."""
    extractor = ProductModelExtractor()
    assert extractor.extract("30.00") == []
    assert extractor.extract("1.5") == []


def test_should_not_extract_pure_text_with_dot():
    """Deve NÃO extrair texto puro sem dígitos como modelo (ex: 'abc.def')."""
    extractor = ProductModelExtractor()
    matches = extractor.extract("abc.def")
    assert matches == []


def test_should_still_extract_alphanumeric_without_dot():
    """Deve preservar modelos sem ponto (regressão: S22, XJ-900, RFX765J9)."""
    extractor = ProductModelExtractor()

    assert len(extractor.extract("S22")) == 1
    assert len(extractor.extract("RFX765J9")) == 1

    xj_matches = extractor.extract("XJ-900")
    assert len(xj_matches) == 1


def test_should_extract_qs250_value_normalized():
    """O valor normalizado de QS.250.08 deve existir e identificar o modelo."""
    extractor = ProductModelExtractor()
    matches = extractor.extract("QS.250.08")
    assert len(matches) == 1
    # O valor deve ser algo que identifica unicamente o modelo (sem pontos)
    value = str(matches[0].value)
    assert "QS" in value or "qs" in value.lower()


def test_should_extract_model_in_sentence_context():
    """Deve extrair modelo no meio de uma frase."""
    extractor = ProductModelExtractor()
    matches = extractor.extract("Preciso do modelo QS.250.08 urgente")
    assert len(matches) >= 1
    model_match = next(
        (
            m
            for m in matches
            if "QS" in str(m.value).upper() or "QS" in m.text_matched.upper()
        ),
        None,
    )
    assert model_match is not None


# ---------------------------------------------------------------------------
# Grupo B — Entity Overlap sem falso positivo por substring (Bug 2)
# ---------------------------------------------------------------------------


def test_should_not_match_numeric_substring_as_entity():
    """Deve retornar 0.0 quando '250' é substring de '250080612'."""
    algo = EntityIntersectionSimilarity()
    # Simula textos pós-pipeline com tags de entidade
    text1 = "<productmodel:250>"
    text2 = "<productmodel:250080612>"
    # Com match exato, NÃO deve haver interseção → 0.0
    assert algo.compare(text1, text2) == 0.0


def test_should_return_1_for_exact_entity_match():
    """Deve retornar 1.0 apenas para match exato da tag completa."""
    algo = EntityIntersectionSimilarity()
    text1 = "<productmodel:QS25008>"
    text2 = "<productmodel:QS25008>"
    assert algo.compare(text1, text2) == 1.0


def test_should_return_0_for_prefix_containment():
    """PPW38002 em PPW38002PROFEMUR não dispara short-circuit (tradeoff v0.7.1)."""
    algo = EntityIntersectionSimilarity()
    text1 = "<productmodel:PPW38002>"
    text2 = "<productmodel:PPW38002PROFEMUR>"
    # Com match exato, NÃO é interseção → 0.0
    assert algo.compare(text1, text2) == 0.0


def test_should_return_0_when_no_tags():
    """Deve retornar 0.0 quando nenhum texto tem tags de entidade."""
    algo = EntityIntersectionSimilarity()
    assert algo.compare("qs 250 08", "qs 250 08") == 0.0


def test_should_return_0_when_one_text_has_no_tags():
    """Deve retornar 0.0 quando apenas um texto tem tags."""
    algo = EntityIntersectionSimilarity()
    assert algo.compare("<productmodel:QS25008>", "qs 250 08") == 0.0


def test_should_match_different_entity_types_independently():
    """Tags de tipos diferentes NÃO devem dar match entre si."""
    algo = EntityIntersectionSimilarity()
    text1 = "<money:250>"
    text2 = "<productmodel:250>"
    assert algo.compare(text1, text2) == 0.0
