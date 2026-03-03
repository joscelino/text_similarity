# -*- coding: utf-8 -*-
from __future__ import annotations

from text_similarity.entities.extractors.date import DateExtractor
from text_similarity.entities.extractors.dimension import DimensionExtractor
from text_similarity.entities.extractors.money import MoneyExtractor
from text_similarity.entities.extractors.number import NumberExtractor
from text_similarity.entities.extractors.product_model import ProductModelExtractor
from text_similarity.entities.inspector import EntityInspector
from text_similarity.entities.normalizer import EntityNormalizer


def test_date_extractor_relative() -> None:
    extractor = DateExtractor()
    text = "Eu comprei isso ontem e vou devolver amanhã."

    matches = extractor.extract(text)

    assert len(matches) == 2
    matched_texts = [m.text_matched.lower() for m in matches]

    assert "ontem" in matched_texts
    assert "amanhã" in matched_texts

    for m in matches:
        assert m.entity_type == "date"


def test_date_extractor_absolute() -> None:
    extractor = DateExtractor()
    text = "O evento será dia 25 de abril de 2024 às 10h."

    matches = extractor.extract(text)
    assert len(matches) == 1

    m = matches[0]
    assert m.value == "2024-04-25"
    assert "25 de abril de 2024" in m.text_matched.lower()


def test_money_extractor() -> None:
    extractor = MoneyExtractor()
    text = "Custou R$ 1.500,50 e outro de 30 reais."
    matches = extractor.extract(text)

    assert len(matches) == 2
    assert matches[0].value == 1500.50
    assert matches[1].value == 30.0


def test_dimension_extractor() -> None:
    extractor = DimensionExtractor()
    text = "Comprei 2kg de arroz e 1.5l de suco."
    matches = extractor.extract(text)

    assert len(matches) == 2
    assert matches[0].value == "2.0:kg"
    assert matches[1].value == "1.5:l"


def test_number_extractor() -> None:
    extractor = NumberExtractor()
    text = "Eu tenho três gatos e 2 cachorros."
    matches = extractor.extract(text)

    assert len(matches) == 2
    assert matches[0].value == 3  # "três"
    assert matches[1].value == 2  # "2"


def test_product_model_extractor() -> None:
    extractor = ProductModelExtractor()
    text = "Temos o S22 Ultra e o iPhone 13. Também a peça XJ-900."
    matches = extractor.extract(text)

    # "S22", "13" (o número 13 atende à regex de número, mas não
    # a de let+num do produto sozinho, 'iPhone 13' não vai bater
    # a menos que mude a regex, vamos ver), "XJ-900"
    # Nossa regex exige \d+[-][A-Za-z] ou [A-Za-z]+[-]\d+
    assert len(matches) >= 2
    vals = [m.value for m in matches]
    assert "S22" in vals
    assert "XJ-900" in vals


def test_inspector_all() -> None:
    inspector = EntityInspector()
    text = "Comprei ontem 2kg por R$ 50,00."
    matches = inspector.inspect(text)

    # ontem (date)
    # 2kg (dimension e number -> sobreposição na extração "2" como número)
    # R$ 50,00 (money e number -> sobreposição 50.00 como número)
    # O Inspector não filtra sobreposições, então ele deve achar várias entidades
    assert len(matches) > 0
    types = [m.entity_type for m in matches]
    assert "date" in types
    assert "dimension" in types
    assert "money" in types


def test_normalizer() -> None:
    normalizer = EntityNormalizer()
    text = "Custou 30 reais na promoção de ontem."
    # 30 reais -> <money:30.0>
    # ontem -> <date:XXXX-XX-XX>
    normalized = normalizer.normalize(text)

    assert "<money:30.0>" in normalized
    assert "<date:" in normalized
    # A palavra 'ontem' deve ter sumido do texto
    assert "ontem" not in normalized
