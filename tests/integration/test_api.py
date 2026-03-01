
from text_similarity.api import Comparator


def test_comparator_basic() -> None:
    comp = Comparator.basic()

    score1 = comp.compare("iphone 13 pro", "iphone pro 13")
    # Bag of words idêntica, levenshtein alterado (+ fonetica se mode=smart)
    # Como mode=basic usa pesos 0.5 cosseno e 0.5 levenshtein
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

    result = comp.explain("televisão samsung 55 polegadas", "tv samsung 55\"")

    assert "score" in result
    assert "details" in result
    assert "cosine" in result["details"]
    assert "edit" in result["details"]
    assert "phonetic" in result["details"]

    assert result["score"] >= 0.0
