import pytest

from text_similarity.api import Comparator


@pytest.fixture
def smart_comp():
    return Comparator.smart(entities=["product_model"])


def test_compare_batch_basic(smart_comp):
    query = "Comprei um iPhone 13"
    candidates = [
        "celular iphone 13 novo",
        "samsung galaxy s22",
        "capa para iphone 11",
        "carregador",
        "comprei o iphone 13 ontem",
    ]

    results = smart_comp.compare_batch(query, candidates, top_n=3, min_cosine=0.0)

    assert len(results) == 3
    # "comprei o iphone 13 ontem" e "celular iphone 13 novo" devem estar no topo
    assert "iphone 13" in results[0]["candidate"].lower()
    assert results[0]["score"] > 0.5


def test_compare_batch_short_circuit(smart_comp):
    query = "samsung s22 ultra"
    candidates = [
        "Vende-se samsung galaxy s22 ultra na caixa",
        "samsung s21 barato",
        "televisão samsung 50 polegadas",
        "película s22 ultra",
    ]

    results = smart_comp.compare_batch(query, candidates, top_n=5, min_cosine=0.0)

    assert iter(results)
    assert len(results) == 4

    top_hit = results[0]
    # O candidato contendo exatamente a entidade "s22 ultra" ganha preferência
    assert "s22 ultra" in top_hit["candidate"].lower()
    if "entity" in top_hit["details"]:
        assert top_hit["details"]["entity"]["score"] == 1.0
        assert top_hit["score"] == 0.95


def test_compare_batch_min_cosine_filter(smart_comp):
    query = "mesa de madeira maçica"
    candidates = [
        "cadeira de plastico",
        "livro de receitas",
        "mesa de madeira rustica",
        "mesa de centro de vidro",
        "processador intel i7",
    ]

    # Com um min_cosine de 0.2, os candidatos totalmente irrelavantes sofrem filtro
    results = smart_comp.compare_batch(query, candidates, min_cosine=0.2)

    for r in results:
        # Pelo menos "mesa" ou "madeira" está nos que restam
        assert "mesa" in r["candidate"] or "madeira" in r["candidate"]

    assert len(results) < len(candidates)
