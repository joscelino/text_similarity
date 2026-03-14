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


def test_compare_batch_strategy_vectorized(smart_comp):
    query = "Comprei um iPhone 13"
    candidates = [
        "celular iphone 13 novo",
        "samsung galaxy s22",
        "comprei o iphone 13 ontem",
    ]

    results = smart_comp.compare_batch(
        query, candidates, top_n=3, min_cosine=0.0, strategy="vectorized"
    )

    assert len(results) > 0
    assert "iphone 13" in results[0]["candidate"].lower()


def test_compare_batch_invalid_strategy(smart_comp):
    with pytest.raises(ValueError, match="não suportada"):
        smart_comp.compare_batch("qualquer", ["texto"], strategy="invalida")


def test_compare_batch_empty_candidates(smart_comp):
    results = smart_comp.compare_batch("qualquer", [])
    assert results == []


# --- Testes com indexing_strategy="bm25" ---


@pytest.fixture
def bm25_comp():
    return Comparator.smart(entities=["product_model"], indexing_strategy="bm25")


def test_compare_batch_bm25_returns_results(bm25_comp):
    """BM25 retorna resultados válidos em compare_batch."""
    query = "Comprei um iPhone 13"
    candidates = [
        "celular iphone 13 novo",
        "samsung galaxy s22",
        "capa para iphone 11",
        "carregador",
        "comprei o iphone 13 ontem",
    ]

    results = bm25_comp.compare_batch(query, candidates, top_n=3, min_cosine=0.0)

    assert len(results) > 0
    assert results[0]["score"] > 0.0
    assert "candidate" in results[0]
    assert "details" in results[0]


def test_compare_many_bm25_returns_results(bm25_comp):
    """BM25 retorna resultados válidos em compare_many_to_many."""
    queries = ["iPhone 13", "Galaxy S22"]
    candidates = [
        "celular iphone 13 novo",
        "samsung galaxy s22 ultra",
        "notebook dell inspiron",
    ]

    results = bm25_comp.compare_many_to_many(
        queries, candidates, top_n=3, min_cosine=0.0
    )

    assert len(results) == 2
    assert len(results[0]) > 0
    assert len(results[1]) > 0


def test_compare_batch_bm25_with_entities(bm25_comp):
    """BM25 funciona com extração de entidades."""
    query = "samsung s22 ultra"
    candidates = [
        "Vende-se samsung galaxy s22 ultra na caixa",
        "samsung s21 barato",
        "película s22 ultra",
    ]

    results = bm25_comp.compare_batch(query, candidates, top_n=5, min_cosine=0.0)

    assert len(results) > 0
    # Candidato com "s22 ultra" deve estar no topo
    assert "s22 ultra" in results[0]["candidate"].lower()


def test_compare_batch_bm25_empty_candidates(bm25_comp):
    """BM25 retorna lista vazia para candidatos vazios."""
    results = bm25_comp.compare_batch("qualquer", [])
    assert results == []


def test_compare_batch_bm25_custom_params():
    """BM25 aceita parâmetros k1 e b customizados."""
    comp = Comparator.smart(
        entities=["product_model"],
        indexing_strategy="bm25",
        bm25_k1=1.5,
        bm25_b=0.3,
    )

    results = comp.compare_batch(
        "notebook dell", ["notebook dell inspiron", "mouse"], top_n=2, min_cosine=0.0
    )

    assert len(results) > 0
