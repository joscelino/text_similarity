import pytest

from text_similarity.api import Comparator


@pytest.fixture
def smart_comp():
    return Comparator.smart(entities=["product_model"])


@pytest.fixture
def basic_comp():
    return Comparator.basic()


class TestCompareManyToManyBasic:
    """Testes fundamentais para compare_many_to_many."""

    def test_multiple_queries_returns_list_of_lists(self, smart_comp):
        queries = ["iphone 13", "samsung s22"]
        candidates = [
            "celular iphone 13 novo",
            "samsung galaxy s22 ultra",
            "carregador universal",
        ]

        results = smart_comp.compare_many_to_many(queries, candidates, min_cosine=0.0)

        assert isinstance(results, list)
        assert len(results) == 2
        for query_results in results:
            assert isinstance(query_results, list)

    def test_each_result_has_expected_keys(self, smart_comp):
        queries = ["mesa de madeira"]
        candidates = ["mesa de madeira rustica", "cadeira de ferro"]

        results = smart_comp.compare_many_to_many(queries, candidates, min_cosine=0.0)

        for item in results[0]:
            assert "candidate" in item
            assert "score" in item
            assert "details" in item

    def test_results_ordered_by_score(self, smart_comp):
        queries = ["arroz integral"]
        candidates = [
            "arroz integral tipo 1",
            "feijão preto",
            "arroz parboilizado integral",
            "macarrão",
        ]

        results = smart_comp.compare_many_to_many(queries, candidates, min_cosine=0.0)

        scores = [r["score"] for r in results[0]]
        assert scores == sorted(scores, reverse=True)


class TestCompareManyToManyEdgeCases:
    """Testes de borda e casos vazios."""

    def test_empty_queries_returns_empty(self, smart_comp):
        results = smart_comp.compare_many_to_many(queries=[], candidates=["qualquer"])
        assert results == []

    def test_empty_candidates_returns_empty_lists(self, smart_comp):
        results = smart_comp.compare_many_to_many(
            queries=["query1", "query2"], candidates=[]
        )
        assert len(results) == 2
        assert results[0] == []
        assert results[1] == []

    def test_single_query_single_candidate(self, smart_comp):
        results = smart_comp.compare_many_to_many(
            queries=["notebook dell"], candidates=["notebook dell inspiron"]
        )
        assert len(results) == 1
        assert len(results[0]) == 1
        assert results[0][0]["score"] > 0


class TestCompareManyToManyMinCosineFilter:
    """Testes para filtragem por limiar de cosseno."""

    def test_min_cosine_filters_irrelevant(self, smart_comp):
        queries = ["mesa de madeira maciça"]
        candidates = [
            "cadeira de plastico",
            "processador intel i7",
            "mesa de madeira rustica",
            "livro de receitas",
        ]

        results = smart_comp.compare_many_to_many(queries, candidates, min_cosine=0.2)

        for r in results[0]:
            assert "mesa" in r["candidate"] or "madeira" in r["candidate"]


class TestCompareManyToManyShortCircuit:
    """Testes para short-circuit de entidades."""

    def test_entity_short_circuit(self, smart_comp):
        queries = ["samsung s22 ultra"]
        candidates = [
            "Vende-se samsung galaxy s22 ultra na caixa",
            "samsung s21 barato",
            "televisão samsung 50 polegadas",
        ]

        results = smart_comp.compare_many_to_many(
            queries, candidates, top_n=5, min_cosine=0.0
        )

        top_hit = results[0][0]
        assert "s22 ultra" in top_hit["candidate"].lower()
        if "entity" in top_hit["details"]:
            assert top_hit["details"]["entity"]["score"] == 1.0
            assert top_hit["score"] == 0.95


class TestCompareManyToManyEquivalence:
    """Garante que compare_many_to_many([q], cands) == compare_batch(q, cands)."""

    def test_single_query_matches_batch(self, smart_comp):
        query = "Comprei um iPhone 13"
        candidates = [
            "celular iphone 13 novo",
            "samsung galaxy s22",
            "capa para iphone 11",
            "carregador",
            "comprei o iphone 13 ontem",
        ]

        batch_results = smart_comp.compare_batch(
            query, candidates, top_n=3, min_cosine=0.0
        )
        many_results = smart_comp.compare_many_to_many(
            [query], candidates, top_n=3, min_cosine=0.0
        )[0]

        assert len(batch_results) == len(many_results)
        for b, m in zip(batch_results, many_results):
            assert b["candidate"] == m["candidate"]
            assert abs(b["score"] - m["score"]) < 1e-6

    def test_basic_mode_equivalence(self, basic_comp):
        query = "mesa de escritorio"
        candidates = [
            "mesa de escritório grande",
            "cadeira giratoria",
            "estante de livros",
        ]

        batch_results = basic_comp.compare_batch(
            query, candidates, top_n=5, min_cosine=0.0
        )
        many_results = basic_comp.compare_many_to_many(
            [query], candidates, top_n=5, min_cosine=0.0
        )[0]

        assert len(batch_results) == len(many_results)
        for b, m in zip(batch_results, many_results):
            assert b["candidate"] == m["candidate"]
            assert abs(b["score"] - m["score"]) < 1e-6
