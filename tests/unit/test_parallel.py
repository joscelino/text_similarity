import pytest

from text_similarity.api import Comparator


@pytest.fixture
def smart_comp():
    return Comparator.smart(entities=["product_model"])


@pytest.fixture
def basic_comp():
    return Comparator.basic()


class TestParallelMatchesVectorized:
    """Garante que strategy='parallel' produz resultados idênticos a 'vectorized'."""

    def test_equivalence_smart_mode(self, smart_comp):
        queries = [
            "Comprei um iPhone 13",
            "samsung s22 ultra",
            "mesa de madeira maciça",
        ]
        candidates = [
            "celular iphone 13 novo",
            "samsung galaxy s22 ultra na caixa",
            "mesa de madeira rustica",
            "carregador universal",
            "comprei o iphone 13 ontem",
        ]

        vec_results = smart_comp.compare_many_to_many(
            queries, candidates, top_n=5, min_cosine=0.0,
            strategy="vectorized",
        )
        par_results = smart_comp.compare_many_to_many(
            queries, candidates, top_n=5, min_cosine=0.0,
            strategy="parallel", n_workers=2,
        )

        assert len(vec_results) == len(par_results)
        for v_list, p_list in zip(vec_results, par_results):
            assert len(v_list) == len(p_list)
            for v, p in zip(v_list, p_list):
                assert v["candidate"] == p["candidate"]
                assert abs(v["score"] - p["score"]) < 1e-6

    def test_equivalence_basic_mode(self, basic_comp):
        queries = ["mesa de escritório", "cadeira giratória"]
        candidates = [
            "mesa de escritório grande",
            "cadeira giratória preta",
            "estante de livros",
        ]

        vec_results = basic_comp.compare_many_to_many(
            queries, candidates, top_n=5, min_cosine=0.0,
            strategy="vectorized",
        )
        par_results = basic_comp.compare_many_to_many(
            queries, candidates, top_n=5, min_cosine=0.0,
            strategy="parallel", n_workers=2,
        )

        assert len(vec_results) == len(par_results)
        for v_list, p_list in zip(vec_results, par_results):
            assert len(v_list) == len(p_list)
            for v, p in zip(v_list, p_list):
                assert v["candidate"] == p["candidate"]
                assert abs(v["score"] - p["score"]) < 1e-6


class TestParallelCompareBatch:
    """Testes de compare_batch com strategy='parallel'."""

    def test_batch_parallel_basic(self, smart_comp):
        query = "Comprei um iPhone 13"
        candidates = [
            "celular iphone 13 novo",
            "samsung galaxy s22",
            "comprei o iphone 13 ontem",
        ]

        results = smart_comp.compare_batch(
            query, candidates, top_n=3, min_cosine=0.0,
            strategy="parallel", n_workers=2,
        )

        assert len(results) > 0
        assert "iphone 13" in results[0]["candidate"].lower()

    def test_batch_parallel_matches_vectorized(self, smart_comp):
        query = "notebook dell"
        candidates = [
            "notebook dell inspiron",
            "mouse logitech",
            "teclado microsoft",
        ]

        vec_results = smart_comp.compare_batch(
            query, candidates, top_n=5, min_cosine=0.0,
            strategy="vectorized",
        )
        par_results = smart_comp.compare_batch(
            query, candidates, top_n=5, min_cosine=0.0,
            strategy="parallel", n_workers=2,
        )

        assert len(vec_results) == len(par_results)
        for v, p in zip(vec_results, par_results):
            assert v["candidate"] == p["candidate"]
            assert abs(v["score"] - p["score"]) < 1e-6


class TestParallelEdgeCases:
    """Testes de borda para strategy='parallel'."""

    def test_empty_queries(self, smart_comp):
        results = smart_comp.compare_many_to_many(
            queries=[], candidates=["qualquer"],
            strategy="parallel", n_workers=2,
        )
        assert results == []

    def test_empty_candidates(self, smart_comp):
        results = smart_comp.compare_many_to_many(
            queries=["query1", "query2"], candidates=[],
            strategy="parallel", n_workers=2,
        )
        assert len(results) == 2
        assert results[0] == []
        assert results[1] == []

    def test_n_workers_1(self, smart_comp):
        """Com n_workers=1, deve funcionar sem multiprocessing."""
        queries = ["iphone 13"]
        candidates = ["celular iphone 13", "samsung s22"]

        results = smart_comp.compare_many_to_many(
            queries, candidates, top_n=5, min_cosine=0.0,
            strategy="parallel", n_workers=1,
        )

        assert len(results) == 1
        assert len(results[0]) > 0

    def test_single_candidate(self, smart_comp):
        results = smart_comp.compare_many_to_many(
            queries=["teste"],
            candidates=["teste único"],
            strategy="parallel", n_workers=2,
        )
        assert len(results) == 1


class TestParallelShortCircuit:
    """Testa short-circuit de entidades no mode parallel."""

    def test_entity_short_circuit_parallel(self, smart_comp):
        queries = ["samsung s22 ultra"]
        candidates = [
            "Vende-se samsung galaxy s22 ultra na caixa",
            "samsung s21 barato",
            "televisão samsung 50 polegadas",
        ]

        results = smart_comp.compare_many_to_many(
            queries, candidates, top_n=5, min_cosine=0.0,
            strategy="parallel", n_workers=2,
        )

        top_hit = results[0][0]
        assert "s22 ultra" in top_hit["candidate"].lower()
        if "entity" in top_hit["details"]:
            assert top_hit["details"]["entity"]["score"] == 1.0
            assert top_hit["score"] == 0.95
