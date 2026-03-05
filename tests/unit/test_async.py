import pytest

from text_similarity.api import Comparator


@pytest.fixture
def smart_comp():
    return Comparator.smart(entities=["product_model"])


@pytest.fixture
def basic_comp():
    return Comparator.basic()


@pytest.mark.asyncio
class TestCompareBatchAsync:
    """Testes para compare_batch_async."""

    async def test_basic_results(self, smart_comp):
        query = "Comprei um iPhone 13"
        candidates = [
            "celular iphone 13 novo",
            "samsung galaxy s22",
            "comprei o iphone 13 ontem",
        ]

        results = await smart_comp.compare_batch_async(
            query, candidates, top_n=3, min_cosine=0.0, n_workers=1,
        )

        assert len(results) > 0
        assert all("candidate" in r and "score" in r for r in results)

    async def test_matches_sync(self, smart_comp):
        query = "notebook dell inspiron"
        candidates = [
            "notebook dell inspiron 15",
            "mouse logitech wireless",
            "teclado microsoft ergonômico",
        ]

        sync_results = smart_comp.compare_batch(
            query, candidates, top_n=5, min_cosine=0.0,
            strategy="vectorized",
        )
        async_results = await smart_comp.compare_batch_async(
            query, candidates, top_n=5, min_cosine=0.0, n_workers=1,
        )

        assert len(sync_results) == len(async_results)
        for s, a in zip(sync_results, async_results):
            assert s["candidate"] == a["candidate"]
            assert abs(s["score"] - a["score"]) < 1e-6

    async def test_empty_candidates(self, smart_comp):
        results = await smart_comp.compare_batch_async(
            "qualquer", [], n_workers=1,
        )
        assert results == []


@pytest.mark.asyncio
class TestCompareManyToManyAsync:
    """Testes para compare_many_to_many_async."""

    async def test_basic_results(self, smart_comp):
        queries = ["iPhone 13", "samsung s22"]
        candidates = [
            "celular iphone 13 novo",
            "samsung galaxy s22 ultra",
            "mesa de escritório",
        ]

        results = await smart_comp.compare_many_to_many_async(
            queries, candidates, top_n=5, min_cosine=0.0, n_workers=1,
        )

        assert len(results) == 2
        assert all(isinstance(r, list) for r in results)

    async def test_matches_sync(self, basic_comp):
        queries = ["mesa de escritório", "cadeira giratória"]
        candidates = [
            "mesa de escritório grande",
            "cadeira giratória preta",
            "estante de livros",
        ]

        sync_results = basic_comp.compare_many_to_many(
            queries, candidates, top_n=5, min_cosine=0.0,
            strategy="vectorized",
        )
        async_results = await basic_comp.compare_many_to_many_async(
            queries, candidates, top_n=5, min_cosine=0.0, n_workers=1,
        )

        assert len(sync_results) == len(async_results)
        for s_list, a_list in zip(sync_results, async_results):
            assert len(s_list) == len(a_list)
            for s, a in zip(s_list, a_list):
                assert s["candidate"] == a["candidate"]
                assert abs(s["score"] - a["score"]) < 1e-6

    async def test_empty_queries(self, smart_comp):
        results = await smart_comp.compare_many_to_many_async(
            queries=[], candidates=["qualquer"], n_workers=1,
        )
        assert results == []

    async def test_empty_candidates(self, smart_comp):
        results = await smart_comp.compare_many_to_many_async(
            queries=["q1", "q2"], candidates=[], n_workers=1,
        )
        assert len(results) == 2
        assert results[0] == []
        assert results[1] == []

    async def test_entity_short_circuit(self, smart_comp):
        queries = ["samsung s22 ultra"]
        candidates = [
            "Vende-se samsung galaxy s22 ultra na caixa",
            "samsung s21 barato",
        ]

        results = await smart_comp.compare_many_to_many_async(
            queries, candidates, top_n=5, min_cosine=0.0, n_workers=1,
        )

        top_hit = results[0][0]
        assert "s22 ultra" in top_hit["candidate"].lower()
        if "entity" in top_hit["details"]:
            assert top_hit["details"]["entity"]["score"] == 1.0
            assert top_hit["score"] == 0.95
