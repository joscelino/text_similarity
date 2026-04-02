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
            query,
            candidates,
            top_n=3,
            min_cosine=0.0,
            n_workers=1,
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
            query,
            candidates,
            top_n=5,
            min_cosine=0.0,
            strategy="vectorized",
        )
        async_results = await smart_comp.compare_batch_async(
            query,
            candidates,
            top_n=5,
            min_cosine=0.0,
            n_workers=1,
        )

        assert len(sync_results) == len(async_results)
        for s, a in zip(sync_results, async_results):
            assert s["candidate"] == a["candidate"]
            assert abs(s["score"] - a["score"]) < 1e-6

    async def test_empty_candidates(self, smart_comp):
        results = await smart_comp.compare_batch_async(
            "qualquer",
            [],
            n_workers=1,
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
            queries,
            candidates,
            top_n=5,
            min_cosine=0.0,
            n_workers=1,
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
            queries,
            candidates,
            top_n=5,
            min_cosine=0.0,
            strategy="vectorized",
        )
        async_results = await basic_comp.compare_many_to_many_async(
            queries,
            candidates,
            top_n=5,
            min_cosine=0.0,
            n_workers=1,
        )

        assert len(sync_results) == len(async_results)
        for s_list, a_list in zip(sync_results, async_results):
            assert len(s_list) == len(a_list)
            for s, a in zip(s_list, a_list):
                assert s["candidate"] == a["candidate"]
                assert abs(s["score"] - a["score"]) < 1e-6

    async def test_empty_queries(self, smart_comp):
        results = await smart_comp.compare_many_to_many_async(
            queries=[],
            candidates=["qualquer"],
            n_workers=1,
        )
        assert results == []

    async def test_empty_candidates(self, smart_comp):
        results = await smart_comp.compare_many_to_many_async(
            queries=["q1", "q2"],
            candidates=[],
            n_workers=1,
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
            queries,
            candidates,
            top_n=5,
            min_cosine=0.0,
            n_workers=1,
        )

        top_hit = results[0][0]
        assert "s22 ultra" in top_hit["candidate"].lower()
        if "entity" in top_hit["details"]:
            assert top_hit["details"]["entity"]["score"] == 1.0
            assert top_hit["score"] == 0.95


@pytest.mark.asyncio
class TestBM25Async:
    """Testes para métodos async com indexing_strategy='bm25'."""

    @pytest.fixture
    def bm25_comp(self):
        return Comparator.smart(entities=["product_model"], indexing_strategy="bm25")

    async def test_compare_batch_async_with_bm25(self, bm25_comp):
        """compare_batch_async retorna resultados válidos com BM25."""
        query = "notebook dell inspiron"
        candidates = [
            "notebook dell inspiron 15",
            "mouse logitech wireless",
            "teclado microsoft ergonômico",
        ]

        results = await bm25_comp.compare_batch_async(
            query, candidates, top_n=3, min_cosine=0.0, n_workers=1
        )

        assert len(results) > 0
        assert all("candidate" in r and "score" in r for r in results)
        assert "inspiron" in results[0]["candidate"].lower()

    async def test_compare_many_to_many_async_with_bm25(self, bm25_comp):
        """compare_many_to_many_async retorna lista de listas válidas com BM25."""
        queries = ["iPhone 13", "Galaxy S22"]
        candidates = [
            "celular iphone 13 novo",
            "samsung galaxy s22 ultra",
            "mesa de escritório",
        ]

        results = await bm25_comp.compare_many_to_many_async(
            queries, candidates, top_n=3, min_cosine=0.0, n_workers=1
        )

        assert len(results) == 2
        assert all(isinstance(r, list) for r in results)
        assert len(results[0]) > 0
        assert len(results[1]) > 0

    async def test_async_bm25_matches_sync_bm25(self, bm25_comp):
        """Resultado assíncrono com BM25 é idêntico ao síncrono."""
        query = "arroz integral tipo 1"
        candidates = [
            "arroz integral tipo 1 pacote 5kg",
            "feijão preto carioca",
            "arroz parboilizado",
        ]

        sync_results = bm25_comp.compare_batch(
            query, candidates, top_n=3, min_cosine=0.0, strategy="vectorized"
        )
        async_results = await bm25_comp.compare_batch_async(
            query, candidates, top_n=3, min_cosine=0.0, n_workers=1
        )

        assert len(sync_results) == len(async_results)
        for s, a in zip(sync_results, async_results):
            assert s["candidate"] == a["candidate"]
            assert abs(s["score"] - a["score"]) < 1e-6
