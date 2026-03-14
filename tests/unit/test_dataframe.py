"""Testes unitários para compare_dataframe e record_linkage."""

from __future__ import annotations

import pytest

from text_similarity.api import Comparator


@pytest.fixture()
def comparator() -> Comparator:
    """Retorna um Comparator básico para os testes."""
    return Comparator.basic()


# ---------------------------------------------------------------------------
# DataFrame-like mínimo para testes sem dependência de pandas
# ---------------------------------------------------------------------------


class _SimpleDF:
    """DataFrame-like mínimo que suporta subscript por coluna."""

    def __init__(self, data: dict[str, list]) -> None:
        self._data = data
        self.columns = list(data.keys())

    def __getitem__(self, col: str) -> "_SimpleColumn":
        return _SimpleColumn(self._data[col])

    def __len__(self) -> int:
        return len(next(iter(self._data.values()))) if self._data else 0


class _SimpleColumn:
    """Coluna com interface .tolist() (estilo pandas)."""

    def __init__(self, values: list) -> None:
        self._values = values

    def tolist(self) -> list:
        return list(self._values)

    def __iter__(self):
        return iter(self._values)


class _PolarsLikeColumn:
    """Coluna com interface .to_list() (estilo polars)."""

    def __init__(self, values: list) -> None:
        self._values = values

    def to_list(self) -> list:
        return list(self._values)

    def __iter__(self):
        return iter(self._values)


class _PolarsLikeDF:
    """DataFrame-like com interface polars."""

    def __init__(self, data: dict[str, list]) -> None:
        self._data = data
        self.columns = list(data.keys())

    def __getitem__(self, col: str) -> "_PolarsLikeColumn":
        return _PolarsLikeColumn(self._data[col])

    def __len__(self) -> int:
        return len(next(iter(self._data.values()))) if self._data else 0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_df() -> _SimpleDF:
    """DataFrame-like de exemplo com produtos PT-BR (interface .tolist)."""
    return _SimpleDF(
        {
            "id": [1, 2, 3, 4, 5],
            "produto": [
                "televisão samsung 55 polegadas",
                "geladeira electrolux frost free",
                "notebook dell inspiron i5",
                "celular iphone 15 pro max",
                "fogão brastemp 4 bocas",
            ],
            "preco": [2500.0, 3200.0, 4500.0, 8000.0, 1500.0],
        }
    )


@pytest.fixture()
def sample_polars_df() -> _PolarsLikeDF:
    """DataFrame-like com interface .to_list (estilo polars)."""
    return _PolarsLikeDF(
        {
            "id": [1, 2, 3],
            "produto": [
                "televisão samsung 55 polegadas",
                "notebook dell inspiron i5",
                "fogão brastemp 4 bocas",
            ],
        }
    )


# ---------------------------------------------------------------------------
# _extract_column
# ---------------------------------------------------------------------------


class TestExtractColumn:
    """Testes para o helper duck-typing _extract_column."""

    def test_should_extract_tolist_interface(self, comparator: Comparator) -> None:
        df = _SimpleDF({"col": ["a", "b", "c"]})
        assert Comparator._extract_column(df, "col") == ["a", "b", "c"]

    def test_should_extract_to_list_interface(self, comparator: Comparator) -> None:
        df = _PolarsLikeDF({"col": ["x", "y"]})
        assert Comparator._extract_column(df, "col") == ["x", "y"]

    def test_should_fallback_to_generic_iteration(self) -> None:
        class _GenericDF:
            def __getitem__(self, col: str) -> list:
                return ["p", "q"]

        assert Comparator._extract_column(_GenericDF(), "any") == ["p", "q"]


# ---------------------------------------------------------------------------
# compare_dataframe
# ---------------------------------------------------------------------------


class TestCompareDataframe:
    """Testes para o método compare_dataframe."""

    def test_should_return_list(
        self,
        comparator: Comparator,
        sample_df: _SimpleDF,
    ) -> None:
        result = comparator.compare_dataframe(sample_df, "produto", "televisão samsung")
        assert isinstance(result, list)

    def test_should_return_list_of_dicts(
        self,
        comparator: Comparator,
        sample_df: _SimpleDF,
    ) -> None:
        result = comparator.compare_dataframe(sample_df, "produto", "televisão samsung")
        if result:
            assert isinstance(result[0], dict)

    def test_should_have_score_key(
        self,
        comparator: Comparator,
        sample_df: _SimpleDF,
    ) -> None:
        result = comparator.compare_dataframe(sample_df, "produto", "televisão samsung")
        if result:
            assert "score" in result[0]

    def test_should_have_scores_in_range(
        self,
        comparator: Comparator,
        sample_df: _SimpleDF,
    ) -> None:
        result = comparator.compare_dataframe(sample_df, "produto", "televisão samsung")
        for row in result:
            assert 0.0 <= row["score"] <= 1.0

    def test_should_respect_top_n(
        self,
        comparator: Comparator,
        sample_df: _SimpleDF,
    ) -> None:
        result = comparator.compare_dataframe(
            sample_df, "produto", "televisão samsung", top_n=2
        )
        assert len(result) <= 2

    def test_should_preserve_original_keys(
        self,
        comparator: Comparator,
        sample_df: _SimpleDF,
    ) -> None:
        result = comparator.compare_dataframe(sample_df, "produto", "televisão samsung")
        if result:
            for col in sample_df.columns:
                assert col in result[0]

    def test_should_be_sorted_descending(
        self,
        comparator: Comparator,
        sample_df: _SimpleDF,
    ) -> None:
        result = comparator.compare_dataframe(sample_df, "produto", "televisão samsung")
        scores = [r["score"] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_should_work_with_polars_like_df(
        self,
        comparator: Comparator,
        sample_polars_df: _PolarsLikeDF,
    ) -> None:
        result = comparator.compare_dataframe(
            sample_polars_df, "produto", "televisão samsung"
        )
        assert isinstance(result, list)

    def test_should_return_empty_list_when_no_match(
        self,
        comparator: Comparator,
        sample_df: _SimpleDF,
    ) -> None:
        result = comparator.compare_dataframe(
            sample_df, "produto", "xyzzy nada aqui", min_cosine=0.99
        )
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# record_linkage
# ---------------------------------------------------------------------------


class TestRecordLinkage:
    """Testes para o método record_linkage."""

    @pytest.fixture()
    def df_a(self) -> _SimpleDF:
        """DataFrame-like de queries."""
        return _SimpleDF({"texto": ["televisão samsung", "notebook dell"]})

    @pytest.fixture()
    def df_b(self) -> _SimpleDF:
        """DataFrame-like de candidatos."""
        return _SimpleDF(
            {
                "descricao": [
                    "tv samsung 55 polegadas",
                    "geladeira electrolux",
                    "laptop dell inspiron i5",
                    "celular iphone 15",
                ]
            }
        )

    def test_should_return_list(
        self,
        comparator: Comparator,
        df_a: _SimpleDF,
        df_b: _SimpleDF,
    ) -> None:
        result = comparator.record_linkage(df_a, df_b, "texto", "descricao")
        assert isinstance(result, list)

    def test_should_return_list_of_dicts(
        self,
        comparator: Comparator,
        df_a: _SimpleDF,
        df_b: _SimpleDF,
    ) -> None:
        result = comparator.record_linkage(df_a, df_b, "texto", "descricao")
        if result:
            assert isinstance(result[0], dict)

    def test_should_have_required_keys(
        self,
        comparator: Comparator,
        df_a: _SimpleDF,
        df_b: _SimpleDF,
    ) -> None:
        result = comparator.record_linkage(df_a, df_b, "texto", "descricao")
        expected = {"index_a", "text_a", "index_b", "text_b", "score", "details"}
        if result:
            assert expected.issubset(set(result[0].keys()))

    def test_should_have_scores_in_range(
        self,
        comparator: Comparator,
        df_a: _SimpleDF,
        df_b: _SimpleDF,
    ) -> None:
        result = comparator.record_linkage(df_a, df_b, "texto", "descricao")
        for row in result:
            assert 0.0 <= row["score"] <= 1.0

    def test_should_respect_top_n_per_query(
        self,
        comparator: Comparator,
        df_a: _SimpleDF,
        df_b: _SimpleDF,
    ) -> None:
        top_n = 2
        result = comparator.record_linkage(
            df_a, df_b, "texto", "descricao", top_n=top_n
        )
        from collections import Counter

        counts = Counter(r["index_a"] for r in result)
        for count in counts.values():
            assert count <= top_n

    def test_should_be_sorted_descending(
        self,
        comparator: Comparator,
        df_a: _SimpleDF,
        df_b: _SimpleDF,
    ) -> None:
        result = comparator.record_linkage(df_a, df_b, "texto", "descricao")
        scores = [r["score"] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_should_work_with_polars_like_dfs(
        self,
        comparator: Comparator,
    ) -> None:
        pf_a = _PolarsLikeDF({"q": ["televisão samsung"]})
        pf_b = _PolarsLikeDF({"c": ["tv samsung 55", "geladeira electrolux"]})
        result = comparator.record_linkage(pf_a, pf_b, "q", "c")
        assert isinstance(result, list)
