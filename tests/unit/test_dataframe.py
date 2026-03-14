"""Testes unitários para compare_dataframe e record_linkage."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from text_similarity.api import Comparator


@pytest.fixture()
def comparator() -> Comparator:
    """Retorna um Comparator básico para os testes."""
    return Comparator.basic()


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """DataFrame de exemplo com produtos PT-BR."""
    return pd.DataFrame(
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


class TestCompareDataframe:
    """Testes para o método compare_dataframe."""

    def test_should_return_dataframe(
        self,
        comparator: Comparator,
        sample_df: pd.DataFrame,
    ) -> None:
        result = comparator.compare_dataframe(
            sample_df,
            "produto",
            "televisão samsung",
        )
        assert isinstance(result, pd.DataFrame)

    def test_should_have_score_column(
        self,
        comparator: Comparator,
        sample_df: pd.DataFrame,
    ) -> None:
        result = comparator.compare_dataframe(
            sample_df,
            "produto",
            "televisão samsung",
        )
        assert "score" in result.columns

    def test_should_have_scores_in_range(
        self,
        comparator: Comparator,
        sample_df: pd.DataFrame,
    ) -> None:
        result = comparator.compare_dataframe(
            sample_df,
            "produto",
            "televisão samsung",
        )
        if not result.empty:
            assert (result["score"] >= 0.0).all()
            assert (result["score"] <= 1.0).all()

    def test_should_respect_top_n(
        self,
        comparator: Comparator,
        sample_df: pd.DataFrame,
    ) -> None:
        result = comparator.compare_dataframe(
            sample_df,
            "produto",
            "televisão samsung",
            top_n=2,
        )
        assert len(result) <= 2

    def test_should_preserve_original_columns(
        self,
        comparator: Comparator,
        sample_df: pd.DataFrame,
    ) -> None:
        result = comparator.compare_dataframe(
            sample_df,
            "produto",
            "televisão samsung",
        )
        for col in sample_df.columns:
            assert col in result.columns

    def test_should_raise_without_pandas(
        self,
        comparator: Comparator,
        sample_df: pd.DataFrame,
    ) -> None:
        import builtins

        real_import = builtins.__import__

        def mock_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "pandas":
                raise ImportError("No module named 'pandas'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="pandas"):
                comparator.compare_dataframe(
                    sample_df,
                    "produto",
                    "televisão samsung",
                )


class TestRecordLinkage:
    """Testes para o método record_linkage."""

    @pytest.fixture()
    def df_a(self) -> pd.DataFrame:
        """DataFrame de queries."""
        return pd.DataFrame({"texto": ["televisão samsung", "notebook dell"]})

    @pytest.fixture()
    def df_b(self) -> pd.DataFrame:
        """DataFrame de candidatos."""
        return pd.DataFrame(
            {
                "descricao": [
                    "tv samsung 55 polegadas",
                    "geladeira electrolux",
                    "laptop dell inspiron i5",
                    "celular iphone 15",
                ]
            }
        )

    def test_should_return_dataframe(
        self,
        comparator: Comparator,
        df_a: pd.DataFrame,
        df_b: pd.DataFrame,
    ) -> None:
        result = comparator.record_linkage(df_a, df_b, "texto", "descricao")
        assert isinstance(result, pd.DataFrame)

    def test_should_have_required_columns(
        self,
        comparator: Comparator,
        df_a: pd.DataFrame,
        df_b: pd.DataFrame,
    ) -> None:
        result = comparator.record_linkage(df_a, df_b, "texto", "descricao")
        expected = {
            "index_a",
            "text_a",
            "index_b",
            "text_b",
            "score",
            "details",
        }
        assert expected.issubset(set(result.columns))

    def test_should_have_scores_in_range(
        self,
        comparator: Comparator,
        df_a: pd.DataFrame,
        df_b: pd.DataFrame,
    ) -> None:
        result = comparator.record_linkage(df_a, df_b, "texto", "descricao")
        if not result.empty:
            assert (result["score"] >= 0.0).all()
            assert (result["score"] <= 1.0).all()

    def test_should_respect_top_n_per_query(
        self,
        comparator: Comparator,
        df_a: pd.DataFrame,
        df_b: pd.DataFrame,
    ) -> None:
        top_n = 2
        result = comparator.record_linkage(
            df_a, df_b, "texto", "descricao", top_n=top_n
        )
        if not result.empty:
            for _, group in result.groupby("index_a"):
                assert len(group) <= top_n
