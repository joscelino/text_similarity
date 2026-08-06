"""Testes de reuso do índice ativo (SEC-DUP-003).

Cobre :meth:`Comparator._get_or_build_index` e o registry
:data:`INDEX_BUILDERS` — garantem que:

1. O índice é REUTILIZADO quando ``self._active_index`` já é do tipo
   correto para a estratégia solicitada (não faz ``fit`` de novo).
2. O índice é RECRIADO quando o tipo ativo não corresponde à
   estratégia solicitada (ex.: trocar BM25 → Dense).
3. O helper SEMPRE atualiza ``self._active_index`` (invariante).
4. O registry ``INDEX_BUILDERS`` contém as chaves ``"bm25"`` e ``"dense"``.
5. ``compare_many_to_many`` usa o helper — sem duplicação de
   ``if/elif`` para os dois backends.
"""

from __future__ import annotations

import inspect
from typing import Any, List
from unittest.mock import patch

import pytest

from text_similarity import Comparator
from text_similarity.api.index_manager import (
    INDEX_BUILDERS,
    IndexManagerMixin,
    _build_bm25,
    _build_dense,
)
from text_similarity.core._index_protocol import IndexProtocol
from text_similarity.core.bm25 import BM25Index


# ---------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------
class TestIndexBuildersRegistry:
    """Contrato de :data:`INDEX_BUILDERS`."""

    def test_registry_has_bm25_and_dense(self) -> None:
        assert "bm25" in INDEX_BUILDERS
        assert "dense" in INDEX_BUILDERS

    def test_registry_values_are_callables(self) -> None:
        for name, builder in INDEX_BUILDERS.items():
            assert callable(builder), f"Builder '{name}' não é callable"

    def test_bm25_builder_returns_bm25index(self) -> None:
        idx = _build_bm25(bm25_k1=1.2, bm25_b=0.75)
        assert isinstance(idx, BM25Index)
        assert idx.k1 == 1.2
        assert idx.b == 0.75

    def test_bm25_builder_ignores_unrelated_kwargs(self) -> None:
        """O builder aceita kwargs unificados sem quebrar."""
        idx = _build_bm25(
            bm25_k1=1.5,
            bm25_b=0.6,
            dense_model_name="fake",
            dense_precision="float32",
        )
        assert idx.k1 == 1.5
        assert idx.b == 0.6

    def test_dense_builder_signature(self) -> None:
        """O builder Dense aceita os kwargs esperados sem quebrar."""
        sig = inspect.signature(_build_dense)
        assert "dense_model_name" in sig.parameters
        assert "dense_precision" in sig.parameters


# ---------------------------------------------------------------------
# _get_or_build_index — reuso e criação
# ---------------------------------------------------------------------
class TestGetOrBuildIndexBM25:
    """Reuso e reconstrução do índice BM25 via helper."""

    def test_first_call_creates_and_fits(self) -> None:
        comp = Comparator(mode="basic", indexing_strategy="bm25")
        assert comp._active_index is None

        p_candidates: List[str] = ["produto a", "produto b", "produto c"]
        idx = comp._get_or_build_index("bm25", p_candidates)

        assert isinstance(idx, BM25Index)
        assert comp._active_index is idx  # invariante: sempre atualiza
        assert idx._corpus_size == 3  # foi ajustado

    def test_second_call_reuses_same_instance(self) -> None:
        """Sem trocar a estratégia, ``fit`` NÃO é chamado novamente."""
        comp = Comparator(mode="basic", indexing_strategy="bm25")
        p_candidates = ["a b c", "b c d", "c d e"]

        first = comp._get_or_build_index("bm25", p_candidates)

        # Segunda chamada — mesmo p_candidates
        with patch.object(BM25Index, "fit", autospec=True) as mock_fit:
            second = comp._get_or_build_index("bm25", p_candidates)
            mock_fit.assert_not_called()

        assert first is second
        assert comp._active_index is second

    def test_recreates_when_active_index_is_wrong_type(self) -> None:
        """Se o ``_active_index`` for de outro tipo, reconstrói do zero."""
        comp = Comparator(mode="basic", indexing_strategy="bm25")

        # Simular índice ativo "estranho"
        comp._active_index = object()

        p_candidates = ["x y z"]
        idx = comp._get_or_build_index("bm25", p_candidates)

        assert isinstance(idx, BM25Index)
        assert comp._active_index is idx

    def test_helper_always_updates_active_index(self) -> None:
        """Invariante SEC-DUP-003: sempre grava em self._active_index."""
        comp = Comparator(mode="basic", indexing_strategy="bm25")
        p_candidates = ["um dois", "tres quatro"]

        for _ in range(3):
            idx = comp._get_or_build_index("bm25", p_candidates)
            assert comp._active_index is idx

    def test_returns_index_protocol_conformant(self) -> None:
        """O objeto retornado satisfaz :class:`IndexProtocol`."""
        comp = Comparator(mode="basic", indexing_strategy="bm25")
        idx = comp._get_or_build_index("bm25", ["alfa beta", "gama delta"])

        # Duck typing via runtime_checkable Protocol
        assert isinstance(idx, IndexProtocol)
        # Métodos-chave existem e são chamáveis
        assert callable(getattr(idx, "fit", None))
        assert callable(getattr(idx, "get_scores_normalized", None))


# ---------------------------------------------------------------------
# Erros
# ---------------------------------------------------------------------
class TestGetOrBuildIndexErrors:
    """Chamadas inválidas propagam erro claro."""

    def test_unknown_strategy_raises_keyerror(self) -> None:
        comp = Comparator(mode="basic", indexing_strategy="bm25")
        with pytest.raises(KeyError, match="não registrada"):
            comp._get_or_build_index("desconhecida", ["a b"])


# ---------------------------------------------------------------------
# Contrato compare_many_to_many x helper
# ---------------------------------------------------------------------
class TestCompareManyToManyUsesHelper:
    """Contrato ``compare_many_to_many`` × helper.

    ``compare_many_to_many`` deve chamar ``_get_or_build_index`` — não
    duplicar a lógica para BM25/Dense.
    """

    def test_bm25_path_calls_helper(self) -> None:
        comp = Comparator(mode="basic", indexing_strategy="bm25")

        p_candidates = ["produto x", "produto y", "produto z"]
        candidates = list(p_candidates)  # simula catálogo já normalizado
        queries = ["produto x"]

        real_helper = comp._get_or_build_index
        called_with: dict[str, Any] = {}

        def _spy(strategy: str, p_cands: List[str]) -> Any:
            called_with["strategy"] = strategy
            called_with["len"] = len(p_cands)
            return real_helper(strategy, p_cands)

        with patch.object(comp, "_get_or_build_index", side_effect=_spy):
            results = comp.compare_many_to_many(
                queries=queries,
                candidates=candidates,
                preprocess=False,
                top_n=5,
            )

        assert called_with.get("strategy") == "bm25"
        assert called_with.get("len") == 3
        assert isinstance(results, list) and len(results) == 1

    def test_tfidf_path_does_not_call_helper(self) -> None:
        """TF-IDF ainda vive fora do registry (fallback nativo do sklearn)."""
        comp = Comparator(mode="basic", indexing_strategy="tfidf")
        with patch.object(comp, "_get_or_build_index") as mock_helper:
            comp.compare_many_to_many(
                queries=["a b c"],
                candidates=["a b", "b c", "c d"],
                preprocess=False,
            )
            mock_helper.assert_not_called()


# ---------------------------------------------------------------------
# API pública mantida
# ---------------------------------------------------------------------
class TestPublicSurfacePreserved:
    """``from text_similarity.api import Comparator`` continua funcionando."""

    def test_import_from_facade(self) -> None:
        from text_similarity.api import Comparator as C

        assert C is Comparator

    def test_helper_exists_on_mixin(self) -> None:
        assert hasattr(IndexManagerMixin, "_get_or_build_index")
        assert callable(IndexManagerMixin._get_or_build_index)

    def test_helper_signature(self) -> None:
        sig = inspect.signature(IndexManagerMixin._get_or_build_index)
        # (self, strategy, p_candidates)
        params = list(sig.parameters.keys())
        assert params[:3] == ["self", "strategy", "p_candidates"]
