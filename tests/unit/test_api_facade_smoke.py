"""Teste de fumaça da fachada :mod:`text_similarity.api` (SEC-STD-001).

Garante que, após o refactor que quebrou ``api.py`` em submódulos por
domínio (``comparator``, ``scoring``, ``dataframe_ops``,
``index_manager``, ``batch``), a superfície pública histórica continua
funcionando:

- ``from text_similarity import Comparator``
- ``from text_similarity.api import Comparator``
- ``Comparator.basic()`` cria instância sem exceção.
- ``Comparator.compare`` retorna um float em ``[0, 1]``.
- ``Comparator.compare_batch`` executa e devolve lista.

Se este teste falhar, é sinal de regressão da fachada.
"""

from __future__ import annotations


class TestFacadeImports:
    """Imports públicos preservados pelo refactor."""

    def test_top_level_import(self) -> None:
        from text_similarity import Comparator

        assert callable(Comparator)

    def test_api_module_import(self) -> None:
        from text_similarity.api import Comparator

        assert callable(Comparator)

    def test_both_imports_reference_same_class(self) -> None:
        from text_similarity import Comparator as A
        from text_similarity.api import Comparator as B

        assert A is B

    def test_facade_exports_only_comparator(self) -> None:
        import text_similarity.api as api_pkg

        assert "Comparator" in api_pkg.__all__


class TestFacadeSubmodulesExist:
    """Os submódulos exigidos pela SPEC estão presentes e importáveis."""

    def test_comparator_submodule(self) -> None:
        from text_similarity.api import comparator as m

        assert hasattr(m, "Comparator")

    def test_scoring_submodule(self) -> None:
        from text_similarity.api import scoring as m

        assert hasattr(m, "ScoringMixin")

    def test_dataframe_ops_submodule(self) -> None:
        from text_similarity.api import dataframe_ops as m

        assert hasattr(m, "DataFrameOpsMixin")

    def test_index_manager_submodule(self) -> None:
        from text_similarity.api import index_manager as m

        assert hasattr(m, "IndexManagerMixin")
        assert hasattr(m, "INDEX_BUILDERS")

    def test_batch_submodule(self) -> None:
        from text_similarity.api import batch as m

        assert hasattr(m, "BatchMixin")


class TestBasicUsageSmoke:
    """Fluxo mínimo de uso deve funcionar sem raise."""

    def test_basic_factory_creates_instance(self) -> None:
        from text_similarity import Comparator

        comp = Comparator.basic()
        assert comp is not None
        assert comp.mode == "basic"

    def test_basic_compare_returns_float(self) -> None:
        from text_similarity import Comparator

        comp = Comparator.basic()
        score = comp.compare("produto teste um", "produto teste dois")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_basic_compare_identical_texts(self) -> None:
        """Textos idênticos devem produzir score alto."""
        from text_similarity import Comparator

        comp = Comparator.basic()
        score = comp.compare("mesmo texto", "mesmo texto")
        assert score >= 0.9

    def test_basic_compare_batch_returns_list(self) -> None:
        from text_similarity import Comparator

        comp = Comparator.basic()
        results = comp.compare_batch(
            "notebook dell",
            ["notebook dell inspiron", "impressora hp", "notebook dell xps"],
            top_n=5,
        )
        assert isinstance(results, list)
        assert all("candidate" in r and "score" in r for r in results)


class TestMixinComposition:
    """A classe :class:`Comparator` herda os quatro mixins do refactor."""

    def test_comparator_mro_includes_mixins(self) -> None:
        from text_similarity import Comparator
        from text_similarity.api.batch import BatchMixin
        from text_similarity.api.dataframe_ops import DataFrameOpsMixin
        from text_similarity.api.index_manager import IndexManagerMixin
        from text_similarity.api.scoring import ScoringMixin

        mro = Comparator.__mro__
        assert BatchMixin in mro
        assert DataFrameOpsMixin in mro
        assert IndexManagerMixin in mro
        assert ScoringMixin in mro
