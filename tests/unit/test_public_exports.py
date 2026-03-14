"""Testa consistência das exportações públicas do pacote."""

import text_similarity


def test_should_export_comparator() -> None:
    assert hasattr(text_similarity, "Comparator")


def test_should_export_bm25_index() -> None:
    assert hasattr(text_similarity, "BM25Index")


def test_should_export_dense_index() -> None:
    assert hasattr(text_similarity, "DenseIndex")


def test_should_export_rr_fusion() -> None:
    assert hasattr(text_similarity, "RRFusion")


def test_should_export_exceptions() -> None:
    assert hasattr(text_similarity, "TextSimilarityError")
    assert hasattr(text_similarity, "PipelineError")
    assert hasattr(text_similarity, "StageProcessingError")
    assert hasattr(text_similarity, "StageConfigError")


def test_should_expose_version() -> None:
    assert hasattr(text_similarity, "__version__")
    assert text_similarity.__version__ == "0.7.0"


def test_all_exports_are_importable() -> None:
    """Verifica que todos os símbolos em __all__ são importáveis do pacote."""
    for name in text_similarity.__all__:
        assert hasattr(text_similarity, name), (
            f"'{name}' está em __all__ mas não é importável do pacote"
        )


def test_core_classes_are_correct_types() -> None:
    from text_similarity import BM25Index, Comparator, DenseIndex, RRFusion

    assert isinstance(Comparator.basic(), Comparator)
    assert isinstance(BM25Index(), BM25Index)
    assert isinstance(DenseIndex(), DenseIndex)
    assert isinstance(RRFusion(), RRFusion)
