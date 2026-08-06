"""Testes para SEC-LOGIC-004: modo strict em ``SemanticSimilarity``.

Verifica:
1. ``SemanticSimilarityError`` existe e é subclasse de ``Exception``.
2. ``__init__`` aceita ``strict: bool = True`` (default estrito).
3. Em ``strict=True`` (padrão), ``RuntimeError`` (inclui
   ``torch.cuda.OutOfMemoryError``) do backend é re-lançado como
   ``SemanticSimilarityError``.
4. Exceções inesperadas (fora do conjunto conhecido) propagam
   livremente — sem ``except Exception`` engolindo bugs.
5. Em ``strict=False`` o comportamento legado tolerante é mantido:
   retorna ``0.0`` mas registra ``logger.error`` com ``exc_info=True``.
6. ``Comparator`` propaga ``strict`` para o ``SemanticSimilarity``
   instanciado dentro do ``HybridSimilarity``.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from text_similarity.core.semantic import (
    SemanticSimilarity,
    SemanticSimilarityError,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def mocked_backend(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Substitui os globais do módulo semantic por mocks controláveis.

    Devolve o mock do modelo — testes ajustam ``model.encode.side_effect``
    para simular diferentes falhas do backend PyTorch.
    """
    import text_similarity.core.semantic as sem_module

    fake_model = MagicMock(name="fake_sentence_transformer")
    fake_util = MagicMock(name="fake_sentence_util")

    monkeypatch.setattr(sem_module, "_GLOBAL_MODEL", fake_model, raising=False)
    monkeypatch.setattr(
        sem_module,
        "_CURRENT_MODEL_KEY",
        ("paraphrase-multilingual-MiniLM-L12-v2", None, None),
        raising=False,
    )
    monkeypatch.setattr(sem_module, "_SENTENCE_UTIL", fake_util, raising=False)
    return fake_model


# ---------------------------------------------------------------------------
# 1. Classe de exceção
# ---------------------------------------------------------------------------


def test_semantic_similarity_error_is_exception_subclass() -> None:
    """SemanticSimilarityError é uma exceção customizada válida."""
    assert issubclass(SemanticSimilarityError, Exception)
    # Instanciável com mensagem
    err = SemanticSimilarityError("boom")
    assert str(err) == "boom"


# ---------------------------------------------------------------------------
# 2. Assinatura do __init__
# ---------------------------------------------------------------------------


def test_init_accepts_strict_parameter_default_true() -> None:
    """__init__ aceita strict e default é True (produção-safe)."""
    sem = SemanticSimilarity()
    assert sem.strict is True

    sem_tolerant = SemanticSimilarity(strict=False)
    assert sem_tolerant.strict is False


# ---------------------------------------------------------------------------
# 3. strict=True: RuntimeError vira SemanticSimilarityError
# ---------------------------------------------------------------------------


def test_strict_reraises_runtime_error_as_semantic_error(
    mocked_backend: MagicMock,
) -> None:
    """RuntimeError do model.encode → SemanticSimilarityError."""
    mocked_backend.encode.side_effect = RuntimeError("simulated backend error")

    sem = SemanticSimilarity(strict=True)

    with pytest.raises(SemanticSimilarityError) as exc_info:
        sem.compare("texto A", "texto B")

    # Traceback original preservado via raise ... from
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert "simulated backend error" in str(exc_info.value.__cause__)


def test_strict_reraises_cuda_oom_as_semantic_error(
    mocked_backend: MagicMock,
) -> None:
    """torch.cuda.OutOfMemoryError herda de RuntimeError → convertida também."""

    class _FakeCudaOOM(RuntimeError):
        """Simula torch.cuda.OutOfMemoryError sem exigir torch instalado."""

    mocked_backend.encode.side_effect = _FakeCudaOOM("CUDA out of memory")

    sem = SemanticSimilarity(strict=True)

    with pytest.raises(SemanticSimilarityError) as exc_info:
        sem.compare("texto A", "texto B")

    assert isinstance(exc_info.value.__cause__, _FakeCudaOOM)


# ---------------------------------------------------------------------------
# 4. strict=True: exceções inesperadas propagam SEM engolir
# ---------------------------------------------------------------------------


def test_strict_lets_unexpected_exceptions_propagate(
    mocked_backend: MagicMock,
) -> None:
    """ValueError (não é RuntimeError) NÃO deve virar SemanticSimilarityError."""
    mocked_backend.encode.side_effect = ValueError("input inesperado")

    sem = SemanticSimilarity(strict=True)

    # Deve propagar como ValueError puro — evidência de que o antigo
    # ``except Exception`` foi removido do caminho estrito.
    with pytest.raises(ValueError, match="input inesperado"):
        sem.compare("texto A", "texto B")


def test_strict_lets_type_error_propagate(mocked_backend: MagicMock) -> None:
    """TypeError também propaga — regressão-guard contra except Exception."""
    mocked_backend.encode.side_effect = TypeError("assinatura errada")

    sem = SemanticSimilarity(strict=True)

    with pytest.raises(TypeError, match="assinatura errada"):
        sem.compare("texto A", "texto B")


# ---------------------------------------------------------------------------
# 5. strict=False: fallback silencioso + stacktrace no logger
# ---------------------------------------------------------------------------


def test_non_strict_returns_zero_on_runtime_error(
    mocked_backend: MagicMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Modo tolerante: RuntimeError vira 0.0 (não é re-lançado)."""
    mocked_backend.encode.side_effect = RuntimeError("kaboom")

    sem = SemanticSimilarity(strict=False)

    with caplog.at_level(logging.ERROR, logger="text_similarity.core.semantic"):
        score = sem.compare("texto A", "texto B")

    assert score == 0.0
    # Stacktrace deve ter sido registrado (exc_info=True)
    assert any(
        rec.exc_info is not None and rec.levelno == logging.ERROR
        for rec in caplog.records
    ), "modo strict=False deve logar exc_info=True para diagnóstico"


def test_non_strict_returns_zero_on_generic_exception(
    mocked_backend: MagicMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Modo tolerante engole QUALQUER exceção (comportamento legado)."""
    mocked_backend.encode.side_effect = ValueError("qualquer coisa")

    sem = SemanticSimilarity(strict=False)

    with caplog.at_level(logging.ERROR, logger="text_similarity.core.semantic"):
        score = sem.compare("a", "b")

    assert score == 0.0
    assert any(rec.exc_info is not None for rec in caplog.records)


# ---------------------------------------------------------------------------
# 6. Empty inputs: comportamento inalterado em ambos os modos
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strict", [True, False])
def test_empty_inputs_return_zero_regardless_of_strict(strict: bool) -> None:
    """Textos vazios curto-circuitam antes do backend em ambos os modos."""
    sem = SemanticSimilarity(strict=strict)
    assert sem.compare("", "algo") == 0.0
    assert sem.compare("algo", "") == 0.0
    assert sem.compare("", "") == 0.0


# ---------------------------------------------------------------------------
# 7. Propagação via Comparator
# ---------------------------------------------------------------------------


def test_comparator_propagates_strict_to_semantic_similarity() -> None:
    """Comparator(strict=X) → SemanticSimilarity dentro do Hybrid com strict=X."""
    from text_similarity.api import Comparator
    from text_similarity.core.hybrid import HybridSimilarity

    # Default: strict=True
    comp_strict = Comparator(mode="smart", use_embeddings=True)
    assert isinstance(comp_strict.algorithm, HybridSimilarity)
    semantic = comp_strict.algorithm.algorithms.get("semantic")
    assert semantic is not None
    assert getattr(semantic, "strict", None) is True

    # Explicit strict=False
    comp_tolerant = Comparator(mode="smart", use_embeddings=True, strict=False)
    assert isinstance(comp_tolerant.algorithm, HybridSimilarity)
    semantic_t = comp_tolerant.algorithm.algorithms.get("semantic")
    assert semantic_t is not None
    assert getattr(semantic_t, "strict", None) is False


def test_comparator_smart_propagates_strict() -> None:
    """Comparator.smart(strict=X) chega ao SemanticSimilarity."""
    from text_similarity.api import Comparator
    from text_similarity.core.hybrid import HybridSimilarity

    comp = Comparator.smart(use_embeddings=True, strict=False)
    assert isinstance(comp.algorithm, HybridSimilarity)
    semantic = comp.algorithm.algorithms.get("semantic")
    assert semantic is not None
    assert getattr(semantic, "strict", None) is False


def test_comparator_strict_end_to_end_propagates_runtime_error(
    mocked_backend: MagicMock,
) -> None:
    """Fluxo real: Comparator.compare (strict=True) re-lança SemanticSimilarityError."""
    from text_similarity.api import Comparator

    mocked_backend.encode.side_effect = RuntimeError("oom")

    comp = Comparator(mode="basic", use_embeddings=True, strict=True)

    # No modo basic com use_embeddings, HybridSimilarity chama .compare
    # em cada algoritmo; o semantic deve levantar
    # SemanticSimilarityError, não ser engolido.
    with pytest.raises(SemanticSimilarityError):
        comp.compare("texto exemplo um", "texto exemplo dois")
