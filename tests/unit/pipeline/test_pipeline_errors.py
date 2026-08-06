"""Testes de conversão de exceções em PreprocessingPipeline.process."""

from __future__ import annotations

import pytest

from text_similarity.exceptions import StageProcessingError
from text_similarity.pipeline.pipeline import PreprocessingPipeline
from text_similarity.pipeline.stage import PipelineContext, PipelineStage


class _KeyErrorStage(PipelineStage):
    """Estágio de teste que levanta KeyError."""

    def process(self, ctx: PipelineContext) -> PipelineContext:
        raise KeyError("missing_key")


class _AttributeErrorStage(PipelineStage):
    """Estágio de teste que levanta AttributeError."""

    def process(self, ctx: PipelineContext) -> PipelineContext:
        raise AttributeError("missing_attr")


class _LookupErrorStage(PipelineStage):
    """Estágio de teste que levanta LookupError."""

    def process(self, ctx: PipelineContext) -> PipelineContext:
        raise LookupError("missing_resource")


class _RuntimeErrorStage(PipelineStage):
    """Estágio de teste que levanta RuntimeError."""

    def process(self, ctx: PipelineContext) -> PipelineContext:
        raise RuntimeError("backend_failure")


class _ValueErrorStage(PipelineStage):
    """Estágio de teste que levanta ValueError."""

    def process(self, ctx: PipelineContext) -> PipelineContext:
        raise ValueError("bad_value")


class _KeyboardInterruptStage(PipelineStage):
    """Estágio de teste que levanta KeyboardInterrupt."""

    def process(self, ctx: PipelineContext) -> PipelineContext:
        raise KeyboardInterrupt


class _SystemExitStage(PipelineStage):
    """Estágio de teste que levanta SystemExit."""

    def process(self, ctx: PipelineContext) -> PipelineContext:
        raise SystemExit(1)


class _MemoryErrorStage(PipelineStage):
    """Estágio de teste que levanta MemoryError."""

    def process(self, ctx: PipelineContext) -> PipelineContext:
        raise MemoryError


@pytest.mark.parametrize(
    ("stage", "expected_original"),
    [
        (_KeyErrorStage(), KeyError),
        (_AttributeErrorStage(), AttributeError),
        (_LookupErrorStage(), LookupError),
        (_RuntimeErrorStage(), RuntimeError),
        (_ValueErrorStage(), ValueError),
    ],
)
def test_recoverable_exceptions_become_stage_processing_error(
    stage: PipelineStage, expected_original: type[BaseException]
) -> None:
    """Exceções recuperáveis devem ser convertidas em StageProcessingError."""
    pipeline = PreprocessingPipeline([stage])

    with pytest.raises(StageProcessingError) as exc_info:
        pipeline.process("texto de entrada")

    assert exc_info.value.stage_name == stage.__class__.__name__
    assert isinstance(exc_info.value.original_error, expected_original)
    assert exc_info.value.__cause__ is exc_info.value.original_error


@pytest.mark.parametrize(
    "stage",
    [
        _KeyboardInterruptStage(),
        _SystemExitStage(),
        _MemoryErrorStage(),
    ],
)
def test_critical_exceptions_propagate_bare(stage: PipelineStage) -> None:
    """KeyboardInterrupt, SystemExit e MemoryError não devem ser capturadas."""
    pipeline = PreprocessingPipeline([stage])

    with pytest.raises(BaseException) as exc_info:
        pipeline.process("texto de entrada")

    assert isinstance(
        exc_info.value,
        (KeyboardInterrupt, SystemExit, MemoryError),
    )
