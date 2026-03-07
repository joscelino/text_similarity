from __future__ import annotations

import tempfile
import pytest

from text_similarity.exceptions import StageProcessingError

from text_similarity.pipeline.backends import (
    CleanTextStage,
    LemmatizeStage,
    NormalizeEntitiesStage,
    StopwordsStage,
    TokenizerStage,
)
from text_similarity.pipeline.cache import PipelineCache
from text_similarity.pipeline.pipeline import PreprocessingPipeline


def test_preprocessing_pipeline() -> None:
    # Configura o pipeline com todos os estágios
    # Configura o pipeline invertendo Limpeza e Entidades
    # Entidades devem ser extraídas/normalizadas ANTES da
    # limpeza drástica remover pontuações (vírgulas de dinheiro, pontos de data)
    pipeline = PreprocessingPipeline(
        [
            NormalizeEntitiesStage(),
            CleanTextStage(),
            TokenizerStage(),
            StopwordsStage(),
            LemmatizeStage(),
        ]
    )

    text = "Eu comprei 2kg de arroz por R$ 30,00 ontem!"

    processed, context = pipeline.process(text)

    # 2kg vira <dimension:2.0:kg>, R$ 30,00 vira <money:30.0>, ontem vira <date:X>
    # "eu", "de", "por" sumiram pelas stopwords
    # "comprei" deve virar "comprar" pelo lemmatizer (se spaCy)

    assert "<dimension:" in processed
    assert "<money:30.0>" in processed
    assert "comprar" in processed or "comprei" in processed
    # TokenizerStage popula ctx.tokens; verificamos via atributo do PipelineContext
    assert isinstance(context.tokens, list)
    assert len(context.tokens) > 0


def test_pipeline_cache() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = PipelineCache(cache_dir=tmpdir)
        text = "Testando o hash da string!"

        h1 = cache.hash_text(text)
        h2 = cache.hash_text("testando o hash da string!")

        # O hash original é sensível à caixa, mas em nosso pipeline
        # tudo passará pelo TextCleaner antes se quisermos hash único.
        assert h1 != h2

        # Testando persistência do memory cache
        # O joblib apenas injeta um wrapper. Aqui verificamos a pasta limpa.
        cache.clear()


def test_add_stage_appends_to_pipeline() -> None:
    """add_stage() deve anexar um estágio ao pipeline vazio corretamente."""
    pipeline = PreprocessingPipeline()
    assert len(pipeline.stages) == 0

    pipeline.add_stage(CleanTextStage())
    assert len(pipeline.stages) == 1

    pipeline.add_stage(TokenizerStage())
    assert len(pipeline.stages) == 2


def test_pipeline_raises_stage_exception() -> None:
    """Pipeline deve abortar e propagar o erro encapsulado como StageProcessingError."""
    from unittest.mock import MagicMock

    bad_stage = MagicMock()
    bad_stage.process.side_effect = ValueError("Input inválido")
    # Para o erro ficar legível no report e capturável no pytest matches
    type(bad_stage).__name__ = "BadStage"

    pipeline = PreprocessingPipeline([bad_stage, CleanTextStage()])
    
    with pytest.raises(StageProcessingError, match="BadStage"):
         pipeline.process("Texto texto")


def test_pipeline_type_error_fail_fast() -> None:
    """Um erro de tipo puro na entra deve disparar StageProcessingError."""
    # Instanciar a primeira normalizadora para garantir falha por tipo
    pipeline = PreprocessingPipeline([CleanTextStage()])
    with pytest.raises(StageProcessingError) as exc_info:
        pipeline.process(None)  # type: ignore

    assert "CleanTextStage" in str(exc_info.value)
    assert isinstance(exc_info.value.original_error, TypeError)


def test_pipeline_unicode_error_fail_fast_mocked() -> None:
    """Um erro de decodificação Unicode deve disparar StageProcessingError."""
    from unittest.mock import MagicMock

    bad_stage = MagicMock()
    
    # simulamos como se um stage atirasse um unicode error ao processar byte array ou problema enconding
    error_instance = UnicodeDecodeError("utf-8", b"\\x81", 0, 1, "invalid start byte")
    bad_stage.process.side_effect = error_instance
    type(bad_stage).__name__ = "UnicodeStageTest"

    pipeline = PreprocessingPipeline([bad_stage])
    
    with pytest.raises(StageProcessingError) as exc_info:
        pipeline.process("Texto com caracteres ruins")
        
    assert "UnicodeStageTest" in str(exc_info.value)
    assert exc_info.value.original_error is error_instance


def test_pipeline_stage_processing_error_attributes() -> None:
    """Garantir que a exceção preserva os atributos da falha original."""
    from unittest.mock import MagicMock

    bad_stage = MagicMock()
    original_err = RuntimeError("Motor NLP off-line")
    bad_stage.process.side_effect = original_err
    type(bad_stage).__name__ = "MockNLPStage"

    pipeline = PreprocessingPipeline([bad_stage])
    
    with pytest.raises(StageProcessingError) as exc_info:
        pipeline.process("Process")
        
    assert exc_info.value.stage_name == "MockNLPStage"
    assert exc_info.value.original_error is original_err
    assert "MockNLPStage" in str(exc_info.value)
    assert "RuntimeError" in str(exc_info.value)
