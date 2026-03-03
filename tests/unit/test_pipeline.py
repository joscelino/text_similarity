from __future__ import annotations

import tempfile

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


def test_pipeline_catches_stage_exception() -> None:
    """Pipeline deve capturar exceção de estágio e continuar processando."""
    from unittest.mock import MagicMock

    from text_similarity.pipeline.stage import PipelineContext

    bad_stage = MagicMock()
    bad_stage.process.side_effect = RuntimeError("Estágio quebrado")

    pipeline = PreprocessingPipeline([bad_stage, CleanTextStage()])
    # Não deve levantar exceção — apenas logar o erro e continuar
    result, ctx = pipeline.process("Texto qualquer para processar")
    assert isinstance(result, str)
    assert isinstance(ctx, PipelineContext)
