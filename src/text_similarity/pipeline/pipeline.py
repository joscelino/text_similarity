"""Orquestrador principal do processamento de NLP em pipeline."""

from __future__ import annotations

import logging
from typing import List

from .stage import PipelineContext, PipelineStage

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """Orquestrador de pré-processamento.

    Encadeia componentes (EntityNormalizer -> TextCleaner ->
    Tokenizer -> Stopwords -> Lemmatizer) passando um `PipelineContext`
    entre cada estágio.
    """

    def __init__(self, stages: List[PipelineStage] | None = None) -> None:
        """Inicializa a pipeline opcionalmente com estágios pré-carregados."""
        self.stages = stages or []

    def add_stage(self, stage: PipelineStage) -> None:
        """Anexa um novo estágio ao fim da fila de execução da pipeline."""
        self.stages.append(stage)

    def process(self, text: str) -> tuple[str, PipelineContext]:
        """Executa os estágios em sequência sobre o texto fornecido.

        Cria um `PipelineContext` inicial e o passa por cada estágio,
        acumulando transformações e metadados ao longo do caminho.

        Args:
            text: Texto bruto de entrada.

        Returns:
            Tuple: (texto_processado, contexto_completo_da_execução)
        """
        ctx = PipelineContext(text=text)

        for stage in self.stages:
            try:
                ctx = stage.process(ctx)
            except Exception as e:
                logger.error(f"Erro no estágio {stage.__class__.__name__}: {e}")

        return ctx.text, ctx
