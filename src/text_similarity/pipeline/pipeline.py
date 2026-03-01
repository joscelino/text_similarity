"""Orquestrador principal do processamento de NLP em pipeline."""

from __future__ import annotations

import logging
from typing import Any, List

from .stage import PipelineStage

logger = logging.getLogger(__name__)

class PreprocessingPipeline:
    """Orquestrador de pré-processamento.
    
    Encadeia componentes (TextCleaner -> EntityNormalizer ->
    Tokenizer -> Stopwords -> Lemmatizer).
    """

    def __init__(self, stages: List[PipelineStage] | None = None) -> None:
        """Inicializa a pipeline opcionalmente com estágios pré-carregados."""
        self.stages = stages or []

    def add_stage(self, stage: PipelineStage) -> None:
        """Anexa um novo estágio ao fim da fila de execução da pipeline."""
        self.stages.append(stage)

    def process(self, text: str) -> tuple[str, dict[str, Any]]:
        """Executa os estágios em sequência.

        Returns:
            Tuple: (texto_processado, dicionário_de_contexto_da_execucao)
        """
        context: dict[str, Any] = {"entities_found": []}
        current_text = text

        for stage in self.stages:
            try:
                current_text = stage.process(current_text, context)
            except Exception as e:
                logger.error(f"Erro no estágio {stage.__class__.__name__}: {e}")

        return current_text, context
