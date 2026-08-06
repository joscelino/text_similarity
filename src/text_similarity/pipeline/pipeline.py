"""Orquestrador principal do processamento de NLP em pipeline."""

from __future__ import annotations

import logging
from typing import List

from text_similarity.exceptions import StageProcessingError

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

        Quando um estágio levanta uma exceção recuperável, ela é convertida
        para `StageProcessingError`, preservando a causa original via
        ``raise ... from e``. As seguintes exceções são tratadas:

        - ``TypeError``, ``ValueError``: dados ou input inválido.
        - ``KeyError``, ``AttributeError``, ``LookupError``: falhas de
          acesso a dados, atributos ou recursos (dicionários, modelos,
          datasets NLP ausentes).
        - ``UnicodeDecodeError``, ``UnicodeEncodeError``: erros de encoding.
        - ``OSError``: erros de I/O ou modelo não encontrado.
        - ``RuntimeError``: runtime de backend NLP (ex: Regex, SpaCy).

        As exceções abaixo NÃO são capturadas e propagam "bare", pois
        representam condições críticas do ambiente de execução:

        - ``KeyboardInterrupt``: interrupção explícita pelo usuário.
        - ``SystemExit``: finalização solicitada pelo sistema.
        - ``MemoryError``: esgotamento de memória; capturá-la poderia
          mascarar falhas graves do ambiente.

        Args:
            text: Texto bruto de entrada.

        Returns:
            Tuple: (texto_processado, contexto_completo_da_execução)
        """
        ctx = PipelineContext(text=text)

        for stage in self.stages:
            try:
                ctx = stage.process(ctx)
            except StageProcessingError:
                # Já é exceção de pipeline, propague
                raise
            except (TypeError, ValueError) as e:
                # Erro de dados/input inválido
                raise StageProcessingError(stage.__class__.__name__, e) from e
            except (KeyError, AttributeError, LookupError) as e:
                # Falha de acesso a dados/atributos/recursos
                raise StageProcessingError(stage.__class__.__name__, e) from e
            except (UnicodeDecodeError, UnicodeEncodeError) as e:
                # Erro de encoding
                raise StageProcessingError(stage.__class__.__name__, e) from e
            except OSError as e:
                # Erro de I/O ou modelo não encontrado
                raise StageProcessingError(stage.__class__.__name__, e) from e
            except RuntimeError as e:
                # Erro de runtime de backend NLP (ex: Regex, SpaCy)
                raise StageProcessingError(stage.__class__.__name__, e) from e

        return ctx.text, ctx
