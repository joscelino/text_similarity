"""Módulo base que define contratos dos adaptadores do Pipeline."""

from __future__ import annotations

import abc
import dataclasses
from typing import Any


@dataclasses.dataclass
class PipelineContext:
    """Contexto compartilhado e mutável entre os estágios do pipeline.

    Substitui o dict genérico, centralizando os dados do processamento
    em atributos fortemente tipados.

    Attributes:
        text: Texto atual em transformação, modificado por cada estágio.
        tokens: Lista de tokens gerada pelo `TokenizerStage` e mantida
            atualizada pelos estágios seguintes.
        entities_found: Entidades detectadas pelo `NormalizeEntitiesStage`.
        metadata: Dados arbitrários opcionais para extensibilidade futura.
    """

    text: str
    tokens: list[str] = dataclasses.field(default_factory=list)
    entities_found: list[Any] = dataclasses.field(default_factory=list)
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


class PipelineStage(abc.ABC):
    """Interface base para qualquer estágio do pipeline de similaridade.

    Cada estágio recebe o contexto completo do processamento, podendo
    modificar tanto o texto quanto os metadados associados.
    """

    @abc.abstractmethod
    def process(self, ctx: PipelineContext) -> PipelineContext:
        """Processa e transforma o contexto do pipeline.

        Args:
            ctx: Contexto com o estado atual do processamento.

        Returns:
            PipelineContext: O mesmo contexto, modificado pelo estágio.
        """
        pass
