from __future__ import annotations

import abc
from typing import Any


class PipelineStage(abc.ABC):
    """
    Interface base para qualquer estágio do pipeline de similaridade.
    Cada estágio recebe o resultado do anterior e pode transformá-lo
    ou apenas adicionar metadados ao contexto compartilhado.
    """

    @abc.abstractmethod
    def process(self, text: str, context: dict[str, Any]) -> str:
        """
        Processa o texto ou extrai dados dele.

        Args:
            text: Texto de entrada para este estágio.
            context: Dicionário compartilhado contendo dados globais da execução
                     (ex: entidades encontradas, flags, métricas).

        Returns:
            str: O texto transformado ou original, dependendo do estágio.
        """
        pass
