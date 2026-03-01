"""Classes de modelagem fundamentais para a extração arquitetada."""

from __future__ import annotations

import abc
from dataclasses import dataclass


@dataclass
class EntityMatch:
    """Representação de uma entidade localizada dentro do texto."""

    entity_type: str
    value: str | float | int
    start: int
    end: int
    text_matched: str


class EntityExtractor(abc.ABC):
    """Interface base para todos os extratores de entidades (moeda, data, etc.)."""

    @abc.abstractmethod
    def extract(self, text: str) -> list[EntityMatch]:
        """Analisa a string de entrada e retorna as entidades encontradas.

        Args:
            text (str): Texto de entrada a ser analisado em português.

        Returns:
            list[EntityMatch]: Lista de entidades identificadas.
        """
        pass
