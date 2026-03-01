from __future__ import annotations

from .base import EntityMatch
from .registry import ExtractorRegistry


class EntityInspector:
    """
    Inspeciona o texto e retorna as entidades encontradas (para debug/testes)
    sem aplicar substituições restritivas.
    """

    def __init__(self, entities: list[str] = None):
        """
        Args:
            entities: Lista de tipos de entidades a procurar.
                      Se None, utiliza todos os disponíveis no registro.
        """
        self.entities = entities or ExtractorRegistry.available_extractors()
        self.extractors = [
            ExtractorRegistry.get_extractor(name) for name in self.entities
        ]

    def inspect(self, text: str) -> list[EntityMatch]:
        """
        Varre o texto executando os extratores configurados e agrupa os resultados.
        """
        all_matches = []
        for ext in self.extractors:
            all_matches.extend(ext.extract(text))

        # Ordena os matches pelo início de sua aparição no texto
        all_matches.sort(key=lambda m: m.start)
        return all_matches
