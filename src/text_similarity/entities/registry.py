from __future__ import annotations

from typing import Type

from .base import EntityExtractor


class ExtractorRegistry:
    """
    Registro centralizado de extratores de entidades.
    Permite acoplar e invocar extratores pelo nome (ex: 'money', 'date').
    """

    _extractors: dict[str, Type[EntityExtractor]] = {}

    @classmethod
    def register(cls, name: str, extractor_cls: Type[EntityExtractor]) -> None:
        """
        Registra um extrator customizado ou padrão.
        """
        cls._extractors[name] = extractor_cls

    @classmethod
    def get_extractor(cls, name: str) -> EntityExtractor:
        """
        Instancia e retorna um extrator pelo nome registrado.
        """
        if name not in cls._extractors:
            raise ValueError(f"Extrador '{name}' não registrado.")
        return cls._extractors[name]()

    @classmethod
    def available_extractors(cls) -> list[str]:
        return list(cls._extractors.keys())
