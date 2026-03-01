"""Módulo de gerência e catálogo de extratores habilitados de entidades."""

from __future__ import annotations

from typing import Type

from .base import EntityExtractor


class ExtractorRegistry:
    """Registro centralizado de extratores de entidades.
    
    Permite acoplar e invocar extratores pelo nome (ex: 'money', 'date').
    """

    _extractors: dict[str, Type[EntityExtractor]] = {}

    @classmethod
    def register(cls, name: str, extractor_cls: Type[EntityExtractor]) -> None:
        """Registra um extrator customizado ou padrão."""
        cls._extractors[name] = extractor_cls

    @classmethod
    def get_extractor(cls, name: str) -> EntityExtractor:
        """Instancia e retorna um extrator pelo nome registrado."""
        if name not in cls._extractors:
            raise ValueError(f"Extrador '{name}' não registrado.")
        return cls._extractors[name]()

    @classmethod
    def available_extractors(cls) -> list[str]:
        """Apresenta a lista estática de chaves permitidas instanciáveis no registro."""
        return list(cls._extractors.keys())

    @classmethod
    def load_defaults(cls) -> None:
        """Garante o carregamento dos extratores padrão distribuídos com a lib."""
        # A importação no final ou dentro da função evita problemas de
        # importação circular se os extratores tentarem usar o registro
        # logo na definição dependendo de como for estruturado
        from .extractors.date import DateExtractor
        from .extractors.dimension import DimensionExtractor
        from .extractors.money import MoneyExtractor
        from .extractors.number import NumberExtractor
        from .extractors.product_model import ProductModelExtractor

        cls.register("date", DateExtractor)
        cls.register("dimension", DimensionExtractor)
        cls.register("money", MoneyExtractor)
        cls.register("number", NumberExtractor)
        cls.register("product_model", ProductModelExtractor)


# Executado automaticamente ao carregar módulo
ExtractorRegistry.load_defaults()
