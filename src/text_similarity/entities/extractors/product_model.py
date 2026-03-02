"""Módulo para a detecção de modelos e classes de entidades combinando alfanuméricos."""

from __future__ import annotations

import re

from ..base import EntityExtractor, EntityMatch
from ..registry import ExtractorRegistry


class ProductModelExtractor(EntityExtractor):
    """Extrator de modelos de produtos.

    Foca em códigos alfanuméricos comuns em eletrônicos, peças e afins.
    Ex: "S22", "XJ-900", "iPhone 13", "M1", "RFX765J9".

    O texto de entrada já deve ter sido pré-normalizado pelo
    `NormalizeEntitiesStage` (via `_collapse_model_spaces`) para que
    padrões como "RFX 765J9" (com espaço interno) sejam colados como
    "RFX765J9" antes de chegar ao extrator.
    """

    def extract(self, text: str) -> list[EntityMatch]:
        """Extrai classes de referência técnica isolando letras hifenizadas e contíguas de numerais."""
        matches: list[EntityMatch] = []

        # Padrões para modelos de produtos:
        # 1. Letras seguidas de números, com ou sem hífen (ex: S22, A-50, 4K, RFX765J9)
        # 2. Números seguidos de letras (ex: 128GB, 4K, 55pol)
        # Requer conter ambos letras e números para não casar com
        # palavras normais ou números puros.
        pattern = (
            r"\b(?:[A-Za-z]+[-]?\d+[A-Za-z\d]*|"
            r"\d+[-]?[A-Za-z]+[A-Za-z\d]*)\b"
        )

        for m in re.finditer(pattern, text):
            start, end = m.span()
            matched_str = text[start:end]

            matches.append(
                EntityMatch(
                    entity_type="productmodel",
                    value=matched_str.upper(),
                    start=start,
                    end=end,
                    text_matched=matched_str,
                )
            )

        matches.sort(key=lambda m: m.start)
        return matches


ExtractorRegistry.register("product_model", ProductModelExtractor)
