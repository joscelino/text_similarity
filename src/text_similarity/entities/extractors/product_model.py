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

    _RE_MODEL = re.compile(
        r"\b(?:[A-Za-z]+[-]?\d+[A-Za-z\d]*|\d+[-]?[A-Za-z]+[A-Za-z\d]*)\b"
    )

    def extract(self, text: str) -> list[EntityMatch]:
        """Extrai referências técnicas isolando letras contíguas de numerais."""
        matches: list[EntityMatch] = []

        for m in self._RE_MODEL.finditer(text):
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
