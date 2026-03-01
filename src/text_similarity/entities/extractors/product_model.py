from __future__ import annotations

import re

from ..base import EntityExtractor, EntityMatch
from ..registry import ExtractorRegistry


class ProductModelExtractor(EntityExtractor):
    """
    Extrator de modelos de produtos.
    Foca em códigos alfanuméricos comuns em eletrônicos, peças e afins.
    Ex: "S22", "XJ-900", "iPhone 13", "M1".
    """

    def extract(self, text: str) -> list[EntityMatch]:
        matches: list[EntityMatch] = []

        # Padrões para modelos de produtos
        # 1. Letras seguidas de números ou vice-versa, com ou sem hífen
        # (ex: S22, A-50, 4K, 128GB, XJ-900)
        # Requer conter ambos letras e números para não casar com
        # palavras normais ou números puros.
        # Nós procuramos limites de palavras para isolar.
        pattern = (
            r"\b(?:[A-Za-z]+[-]?\d+[A-Za-z\d]*|"
            r"\d+[-]?[A-Za-z]+[A-Za-z\d]*)\b"
        )

        for m in re.finditer(pattern, text):
            start, end = m.span()
            matched_str = text[start:end]

            # Filtro para ignorar palavras que podem ser dimensões comuns
            # pegas acidentalmente por serem número + letra, se for o caso
            # do DimensionExtractor cobrir, mas pra garantir deixamos o Match
            # rolar e dependendo o Normalizer prioriza.
            # Aqui a heurística vai varrer livremente.
            matches.append(
                EntityMatch(
                    entity_type="product_model",
                    value=matched_str.upper(),  # Normalizamos pra maiúsculo
                    start=start,
                    end=end,
                    text_matched=matched_str,
                )
            )

        matches.sort(key=lambda m: m.start)
        return matches


ExtractorRegistry.register("product_model", ProductModelExtractor)
