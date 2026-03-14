"""Módulo que detecta dimensões físicas (peso, comprimento, etc) de produtos."""

from __future__ import annotations

import re

from ..base import EntityExtractor, EntityMatch
from ..registry import ExtractorRegistry


class DimensionExtractor(EntityExtractor):
    """Extrator de dimensões físicas (kg, mg, m, cm, l, ml, etc)."""

    _RE_DIM = re.compile(
        r"(\d+(?:[\.,]\d+)?)\s*(kg|g|mg|m|cm|mm|l|ml)\b", re.IGNORECASE
    )

    def extract(self, text: str) -> list[EntityMatch]:
        """Varre o texto com regex buscando medidas associadas a unidades."""
        matches: list[EntityMatch] = []

        for m in self._RE_DIM.finditer(text):
            start, end = m.span()
            matched_str = text[start:end]

            val_str = m.group(1).replace(",", ".")
            unit = m.group(2).lower()

            try:
                numeric_val = float(val_str)
            except ValueError:
                continue

            # O valor final pode ser retornado composto ex: "2.0:kg"
            final_val = f"{numeric_val}:{unit}"

            matches.append(
                EntityMatch(
                    entity_type="dimension",
                    value=final_val,
                    start=start,
                    end=end,
                    text_matched=matched_str,
                )
            )

        matches.sort(key=lambda m: m.start)
        return matches


ExtractorRegistry.register("dimension", DimensionExtractor)
