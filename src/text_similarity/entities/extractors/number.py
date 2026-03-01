"""Módulo extrator de numerais, mapeando algarismos e numerais cardinais em extenso."""

from __future__ import annotations

import re

from ..base import EntityExtractor, EntityMatch
from ..registry import ExtractorRegistry


class NumberExtractor(EntityExtractor):
    """Extrator de números (dígitos e numerais cardinais básicos)."""

    # Mapeamento estático e simples pro MVP.
    # Em produção, uma lib como 'word2number' seria mais completa.
    MAP_NUMBERS = {
        "zero": 0,
        "um": 1,
        "uma": 1,
        "dois": 2,
        "duas": 2,
        "tres": 3,
        "três": 3,
        "quatro": 4,
        "cinco": 5,
        "seis": 6,
        "sete": 7,
        "oito": 8,
        "nove": 9,
        "dez": 10,
        "cem": 100,
        "mil": 1000,
    }

    def extract(self, text: str) -> list[EntityMatch]:
        """Localiza ocorrências de numerais por Regex e converte numerais por extenso através de mapeamento em dict."""
        matches: list[EntityMatch] = []

        # Dígitos numéricos isolados
        pattern_digits = r"\b(\d+(?:[\.,]\d+)?)\b"

        for m in re.finditer(pattern_digits, text, flags=re.IGNORECASE):
            start, end = m.span()
            matched_str = text[start:end]
            val_str = m.group(1).replace(",", ".")
            try:
                numeric_val = float(val_str)
                # simplifica pra int se for ".0"
                if numeric_val.is_integer():
                    numeric_val = int(numeric_val)

                matches.append(
                    EntityMatch(
                        entity_type="number",
                        value=numeric_val,
                        start=start,
                        end=end,
                        text_matched=matched_str,
                    )
                )
            except ValueError:
                pass

        # Numerais escritos
        text.split()
        for match_iter in re.finditer(r"\b([a-zA-ZçÇêÊ]+\b)", text):
            word = match_iter.group(1).lower()
            if word in self.MAP_NUMBERS:
                start, end = match_iter.span()
                matches.append(
                    EntityMatch(
                        entity_type="number",
                        value=self.MAP_NUMBERS[word],
                        start=start,
                        end=end,
                        text_matched=match_iter.group(1),
                    )
                )

        matches.sort(key=lambda m: m.start)
        return matches


ExtractorRegistry.register("number", NumberExtractor)
