"""Módulo que detecta e infere datas temporais relativas e absolutas."""

from __future__ import annotations

import re

import dateparser

from ..base import EntityExtractor, EntityMatch
from ..registry import ExtractorRegistry


class DateExtractor(EntityExtractor):
    """Extrator de datas utilizando Regex e Dateparser para resolver datas em PT-BR."""

    def extract(self, text: str) -> list[EntityMatch]:
        """Aplica expressões regulares buscando marcações temporais e as padroniza para ISO."""
        matches: list[EntityMatch] = []

        # Padrões comuns de datas em português
        patterns = [
            r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",  # 25/04/2024
            r"\b\d{1,2}\s+de\s+[A-Za-zçÇ]+\s+de\s+\d{2,4}\b",  # 25 de abril de 2024
            r"\b(?:amanhã|ontem|hoje)\b",  # relativas (desconsidera acentos)
        ]

        # Também tentaremos variações em Unicode
        # r"\bamanhã\b"
        patterns.append(r"\bamanhã\b")

        found_spans = set()

        for pattern in patterns:
            for m in re.finditer(pattern, text, flags=re.IGNORECASE):
                start, end = m.span()
                matched_str = text[start:end]

                # Evita duplicatas se as regexes se sobreporem
                if (start, end) in found_spans:
                    continue

                # Usa o dateparser para resolver a string concreta
                parsed = dateparser.parse(matched_str, languages=["pt"])
                if parsed:
                    found_spans.add((start, end))
                    normalized_val = parsed.strftime("%Y-%m-%d")
                    matches.append(
                        EntityMatch(
                            entity_type="date",
                            value=normalized_val,
                            start=start,
                            end=end,
                            text_matched=matched_str,
                        )
                    )

        matches.sort(key=lambda m: m.start)
        return matches


ExtractorRegistry.register("date", DateExtractor)
