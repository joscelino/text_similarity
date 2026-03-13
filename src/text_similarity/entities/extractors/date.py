"""Módulo que detecta e infere datas temporais relativas e absolutas."""

from __future__ import annotations

import functools
import re
from datetime import date

import dateparser

from ..base import EntityExtractor, EntityMatch
from ..registry import ExtractorRegistry


@functools.lru_cache(maxsize=1024)
def _cached_dateparser_parse(text: str, today: str) -> "str | None":
    """Cache de chamadas ao dateparser.parse() para evitar reprocessamento."""
    parsed = dateparser.parse(text, languages=["pt"])
    return parsed.strftime("%Y-%m-%d") if parsed else None


class DateExtractor(EntityExtractor):
    """Extrator de datas utilizando Regex e Dateparser para resolver datas em PT-BR."""

    _PATTERNS = [
        re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", re.IGNORECASE),
        re.compile(
            r"\b\d{1,2}\s+de\s+[A-Za-zçÇ]+\s+de\s+\d{2,4}\b", re.IGNORECASE
        ),
        re.compile(r"\b(?:amanhã|ontem|hoje)\b", re.IGNORECASE),
        re.compile(r"\bamanhã\b", re.IGNORECASE),
    ]

    def extract(self, text: str) -> list[EntityMatch]:
        """Aplica regex buscando marcações temporais e as padroniza para ISO."""
        matches: list[EntityMatch] = []

        found_spans = set()

        for pat in self._PATTERNS:
            for m in pat.finditer(text):
                start, end = m.span()
                matched_str = text[start:end]

                # Evita duplicatas se as regexes se sobreporem
                if (start, end) in found_spans:
                    continue

                # Usa o dateparser para resolver a string concreta (com cache)
                today_str = date.today().isoformat()
                normalized_val = _cached_dateparser_parse(matched_str, today_str)
                if normalized_val:
                    found_spans.add((start, end))
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
