"""Módulo que detecta e extrai valores monetários como reais e centavos."""

from __future__ import annotations

import re

from ..base import EntityExtractor, EntityMatch
from ..registry import ExtractorRegistry


class MoneyExtractor(EntityExtractor):
    """Extrator de valores monetários utilizando Regex.

    Captura "R$ 30,00", "50 reais", etc.
    """

    _RE_MONEY = re.compile(
        r"(?:(?:R\$|BRL)\s*([\d\.,]+))|(?:([\d\.,]+)\s+(?:reais|centavos|BRL))",
        re.IGNORECASE,
    )

    def extract(self, text: str) -> list[EntityMatch]:
        """Aplica regex de montantes buscando números com prefixos monetários."""
        matches: list[EntityMatch] = []

        for m in self._RE_MONEY.finditer(text):
            start, end = m.span()
            matched_str = text[start:end]

            # Pega o valor purificado
            val_str = m.group(1) if m.group(1) else m.group(2)
            if not val_str:
                continue

            # Tratamento de formato PT-BR (1.500,50 -> 1500.50) e (1500,50 -> 1500.50)
            if "," in val_str:
                val_limpo = val_str.replace(".", "").replace(",", ".")
            else:
                # Se não tem vírgula, assumimos que o ponto pode ser
                # decimal (anglo-saxão) ou ignoramos formato milhar.
                val_limpo = val_str

            try:
                numeric_val = float(val_limpo)
            except ValueError:
                continue

            matches.append(
                EntityMatch(
                    entity_type="money",
                    value=numeric_val,
                    start=start,
                    end=end,
                    text_matched=matched_str,
                )
            )

        matches.sort(key=lambda m: m.start)
        return matches


ExtractorRegistry.register("money", MoneyExtractor)
