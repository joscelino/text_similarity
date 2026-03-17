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

    # Aceita separador ponto ou hífen entre grupos alfanuméricos.
    # Requer ao menos um grupo de letras E um grupo de dígitos para evitar
    # falsos positivos com decimais puros (ex: "1.5", "30.00") ou texto puro.
    _RE_MODEL = re.compile(
        r"(?<!\w)"
        r"(?:"
        r"[A-Za-z]+(?:[.\-]\d+)+(?:[.\-][A-Za-z\d]+)*"  # XX.NNN.NN (letra+ponto)
        r"|"
        r"\d+(?:[.\-][A-Za-z]+)+(?:[.\-][A-Za-z\d]+)*"  # 123.AB (dígito+alfa)
        r"|"
        r"[A-Za-z]+[-]?\d+[A-Za-z\d]*"  # S22, XJ-900, RFX765J9  (sem separador ponto)
        r"|"
        r"\d+[-]?[A-Za-z]+[A-Za-z\d]*"  # 3M, 250QS   (dígito-primeiro clássico)
        r")"
        r"(?!\w)"
    )

    def extract(self, text: str) -> list[EntityMatch]:
        """Extrai referências técnicas isolando letras contíguas de numerais."""
        matches: list[EntityMatch] = []

        for m in self._RE_MODEL.finditer(text):
            start, end = m.span()
            matched_str = text[start:end]
            # Remove pontos do valor normalizado para comparação canônica:
            # QS.250.08 → QS25008, XJ-900 permanece XJ-900
            normalized_value = matched_str.upper().replace(".", "")

            matches.append(
                EntityMatch(
                    entity_type="productmodel",
                    value=normalized_value,
                    start=start,
                    end=end,
                    text_matched=matched_str,
                )
            )

        matches.sort(key=lambda m: m.start)
        return matches


ExtractorRegistry.register("product_model", ProductModelExtractor)
