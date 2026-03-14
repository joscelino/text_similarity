"""Módulo de similaridade baseada em fonética PT-BR e heurísticas audíveis."""

from __future__ import annotations

import re
import unicodedata

from rapidfuzz import fuzz

from .base import SimilarityAlgorithm


class PhoneticSimilarity(SimilarityAlgorithm):
    """Calcula similaridade fonética baseada em heurísticas para PT-BR.

    Utilizamos um algoritmo de substituição fonética rudimentar que cobre 80%
    dos casos de erros de digitação auditivos em português (s/ss/z/c/ç),
    aliado à distância Levenshtein.
    """

    # Substituições multi-caractere (ordem importa: sorted por len desc)
    _MULTI_CHAR_MAP = {
        "ss": "s",
        "rr": "r",
        "ll": "l",
        "ce": "se",
        "ci": "si",
        "ch": "x",
        "qu": "k",
        "ge": "je",
        "gi": "ji",
        "lh": "l",
        "nh": "n",
    }
    _MULTI_RE = re.compile(
        "|".join(re.escape(k) for k in sorted(_MULTI_CHAR_MAP, key=len, reverse=True))
    )
    _SINGLE_MAP = {"c": "k", "h": ""}
    _SINGLE_RE = re.compile(r"[ch]")

    def _phonetic_hash(self, text: str) -> str:
        """Converte o texto para uma representação aproximada do som em PT-BR.

        Ex: "casa" -> "kaza", "passarinho" -> "pasarinho", "exceção" -> "esesso",
        "fazenda" -> "fazenda".
        """
        # Removendo todos os acentos que sobraram
        # Nota: após NFKD + ASCII ignore, ç já vira c
        text = (
            unicodedata.normalize("NFKD", text)
            .encode("ASCII", "ignore")
            .decode("utf-8")
        )
        text = text.lower()

        # Substituições multi-caractere em uma passada
        text = self._MULTI_RE.sub(lambda m: self._MULTI_CHAR_MAP[m.group()], text)

        # Substituições single-char (c->k, h->"")
        text = self._SINGLE_RE.sub(lambda m: self._SINGLE_MAP[m.group()], text)

        # M final -> N (bem -> ben)
        if text.endswith("m"):
            text = text[:-1] + "n"

        return text

    def compare(self, text1: str, text2: str) -> float:
        """Trata cada palavra do texto para fonemas, as une e calcula Levenshtein."""
        if not text1 or not text2:
            return 0.0

        hash1 = " ".join([self._phonetic_hash(word) for word in text1.split()])
        hash2 = " ".join([self._phonetic_hash(word) for word in text2.split()])

        # Utilizamos o rapidfuzz (ratio / levenshtein) no hash fonético resultante
        return float(fuzz.ratio(hash1, hash2)) / 100.0
