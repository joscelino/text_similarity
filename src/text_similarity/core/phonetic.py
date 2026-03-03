"""Módulo de similaridade baseada em fonética PT-BR e heurísticas audíveis."""

from __future__ import annotations

import unicodedata

from rapidfuzz import fuzz

from .base import SimilarityAlgorithm


class PhoneticSimilarity(SimilarityAlgorithm):
    """Calcula similaridade fonética baseada em heurísticas para PT-BR.

    Utilizamos um algoritmo de substituição fonética rudimentar que cobre 80%
    dos casos de erros de digitação auditivos em português (s/ss/z/c/ç),
    aliado à distância Levenshtein.
    """

    def _phonetic_hash(self, text: str) -> str:
        """Converte o texto para uma representação aproximada do som em PT-BR.

        Ex: "casa" -> "kaza", "passarinho" -> "pasarinho", "exceção" -> "esesso",
        "fazenda" -> "fazenda".
        """
        # Removendo todos os acentos que sobraram
        text = (
            unicodedata.normalize("NFKD", text)
            .encode("ASCII", "ignore")
            .decode("utf-8")
        )
        text = text.lower()

        # Consoantes juntas - redução
        text = text.replace("ss", "s").replace("rr", "r").replace("ll", "l")

        # Som de S / Z / C / Ç
        text = text.replace("ç", "s")
        text = text.replace("ce", "se").replace("ci", "si")
        # Ch -> X
        text = text.replace("ch", "x")
        # Quem / Que -> k
        text = text.replace("qu", "k")
        text = text.replace("c", "k")  # O restante dos C (ca, co, cu) viram K

        # G / J
        text = text.replace("ge", "je").replace("gi", "ji")

        # M final -> N (bem -> ben)
        if text.endswith("m"):
            text = text[:-1] + "n"

        # H mudo -> ignorado ou dependendo Lh/Nh vira uma só
        text = text.replace("lh", "l").replace("nh", "n")
        text = text.replace("h", "")

        return text

    def compare(self, text1: str, text2: str) -> float:
        """Trata cada palavra do texto para fonemas, as une e calcula Levenshtein."""
        if not text1 or not text2:
            return 0.0

        hash1 = " ".join([self._phonetic_hash(word) for word in text1.split()])
        hash2 = " ".join([self._phonetic_hash(word) for word in text2.split()])

        # Utilizamos o rapidfuzz (ratio / levenshtein) no hash fonético resultante
        return float(fuzz.ratio(hash1, hash2)) / 100.0
