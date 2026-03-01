from __future__ import annotations

from rapidfuzz import fuzz

from .base import SimilarityAlgorithm


class EditDistanceSimilarity(SimilarityAlgorithm):
    """
    Calcula similaridade baseada em Distância de Edição / Levenshtein.
    Utiliza o módulo ultrarrápido rapidfuzz.
    """

    def __init__(self, method: str = "ratio") -> None:
        """
        Args:
            method: 'ratio' (Levenshtein puro) ou
                    'partial_ratio' (Bom para pedaços inclusos em palavras maiores),
                    'token_sort_ratio' (Não importa a ordem das palavras).
        """
        self.method = method

    def compare(self, text1: str, text2: str) -> float:
        if not text1 or not text2:
            return 0.0

        if self.method == "partial_ratio":
            # RapidFuzz retorna 0-100, nós retornamos 0.0 - 1.0
            return float(fuzz.partial_ratio(text1, text2)) / 100.0
        elif self.method == "token_sort_ratio":
            return float(fuzz.token_sort_ratio(text1, text2)) / 100.0
        else:
            return float(fuzz.ratio(text1, text2)) / 100.0
