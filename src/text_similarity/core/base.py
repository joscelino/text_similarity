from __future__ import annotations

import abc


class SimilarityAlgorithm(abc.ABC):
    """
    Interface base para algoritmos de cálculo de similaridade entre duas strings.
    """

    @abc.abstractmethod
    def compare(self, text1: str, text2: str) -> float:
        """
        Recebe dois textos pré-processados e retorna um score de similaridade
        entre 0.0 (totalmente diferente) e 1.0 (idêntico).
        """
        pass
