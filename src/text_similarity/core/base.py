"""Módulo base definindo interfaces dos algoritmos de cálculo de similaridade."""

from __future__ import annotations

import abc


class SimilarityAlgorithm(abc.ABC):
    """Interface base para algoritmos de cálculo de similaridade entre duas strings."""

    @abc.abstractmethod
    def compare(self, text1: str, text2: str) -> float:
        """Recebe dois textos pré-processados e retorna um score numérico.

        A pontuação varia estritamente de 0.0 (totalmente diferente) a 1.0 (idêntico).
        """
        pass
