"""Módulo implementando a similaridade híbrida customizada ponderada."""

from __future__ import annotations

from typing import Any

from .base import SimilarityAlgorithm
from .cosine import CosineSimilarity
from .phonetic import PhoneticSimilarity
from .rapidfuzz_cmp import EditDistanceSimilarity


class HybridSimilarity(SimilarityAlgorithm):
    """Calcula uma similaridade combinada (ponderada) utilizando múltiplos.
    
    Modelos disponíveis (TF-IDF Cosseno, Distância de Edição e Fonética).
    """

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        """Inicializa agregador de distâncias computacionais simultâneas.
        
        Args:
            weights: Dicionário identificando o peso relativo de cada
                algoritmo no resultado final.
                Padrão: {"cosine": 0.4, "edit": 0.4, "phonetic": 0.2}.
        """
        self.weights = weights or {"cosine": 0.4, "edit": 0.4, "phonetic": 0.2}

        # Normalizando pesos para somarem 1.0
        total_weight = sum(self.weights.values())
        if total_weight == 0:
            total_weight = 1.0
        self.weights = {k: v / total_weight for k, v in self.weights.items()}

        # Instanciando algoritmos
        self.algorithms = {
            "cosine": CosineSimilarity(),
            "edit": EditDistanceSimilarity(method="ratio"),
            "phonetic": PhoneticSimilarity()
        }

    def compare(self, text1: str, text2: str) -> float:
        """Soma iterativamente as ponderações de cada algoritmo."""
        if not text1 or not text2:
            return 0.0

        final_score = 0.0
        for name, alg in self.algorithms.items():
            if name in self.weights and self.weights[name] > 0:
                score = alg.compare(text1, text2)
                final_score += score * self.weights[name]

        return final_score

    def explain(self, text1: str, text2: str) -> dict[str, Any]:
        """Funcionalidade especial sugerida: explica a contribuição de cada."""
        if not text1 or not text2:
            return {"score": 0.0, "details": {}}

        details = {}
        final_score = 0.0
        for name, alg in self.algorithms.items():
            if name in self.weights and self.weights[name] > 0:
                score = alg.compare(text1, text2)
                details[name] = {"score": score, "weight": self.weights[name]}
                final_score += score * self.weights[name]

        return {
            "score": final_score,
            "details": details
        }
