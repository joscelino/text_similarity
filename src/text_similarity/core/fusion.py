"""Módulo de fusão de rankings via Reciprocal Rank Fusion (RRF).

Implementa a combinação de múltiplas listas ranqueadas (uma por algoritmo)
em um ranking único, baseando-se na posição dos candidatos em vez dos
scores brutos. Isso elimina a necessidade de normalização de escalas
entre algoritmos distintos.
"""

from __future__ import annotations

from typing import Any, Dict, List


class RRFusion:
    """Reciprocal Rank Fusion para combinação de rankings heterogêneos.

    Funde resultados de diferentes algoritmos (ex: Léxico e Semântico)
    baseando-se na posição dos candidatos em cada ranking, em vez de
    seus scores brutos. A fórmula aplicada é: ``score = Σ 1/(k + rank)``.

    Args:
        k: Parâmetro de suavização que controla a influência de itens
            em posições baixas no ranking. Valores maiores atenuam a
            diferença entre posições (padrão 60, conforme literatura).
    """

    def __init__(self, k: int = 60) -> None:
        """Inicializa o fusionador com a constante de suavização k."""
        self.k = k

    def fuse(
        self,
        rankings: List[List[Dict[str, Any]]],
        algorithm_names: List[str],
    ) -> List[Dict[str, Any]]:
        """Funde múltiplas listas de resultados em um ranking único via RRF.

        Args:
            rankings: Lista de listas, uma por algoritmo. Cada sublista
                contém dicts com ``{"candidate": str, "score": float}``,
                ordenados por score descendente.
            algorithm_names: Nomes dos algoritmos na mesma ordem de
                ``rankings``, usados para montar o dict de detalhes.

        Returns:
            Lista consolidada ordenada por score RRF descendente.
            Cada item contém::

                {
                    "candidate": str,
                    "score": float,        # RRF normalizado em [0, 1]
                    "fusion": "rrf",
                    "details": {
                        "algo_name": {
                            "rank": int,
                            "raw_score": float,
                            "rrf_contribution": float,
                        },
                        ...
                    }
                }
        """
        n_algorithms = len(rankings)
        if n_algorithms == 0:
            return []

        # Máximo teórico: candidato em rank 1 em todos os algoritmos
        max_rrf = n_algorithms / (self.k + 1)

        # Acumuladores por candidato
        rrf_scores: Dict[str, float] = {}
        details: Dict[str, Dict[str, Dict[str, Any]]] = {}

        for algo_idx, ranking in enumerate(rankings):
            algo_name = algorithm_names[algo_idx]
            ranking_size = len(ranking)

            # Mapear candidatos presentes nesta lista
            candidates_in_ranking = set()

            for rank, item in enumerate(ranking, start=1):
                candidate = item["candidate"]
                raw_score = item["score"]
                candidates_in_ranking.add(candidate)

                rrf_contribution = 1.0 / (self.k + rank)

                rrf_scores[candidate] = (
                    rrf_scores.get(candidate, 0.0) + rrf_contribution
                )

                if candidate not in details:
                    details[candidate] = {}
                details[candidate][algo_name] = {
                    "rank": rank,
                    "raw_score": raw_score,
                    "rrf_contribution": rrf_contribution,
                }

            # Penalizar candidatos ausentes desta lista
            penalty_rank = ranking_size + 1
            penalty_contribution = 1.0 / (self.k + penalty_rank)

            for candidate in rrf_scores:
                if candidate not in candidates_in_ranking:
                    rrf_scores[candidate] += penalty_contribution

                    if candidate not in details:
                        details[candidate] = {}
                    if algo_name not in details[candidate]:
                        details[candidate][algo_name] = {
                            "rank": penalty_rank,
                            "raw_score": 0.0,
                            "rrf_contribution": penalty_contribution,
                        }

        # Montar resultado normalizado
        combined: List[Dict[str, Any]] = []
        for candidate, raw_rrf in rrf_scores.items():
            normalized_score = raw_rrf / max_rrf if max_rrf > 0 else 0.0
            normalized_score = min(1.0, normalized_score)

            combined.append(
                {
                    "candidate": candidate,
                    "score": normalized_score,
                    "fusion": "rrf",
                    "details": details.get(candidate, {}),
                }
            )

        combined.sort(key=lambda x: x["score"], reverse=True)
        return combined
