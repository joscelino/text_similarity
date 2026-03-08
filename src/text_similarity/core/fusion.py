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
    seus scores brutos.

    Quando ``weights`` é fornecido, aplica a fórmula ponderada:
    ``score = Σ weight_i * 1/(k + rank_i)``.

    Quando ``weights`` é ``None`` (padrão), todos os algoritmos
    contribuem igualmente: ``score = Σ 1/(k + rank)``.

    Args:
        k: Parâmetro de suavização que controla a influência de itens
            em posições baixas no ranking. Valores maiores atenuam a
            diferença entre posições (padrão 60, conforme literatura).
        weights: Dicionário de pesos por algoritmo (ex:
            ``{"cosine": 0.6, "semantic": 0.4}``). Se ``None``,
            todos os algoritmos recebem peso igual (1.0).
    """

    def __init__(
        self,
        k: int = 60,
        weights: Dict[str, float] | None = None,
    ) -> None:
        """Inicializa o fusionador com a constante de suavização k e pesos opcionais."""
        self.k = k
        self.weights = weights

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

        # Resolver pesos por algoritmo
        algo_weights: Dict[str, float] = {}
        for name in algorithm_names:
            if self.weights is not None and name in self.weights:
                algo_weights[name] = self.weights[name]
            else:
                algo_weights[name] = 1.0

        # Máximo teórico: candidato em rank 1 em todos os algoritmos
        max_rrf = sum(w / (self.k + 1) for w in algo_weights.values())

        # Acumuladores por candidato
        rrf_scores: Dict[str, float] = {}
        details: Dict[str, Dict[str, Dict[str, Any]]] = {}

        for algo_idx, ranking in enumerate(rankings):
            algo_name = algorithm_names[algo_idx]
            weight = algo_weights[algo_name]
            ranking_size = len(ranking)

            # Mapear candidatos presentes nesta lista
            candidates_in_ranking = set()

            for rank, item in enumerate(ranking, start=1):
                candidate = item["candidate"]
                raw_score = item["score"]
                candidates_in_ranking.add(candidate)

                rrf_contribution = weight * 1.0 / (self.k + rank)

                rrf_scores[candidate] = (
                    rrf_scores.get(candidate, 0.0) + rrf_contribution
                )

                if candidate not in details:
                    details[candidate] = {}
                details[candidate][algo_name] = {
                    "rank": rank,
                    "raw_score": raw_score,
                    "rrf_contribution": rrf_contribution,
                    "weight": weight,
                }

            # Penalizar candidatos ausentes desta lista
            penalty_rank = ranking_size + 1
            penalty_contribution = weight * 1.0 / (self.k + penalty_rank)

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
                            "weight": weight,
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
