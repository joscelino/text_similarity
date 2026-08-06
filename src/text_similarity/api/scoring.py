"""Mixin com a lógica de scoring do :class:`Comparator`.

Isola do módulo principal (SEC-STD-001) as três operações de scoring:

- :meth:`ScoringMixin._score_candidates` — dispatcher linear/RRF.
- :meth:`ScoringMixin._score_candidates_linear` — combinação ponderada.
- :meth:`ScoringMixin._score_candidates_rrf` — Reciprocal Rank Fusion.
- :meth:`ScoringMixin._reuse_semantic_from_dense` — guard SEC-LOGIC-002.

Todos os métodos assumem que a classe consumidora expõe os atributos
``algorithm``, ``fusion_strategy``, ``_rrf_fusion``, ``indexing_strategy``,
``dense_precision`` e ``dense_model_name`` (o :class:`Comparator`
satisfaz esse contrato).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List

from text_similarity.core.hybrid import HybridSimilarity

if TYPE_CHECKING:
    from text_similarity.core.fusion import RRFusion


class ScoringMixin:
    """Mixin com scoring linear e RRF do :class:`Comparator`."""

    # Atributos providos pelo Comparator (documentados para o type checker).
    algorithm: Any
    fusion_strategy: str
    _rrf_fusion: "RRFusion | None"
    indexing_strategy: str
    dense_precision: str
    dense_model_name: str
    dense_model_revision: str | None
    _active_index: Any

    def _score_candidates(
        self,
        p_text: str,
        top_candidates: List[dict[str, Any]],
    ) -> List[dict[str, Any]]:
        """Aplica scoring híbrido (entity, edit, phonetic) nos candidatos filtrados.

        Reutilizado internamente por `compare_batch` e `compare_many_to_many`
        para computar os scores finais após a filtragem por cosseno.

        Quando ``fusion_strategy="rrf"``, cada algoritmo produz um ranking
        independente e o resultado final é fundido via Reciprocal Rank Fusion.

        Args:
            p_text: Texto da query já pré-processado.
            top_candidates: Lista de dicts com chaves 'candidate', 'p_candidate'
                e 'cos_score', já filtrados e ordenados por cosseno.

        Returns:
            Lista de dicts com 'candidate', 'score' e 'details', ordenados
            por score final descendente.
        """
        if self.fusion_strategy == "rrf" and self._rrf_fusion is not None:
            return self._score_candidates_rrf(p_text, top_candidates)

        return self._score_candidates_linear(p_text, top_candidates)

    @property
    def _reuse_semantic_from_dense(self) -> bool:
        """Verifica se o score semântico pode ser reutilizado do DenseIndex.

        Verdadeiro quando ``indexing_strategy="dense"``, o modelo do
        DenseIndex é o mesmo do :class:`SemanticSimilarity` e a precisão
        de armazenamento é ``float32``. Reutilizar o ``cos_score`` do
        DenseIndex evita recodificar query e candidatos que já passaram
        pelo encoder na filtragem.

        IMPORTANTE — Restrição de precisão (SEC-LOGIC-002):
            Quando ``dense_precision != "float32"`` (ou seja, ``"int8"``
            ou ``"binary"``), o ``cos_score`` do DenseIndex foi computado
            sobre embeddings quantizados / hamming e **não é comparável
            ao score semântico full-precision** que o
            :class:`SemanticSimilarity` produz em tempo de scoring. Nestes
            casos esta property retorna ``False`` para forçar o recálculo
            explícito via ``SemanticSimilarity.compare()``, preservando a
            fidelidade do ranking.
        """
        if self.indexing_strategy != "dense":
            return False
        if self.dense_precision != "float32":
            return False
        if not isinstance(self.algorithm, HybridSimilarity):
            return False
        semantic = self.algorithm.algorithms.get("semantic")
        if semantic is None:
            return False

        # O reuso só é seguro quando modelo, device e revision coincidem.
        # Se o device ou a revision diferirem, o DenseIndex e o
        # SemanticSimilarity podem estar usando instâncias distintas do
        # SentenceTransformer.
        dense_index = getattr(self, "_active_index", None)
        device_matches = getattr(dense_index, "device", None) == getattr(
            semantic, "device", None
        )
        revision_matches = self.dense_model_revision == getattr(
            semantic, "revision", None
        )
        return (
            self.dense_model_name == getattr(semantic, "model_name", None)
            and device_matches
            and revision_matches
        )

    def _score_candidates_linear(
        self,
        p_text: str,
        top_candidates: List[dict[str, Any]],
    ) -> List[dict[str, Any]]:
        """Scoring via combinação linear ponderada (estratégia padrão)."""
        results: List[dict[str, Any]] = []
        reuse_semantic = self._reuse_semantic_from_dense

        for cand in top_candidates:
            c_p_text = cand["p_candidate"]
            cos_score = cand["cos_score"]

            if isinstance(self.algorithm, HybridSimilarity):
                alg_weights = self.algorithm.weights
                algs = self.algorithm.algorithms
                final_score = 0.0
                details: dict[str, Any] = {}

                # Short-circuit via entidade
                short_circuit = False
                if "entity" in alg_weights and alg_weights["entity"] > 0:
                    ent_score = algs["entity"].compare(p_text, c_p_text)
                    details["entity"] = {
                        "score": ent_score,
                        "weight": alg_weights["entity"],
                    }
                    if ent_score >= 1.0:
                        final_score = 0.95
                        short_circuit = True

                if not short_circuit:
                    details["cosine"] = {
                        "score": cos_score,
                        "weight": alg_weights.get("cosine", 0.0),
                    }
                    final_score += cos_score * alg_weights.get("cosine", 0.0)

                    if "entity" in alg_weights and alg_weights["entity"] > 0:
                        final_score += (
                            details["entity"]["score"] * alg_weights["entity"]
                        )

                    for name in ["edit", "phonetic", "semantic"]:
                        if name in alg_weights and alg_weights[name] > 0:
                            # Reutiliza cos_score do DenseIndex quando o modelo
                            # semântico é o mesmo — evita recodificar query e
                            # candidatos que já passaram pelo encoder na filtragem.
                            if name == "semantic" and reuse_semantic:
                                score = cos_score
                            else:
                                score = algs[name].compare(p_text, c_p_text)
                            details[name] = {
                                "score": score,
                                "weight": alg_weights[name],
                            }
                            final_score += score * alg_weights[name]

                results.append(
                    {
                        "candidate": cand["candidate"],
                        "score": final_score,
                        "details": details,
                    }
                )
            else:
                _score = self.algorithm.compare(p_text, c_p_text)
                results.append(
                    {
                        "candidate": cand["candidate"],
                        "score": _score,
                        "details": {
                            type(self.algorithm).__name__: {
                                "score": _score,
                                "weight": 1.0,
                            }
                        },
                    }
                )

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def _score_candidates_rrf(
        self,
        p_text: str,
        top_candidates: List[dict[str, Any]],
    ) -> List[dict[str, Any]]:
        """Scoring via Reciprocal Rank Fusion.

        Cada algoritmo ativo produz um ranking independente dos candidatos.
        Os rankings são fundidos pelo RRFusion, priorizando candidatos
        que aparecem consistentemente no topo de múltiplas listas.
        """
        if not top_candidates or not isinstance(self.algorithm, HybridSimilarity):
            return []

        alg_weights = self.algorithm.weights
        algs = self.algorithm.algorithms

        # Identificar algoritmos ativos
        active_algos: List[str] = []
        for name in ["cosine", "entity", "edit", "phonetic", "semantic"]:
            if name in alg_weights and alg_weights[name] > 0:
                active_algos.append(name)

        if not active_algos:
            return []

        # Montar um ranking por algoritmo
        per_algo_rankings: List[List[dict[str, Any]]] = []
        reuse_semantic = self._reuse_semantic_from_dense

        for algo_name in active_algos:
            ranking: List[dict[str, Any]] = []

            for cand in top_candidates:
                c_p_text = cand["p_candidate"]

                if algo_name == "cosine":
                    score = cand["cos_score"]
                elif algo_name == "semantic" and reuse_semantic:
                    # Reutiliza cos_score do DenseIndex — mesmo modelo, mesmo encoder.
                    score = cand["cos_score"]
                else:
                    score = algs[algo_name].compare(p_text, c_p_text)

                ranking.append({"candidate": cand["candidate"], "score": score})

            ranking.sort(key=lambda x: x["score"], reverse=True)
            per_algo_rankings.append(ranking)

        assert self._rrf_fusion is not None
        return self._rrf_fusion.fuse(per_algo_rankings, active_algos)


__all__ = ["ScoringMixin"]
