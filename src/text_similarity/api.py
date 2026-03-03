"""Módulo principal da API pública da biblioteca.

A interface central para comparação é através da classe `Comparator`.
"""

from __future__ import annotations

from typing import Any, List

from text_similarity.core.base import SimilarityAlgorithm
from text_similarity.core.hybrid import HybridSimilarity
from text_similarity.pipeline.backends import (
    CleanTextStage,
    LemmatizeStage,
    NormalizeEntitiesStage,
    StopwordsStage,
    TokenizerStage,
)
from text_similarity.pipeline.cache import PipelineCache
from text_similarity.pipeline.pipeline import PreprocessingPipeline
from text_similarity.pipeline.stage import PipelineStage


class Comparator:
    """Classe principal para comparação de similaridade de textos em português."""

    def __init__(
        self,
        mode: str = "basic",
        entities: list[str] | None = None,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> None:
        """Inicializa a classe Comparator preparando o pipeline.

        Args:
            mode: Modo de operação ('basic' ou 'smart').
            entities: Lista de entidades para extrair no modo smart.
            use_cache: Se True, habilita o cache in-memory de textos processados.
            **kwargs: Argumentos arbitrários reservados para extensões futuras.
        """
        self.mode = mode
        self.entities = entities
        self.use_cache = use_cache

        # Cache in-memory: hash SHA-256 do texto original → texto pré-processado
        self.cache: PipelineCache | None = PipelineCache() if use_cache else None
        self._cache_store: dict[str, str] = {}

        # Configuração do Pipeline
        stages: List[PipelineStage] = []
        if self.mode == "smart":
            # Smart habilita a normalização de entidades antes da limpeza
            from text_similarity.entities.normalizer import EntityNormalizer

            stages.append(
                NormalizeEntitiesStage(
                    normalizer=EntityNormalizer(entities=self.entities)
                )
            )

        stages.extend(
            [
                CleanTextStage(),
                TokenizerStage(),
                StopwordsStage(),
                LemmatizeStage(),
            ]
        )
        self.pipeline = PreprocessingPipeline(stages)

        # Configuração do Algoritmo
        if self.mode == "smart":
            # Dá peso maior para a fonética e entidades exatas (tokens)
            self.algorithm: SimilarityAlgorithm = HybridSimilarity(
                weights={"cosine": 0.45, "edit": 0.25, "phonetic": 0.20, "entity": 0.10}
            )
        else:
            self.algorithm = HybridSimilarity(
                weights={"cosine": 0.5, "edit": 0.5, "phonetic": 0.0}
            )

    @classmethod
    def basic(cls) -> "Comparator":
        """Instancia um Comparator no modo básico.

        Sem detecção algorítmica de entidades e focado em Bag of Words + Levenshtein.
        """
        return cls(mode="basic")

    @classmethod
    def smart(
        cls,
        entities: list[str] | None = None,
        use_cache: bool = True,
    ) -> "Comparator":
        """Instancia um Comparator no modo inteligente.

        Ativa a extração de entidades, unifica tokens, analisa a fonética PT-BR
        e cruza resultados de múltiplos algoritmos.
        """
        return cls(mode="smart", entities=entities, use_cache=use_cache)

    def _process(self, text: str) -> str:
        """Pré-processa o texto pelo pipeline, com cache in-memory.

        Na primeira chamada para um texto, executa o pipeline completo e
        armazena o resultado. Chamadas subsequentes com o mesmo texto retornam
        o resultado cacheado diretamente.

        Args:
            text: Texto bruto de entrada.

        Returns:
            Texto pré-processado como bag of words.
        """
        if self.cache is not None:
            key = self.cache.hash_text(text)
            if key in self._cache_store:
                return self._cache_store[key]

        processed, _ = self.pipeline.process(text)

        if self.cache is not None:
            self._cache_store[key] = processed

        return processed

    def clear_cache(self) -> None:
        """Limpa o cache in-memory e o cache em disco do Joblib."""
        self._cache_store.clear()
        if self.cache is not None:
            self.cache.clear()

    def compare(self, text1: str, text2: str) -> float:
        """Compara dois textos e retorna um valor global de similaridade.

        Args:
            text1: Primeiro texto para comparação.
            text2: Segundo texto para comparação.

        Returns:
            Score entre 0.0 (completamente diferentes) e 1.0 (idênticos).
        """
        p_text1 = self._process(text1)
        p_text2 = self._process(text2)
        return self.algorithm.compare(p_text1, p_text2)

    def explain(self, text1: str, text2: str) -> dict[str, Any]:
        """Retorna as predições individuais de todos os algoritmos rodados no texto.

        Args:
            text1: Primeiro texto para comparação.
            text2: Segundo texto para comparação.

        Returns:
            Dicionário com 'score' (float) e 'details' (dict por algoritmo).
        """
        p_text1 = self._process(text1)
        p_text2 = self._process(text2)

        if isinstance(self.algorithm, HybridSimilarity):
            return self.algorithm.explain(p_text1, p_text2)

        score = self.algorithm.compare(p_text1, p_text2)
        return {"score": score, "details": {"algorithm": score}}

    def compare_batch(
        self,
        text: str,
        candidates: List[str],
        top_n: int = 50,
        min_cosine: float = 0.1,
    ) -> List[dict[str, Any]]:
        """Compara um único texto contra uma lista de candidatos em lote.

        Otimiza o processo computando a matriz TF-IDF de todos os elementos
        (query + candidatos) de uma só vez, e extraindo os candidatos que passam
        num limiar mínimo de cosseno (min_cosine) para só então aplicar
        as similaridades mais custosas (fonética, distância de edição).

        Args:
            text: Texto principal para buscar.
            candidates: Lista de textos candidatos.
            top_n: Número máximo de candidatos filtrados para a etapa final.
            min_cosine: Limiar mínimo de cosseno para descartar ruidosos.

        Returns:
            Lista de dicionários, ordenados do maior score final para o menor,
            contendo o candidato original, o score e os detalhes da similaridade.
        """
        if not candidates:
            return []

        # 1. Pré-processamento
        p_text = self._process(text)
        p_candidates = [self._process(c) for c in candidates]

        # 2. Computar matriz TF-IDF de todos (query = índice 0, candidatos = 1..)
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import (
            cosine_similarity as sklearn_cosine_similarity,
        )

        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        all_texts = [p_text] + p_candidates

        try:
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            # cosine_similarity retorna (1, len(all_texts)), ignoramos query x query
            cosine_scores = sklearn_cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)[
                0
            ][1:]
        except ValueError:
            # Vocabulary empty
            cosine_scores = [0.0] * len(candidates)

        # 3. Filtrar e encontrar o topN inicial com base no cosseno
        scored_candidates = []
        for idx, (c_text, c_p_text, cos_score) in enumerate(
            zip(candidates, p_candidates, cosine_scores)
        ):
            if cos_score >= min_cosine:
                scored_candidates.append(
                    {
                        "idx": idx,
                        "candidate": c_text,
                        "p_candidate": c_p_text,
                        "cos_score": float(cos_score),
                    }
                )

        # Ordenar os filtrados e pegar os top N
        scored_candidates.sort(key=lambda x: x["cos_score"], reverse=True)
        top_candidates = scored_candidates[:top_n]

        # 4. Processamento híbrido final
        results = []
        for cand in top_candidates:
            c_p_text = cand["p_candidate"]
            cos_score = cand["cos_score"]

            # Aproveitar a arquitetura do HybridSimilarity customizado
            if isinstance(self.algorithm, HybridSimilarity):
                alg_weights = self.algorithm.weights
                algs = self.algorithm.algorithms
                final_score = 0.0
                details = {}

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

                    for name in ["edit", "phonetic"]:
                        if name in alg_weights and alg_weights[name] > 0:
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
                # Fallback, executa o método comum (mas como é batch esperamos Hybrid)
                results.append(
                    {
                        "candidate": cand["candidate"],
                        "score": self.algorithm.compare(p_text, c_p_text),
                        "details": {
                            "algorithm": self.algorithm.compare(p_text, c_p_text)
                        },
                    }
                )

        # Ordenar resultados de forma descendente no score final
        results.sort(key=lambda x: x["score"], reverse=True)
        return results
