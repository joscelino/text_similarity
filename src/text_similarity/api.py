"""Módulo principal da API pública da biblioteca.

A interface central para comparação é através da classe `Comparator`.
"""

from __future__ import annotations

from typing import Any, List, Literal

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

    def _process_batch(self, texts: List[str]) -> List[str]:
        """Pré-processa uma lista de textos em lote, reutilizando cache.

        Cada texto é processado pelo pipeline e armazenado no cache in-memory.
        Textos já processados anteriormente retornam direto do cache,
        evitando reprocessamento redundante.

        Args:
            texts: Lista de textos brutos de entrada.

        Returns:
            Lista de textos pré-processados como bags of words.
        """
        return [self._process(text) for text in texts]

    def _score_candidates(
        self,
        p_text: str,
        top_candidates: List[dict[str, Any]],
    ) -> List[dict[str, Any]]:
        """Aplica scoring híbrido (entity, edit, phonetic) nos candidatos filtrados.

        Reutilizado internamente por `compare_batch` e `compare_many_to_many`
        para computar os scores finais após a filtragem por cosseno.

        Args:
            p_text: Texto da query já pré-processado.
            top_candidates: Lista de dicts com chaves 'candidate', 'p_candidate'
                e 'cos_score', já filtrados e ordenados por cosseno.

        Returns:
            Lista de dicts com 'candidate', 'score' e 'details', ordenados
            por score final descendente.
        """
        results: List[dict[str, Any]] = []

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

    def _filter_by_cosine(
        self,
        candidates: List[str],
        p_candidates: List[str],
        cosine_scores: Any,
        min_cosine: float,
        top_n: int,
    ) -> List[dict[str, Any]]:
        """Filtra candidatos pelo limiar de cosseno e retorna os top-N.

        Args:
            candidates: Lista de textos originais dos candidatos.
            p_candidates: Lista de textos pré-processados dos candidatos.
            cosine_scores: Array de scores de cosseno para cada candidato.
            min_cosine: Limiar mínimo de cosseno.
            top_n: Número máximo de candidatos a retornar.

        Returns:
            Lista de dicts com 'candidate', 'p_candidate' e 'cos_score',
            ordenados por cosseno descendente e limitados a top_n.
        """
        scored: List[dict[str, Any]] = []
        for c_text, c_p_text, cos_score in zip(
            candidates, p_candidates, cosine_scores
        ):
            if cos_score >= min_cosine:
                scored.append(
                    {
                        "candidate": c_text,
                        "p_candidate": c_p_text,
                        "cos_score": float(cos_score),
                    }
                )
        scored.sort(key=lambda x: x["cos_score"], reverse=True)
        return scored[:top_n]

    def compare_batch(
        self,
        text: str,
        candidates: List[str],
        top_n: int = 50,
        min_cosine: float = 0.1,
        strategy: Literal["vectorized"] = "vectorized",
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
            strategy: Estratégia de comparação. Atualmente suporta apenas
                ``"vectorized"`` (padrão), que delega internamente para
                :meth:`compare_many_to_many`.

        Returns:
            Lista de dicionários, ordenados do maior score final para o menor,
            contendo o candidato original, o score e os detalhes da similaridade.

        Raises:
            ValueError: Se ``strategy`` não for um valor suportado.
        """
        if strategy != "vectorized":
            raise ValueError(
                f"Estratégia '{strategy}' não suportada. "
                "Use 'vectorized'."
            )

        results = self.compare_many_to_many(
            queries=[text],
            candidates=candidates,
            top_n=top_n,
            min_cosine=min_cosine,
        )
        return results[0] if results else []

    def compare_many_to_many(
        self,
        queries: List[str],
        candidates: List[str],
        top_n: int = 50,
        min_cosine: float = 0.1,
    ) -> List[List[dict[str, Any]]]:
        """Compara múltiplas queries contra uma lista de candidatos.

        Otimiza cenários multi-query pré-computando a matriz TF-IDF dos
        candidatos **uma única vez** e reutilizando-a para cada query.
        Em cenários como 1.500 queries × 100k candidatos, isso elimina
        o recálculo redundante do ``fit_transform`` a cada chamada.

        O pipeline completo é:

        1. Pré-processamento em lote dos candidatos (com cache).
        2. ``fit_transform`` do TF-IDF nos candidatos (uma vez).
        3. Para cada query: ``transform`` + ``cosine_similarity``.
        4. Filtragem por ``min_cosine`` e ``top_n``.
        5. Scoring híbrido (entity, edit, phonetic) nos top-N.

        Args:
            queries: Lista de textos de busca.
            candidates: Lista de textos candidatos.
            top_n: Número máximo de candidatos por query na etapa final.
            min_cosine: Limiar mínimo de cosseno para descartar ruidosos.

        Returns:
            Lista de listas de dicionários — uma lista de resultados para
            cada query, ordenados do maior score final para o menor.
            Cada dicionário contém 'candidate' (str), 'score' (float)
            e 'details' (dict).
        """
        if not queries:
            return []
        if not candidates:
            return [[] for _ in queries]

        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import (
            cosine_similarity as sklearn_cosine_similarity,
        )

        # 1. Pré-processamento em lote dos candidatos (reutiliza cache)
        p_candidates = self._process_batch(candidates)

        # 2. Ajuste (fit) do vectorizer nos candidatos — UMA ÚNICA VEZ
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        try:
            cand_matrix = vectorizer.fit_transform(p_candidates)
        except ValueError:
            # Vocabulário vazio — retorna listas vazias para todas as queries
            return [[] for _ in queries]

        # 3. Para cada query: transform + cosine + scoring
        all_results: List[List[dict[str, Any]]] = []

        for query in queries:
            p_query = self._process(query)

            try:
                query_vec = vectorizer.transform([p_query])
                cosine_scores = sklearn_cosine_similarity(
                    query_vec, cand_matrix
                )[0]
            except ValueError:
                all_results.append([])
                continue

            # 4. Filtrar pelo cosseno e pegar top-N
            top_candidates = self._filter_by_cosine(
                candidates, p_candidates, cosine_scores, min_cosine, top_n
            )

            # 5. Scoring híbrido completo
            results = self._score_candidates(p_query, top_candidates)
            all_results.append(results)

        return all_results
