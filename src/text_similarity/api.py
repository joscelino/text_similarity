"""Módulo principal da API pública da biblioteca.

A interface central para comparação é através da classe `Comparator`.
"""

from __future__ import annotations

from typing import Any, List, Literal

from text_similarity.core.base import SimilarityAlgorithm
from text_similarity.core.fusion import RRFusion
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
        fusion_strategy: Literal["linear", "rrf"] = "linear",
        rrf_k: int = 60,
        rrf_weights: dict[str, float] | None = None,
        indexing_strategy: Literal["tfidf", "bm25"] = "tfidf",
        bm25_k1: float = 1.2,
        bm25_b: float = 0.75,
        **kwargs: Any,
    ) -> None:
        """Inicializa a classe Comparator preparando o pipeline.

        Args:
            mode: Modo de operação ('basic' ou 'smart').
            entities: Lista de entidades para extrair no modo smart.
            use_cache: Se True, habilita o cache in-memory de textos processados.
            fusion_strategy: Estratégia de fusão para operações batch.
                ``"linear"`` (padrão) usa soma ponderada.
                ``"rrf"`` usa Reciprocal Rank Fusion baseada em posição.
            rrf_k: Parâmetro de suavização do RRF (padrão 60).
                Ignorado quando ``fusion_strategy="linear"``.
            rrf_weights: Pesos por algoritmo para o RRF (ex:
                ``{"cosine": 0.6, "semantic": 0.4}``). Se ``None``,
                todos os algoritmos contribuem igualmente.
                Ignorado quando ``fusion_strategy="linear"``.
            indexing_strategy: Estratégia de indexação para operações batch.
                ``"tfidf"`` (padrão) usa TF-IDF + cosseno.
                ``"bm25"`` usa Okapi BM25 (melhor para textos curtos).
            bm25_k1: Saturação de term frequency do BM25 (padrão 1.2).
                Ignorado quando ``indexing_strategy="tfidf"``.
            bm25_b: Normalização por comprimento do BM25 (padrão 0.75).
                Ignorado quando ``indexing_strategy="tfidf"``.
            **kwargs: Argumentos arbitrários reservados para extensões futuras.
        """
        self.mode = mode
        self.entities = entities
        self.use_cache = use_cache
        self.fusion_strategy = fusion_strategy
        self.rrf_k = rrf_k
        self.rrf_weights = rrf_weights
        self.indexing_strategy = indexing_strategy
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        self._rrf_fusion: RRFusion | None = (
            RRFusion(k=rrf_k, weights=rrf_weights) if fusion_strategy == "rrf" else None
        )

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
            weights = {"cosine": 0.45, "edit": 0.25, "phonetic": 0.20, "entity": 0.10}
            if kwargs.get("use_embeddings", False):
                # Se semântica for pedida no Smart, recalibramos
                weights = {
                    "cosine": 0.30,
                    "edit": 0.15,
                    "phonetic": 0.15,
                    "entity": 0.10,
                    "semantic": 0.30,
                }
            self.algorithm: SimilarityAlgorithm = HybridSimilarity(weights=weights)
        else:
            weights = {"cosine": 0.5, "edit": 0.5, "phonetic": 0.0}
            if kwargs.get("use_embeddings", False):
                weights = {"cosine": 0.3, "edit": 0.3, "phonetic": 0.0, "semantic": 0.4}
            self.algorithm = HybridSimilarity(weights=weights)

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
        use_embeddings: bool = False,
        fusion_strategy: Literal["linear", "rrf"] = "linear",
        rrf_k: int = 60,
        rrf_weights: dict[str, float] | None = None,
        indexing_strategy: Literal["tfidf", "bm25"] = "tfidf",
        bm25_k1: float = 1.2,
        bm25_b: float = 0.75,
    ) -> "Comparator":
        """Instancia um Comparator no modo inteligente.

        Ativa a extração de entidades, unifica tokens, analisa a fonética PT-BR
        e cruza resultados de múltiplos algoritmos.
        Se `use_embeddings=True`, ativa Similaridade Semântica baseada em
        vetores densos.

        Args:
            entities: Lista de entidades para extrair.
            use_cache: Habilita cache in-memory.
            use_embeddings: Ativa similaridade semântica.
            fusion_strategy: ``"linear"`` (soma ponderada) ou ``"rrf"``
                (Reciprocal Rank Fusion). Afeta apenas operações batch.
            rrf_k: Constante de suavização do RRF (padrão 60).
            rrf_weights: Pesos por algoritmo para o RRF (ex:
                ``{"cosine": 0.6, "semantic": 0.4}``). Se ``None``,
                todos os algoritmos contribuem igualmente.
            indexing_strategy: ``"tfidf"`` (padrão) ou ``"bm25"``
                (melhor para textos curtos como produtos).
            bm25_k1: Saturação de term frequency do BM25 (padrão 1.2).
            bm25_b: Normalização por comprimento do BM25 (padrão 0.75).
        """
        return cls(
            mode="smart",
            entities=entities,
            use_cache=use_cache,
            use_embeddings=use_embeddings,
            fusion_strategy=fusion_strategy,
            rrf_k=rrf_k,
            rrf_weights=rrf_weights,
            indexing_strategy=indexing_strategy,
            bm25_k1=bm25_k1,
            bm25_b=bm25_b,
        )

    def _process(self, text: str, preprocess: bool = True) -> str:
        """Pré-processa o texto pelo pipeline, com cache in-memory.

        Na primeira chamada para um texto, executa o pipeline completo e
        armazena o resultado. Chamadas subsequentes com o mesmo texto retornam
        o resultado cacheado diretamente.

        Quando ``preprocess=False``, retorna o texto sem alterações,
        ignorando pipeline e cache. Útil para dados já pré-processados.

        Args:
            text: Texto bruto de entrada.
            preprocess: Se False, bypassa o pipeline e retorna o texto
                diretamente.

        Returns:
            Texto pré-processado como bag of words.
        """
        if not preprocess:
            return text

        if self.cache is not None:
            key = self.cache.hash_text(text)
            if key in self._cache_store:
                return self._cache_store[key]
        else:
            key = None

        processed, _ = self.pipeline.process(text)

        if self.cache is not None and key is not None:
            self._cache_store[key] = processed

        return processed

    def clear_cache(self) -> None:
        """Limpa o cache in-memory e o cache em disco do Joblib."""
        self._cache_store.clear()
        if self.cache is not None:
            self.cache.clear()

    def unload_embeddings_model(self) -> None:
        """Libera o modelo semântico (sentence-transformers) da memória global.

        Útil para liberar RAM/VRAM após uma sessão de inferência intensa,
        ou antes de trocar para um modelo diferente. Após a chamada, o modelo
        será recarregado automaticamente na próxima comparação semântica.

        Não tem efeito se ``use_embeddings=False``.
        """
        semantic = self.algorithm.algorithms.get("semantic")
        if semantic is not None:
            semantic.unload()

    def preprocess_catalog(
        self,
        candidates: List[str],
        cache_path: str = "catalog_cache.pkl",
    ) -> List[str]:
        """Pré-processa candidatos e salva em disco para reuso.

        Na primeira execução, processa todos os candidatos e salva o
        resultado em ``cache_path``. Em execuções subsequentes com o
        mesmo catálogo, carrega direto do disco (~80% economia de tempo).

        A invalidação é automática via hash SHA-256 do conteúdo: se o
        catálogo mudar, o cache é reprocessado.

        Args:
            candidates: Lista de textos candidatos.
            cache_path: Caminho do arquivo de cache em disco.

        Returns:
            Lista de textos pré-processados.
        """
        if self.cache is not None:
            loaded = self.cache.load_catalog(candidates, cache_path)
            if loaded is not None:
                return loaded

        processed = self._process_batch(candidates, preprocess=True)

        if self.cache is not None:
            self.cache.save_catalog(candidates, processed, cache_path)

        return processed

    @property
    def _entity_names(self) -> "list[str] | None":
        """Retorna a lista de entidades configuradas."""
        return self.entities

    def _process_batch(self, texts: List[str], preprocess: bool = True) -> List[str]:
        """Pré-processa uma lista de textos em lote, reutilizando cache.

        Cada texto é processado pelo pipeline e armazenado no cache in-memory.
        Textos já processados anteriormente retornam direto do cache,
        evitando reprocessamento redundante.

        Para lotes grandes (>1000 textos), distribui o trabalho entre
        múltiplos processos via ``parallel_preprocess``.

        Quando ``preprocess=False``, retorna os textos sem alterações.

        Args:
            texts: Lista de textos brutos de entrada.
            preprocess: Se False, bypassa o pipeline e retorna os textos
                diretamente.

        Returns:
            Lista de textos pré-processados como bags of words.
        """
        if not preprocess:
            return list(texts)

        PARALLEL_THRESHOLD = 1000
        if len(texts) > PARALLEL_THRESHOLD:
            from .pipeline.parallel_preprocess import run_parallel_preprocess

            processed = run_parallel_preprocess(texts, self.mode, self._entity_names)
        else:
            processed = [self._process(text, preprocess=preprocess) for text in texts]

        # Atualizar cache in-memory com resultados do processamento paralelo
        if self.cache is not None:
            for text, p_text in zip(texts, processed):
                key = self.cache.hash_text(text)
                self._cache_store[key] = p_text

        return processed

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

    def _score_candidates_linear(
        self,
        p_text: str,
        top_candidates: List[dict[str, Any]],
    ) -> List[dict[str, Any]]:
        """Scoring via combinação linear ponderada (estratégia padrão)."""
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

                    for name in ["edit", "phonetic", "semantic"]:
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

        for algo_name in active_algos:
            ranking: List[dict[str, Any]] = []

            for cand in top_candidates:
                c_p_text = cand["p_candidate"]

                if algo_name == "cosine":
                    score = cand["cos_score"]
                else:
                    score = algs[algo_name].compare(p_text, c_p_text)

                ranking.append({"candidate": cand["candidate"], "score": score})

            ranking.sort(key=lambda x: x["score"], reverse=True)
            per_algo_rankings.append(ranking)

        assert self._rrf_fusion is not None
        return self._rrf_fusion.fuse(per_algo_rankings, active_algos)

    def compare(self, text1: str, text2: str, preprocess: bool = True) -> float:
        """Compara dois textos e retorna um valor global de similaridade.

        Args:
            text1: Primeiro texto para comparação.
            text2: Segundo texto para comparação.
            preprocess: Se False, bypassa o pipeline de pré-processamento.
                Útil quando os textos já foram limpos externamente.

        Returns:
            Score entre 0.0 (completamente diferentes) e 1.0 (idênticos).
        """
        p_text1 = self._process(text1, preprocess=preprocess)
        p_text2 = self._process(text2, preprocess=preprocess)
        return self.algorithm.compare(p_text1, p_text2)

    def explain(
        self, text1: str, text2: str, preprocess: bool = True
    ) -> dict[str, Any]:
        """Retorna as predições individuais de todos os algoritmos rodados no texto.

        Args:
            text1: Primeiro texto para comparação.
            text2: Segundo texto para comparação.
            preprocess: Se False, bypassa o pipeline de pré-processamento.

        Returns:
            Dicionário com 'score' (float) e 'details' (dict por algoritmo).
        """
        p_text1 = self._process(text1, preprocess=preprocess)
        p_text2 = self._process(text2, preprocess=preprocess)

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
        for c_text, c_p_text, cos_score in zip(candidates, p_candidates, cosine_scores):
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
        strategy: Literal["vectorized", "parallel"] = "vectorized",
        n_workers: int | None = None,
        preprocess: bool = True,
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
            strategy: Estratégia de comparação.
                ``"vectorized"`` (padrão) executa sequencialmente.
                ``"parallel"`` distribui queries entre múltiplos processos.
            n_workers: Número de processos para ``strategy="parallel"``.
                Se None, usa ``os.cpu_count()``. Ignorado para ``vectorized``.
            preprocess: Se False, bypassa o pipeline de pré-processamento.
                Útil quando os textos já foram limpos externamente.

        Returns:
            Lista de dicionários, ordenados do maior score final para o menor,
            contendo o candidato original, o score e os detalhes da similaridade.

        Raises:
            ValueError: Se ``strategy`` não for um valor suportado.
        """
        _valid_strategies = ("vectorized", "parallel")
        if strategy not in _valid_strategies:
            raise ValueError(
                f"Estratégia '{strategy}' não suportada. "
                f"Use uma das: {_valid_strategies}."
            )

        results = self.compare_many_to_many(
            queries=[text],
            candidates=candidates,
            top_n=top_n,
            min_cosine=min_cosine,
            strategy=strategy,
            n_workers=n_workers,
            preprocess=preprocess,
        )
        return results[0] if results else []

    def compare_many_to_many(
        self,
        queries: List[str],
        candidates: List[str],
        top_n: int = 50,
        min_cosine: float = 0.1,
        strategy: Literal["vectorized", "parallel"] = "vectorized",
        n_workers: int | None = None,
        preprocess: bool = True,
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

        Quando ``strategy="parallel"``, as etapas 3-5 são distribuídas
        entre múltiplos processos via ``ProcessPoolExecutor``.

        Args:
            queries: Lista de textos de busca.
            candidates: Lista de textos candidatos.
            top_n: Número máximo de candidatos por query na etapa final.
            min_cosine: Limiar mínimo de cosseno para descartar ruidosos.
            strategy: Estratégia de execução.
                ``"vectorized"`` (padrão) executa sequencialmente.
                ``"parallel"`` distribui queries entre múltiplos processos.
            n_workers: Número de processos para ``strategy="parallel"``.
                Se None, usa ``os.cpu_count()``. Ignorado para ``vectorized``.
            preprocess: Se False, bypassa o pipeline de pré-processamento.
                Útil quando os textos já foram limpos externamente.

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

        # 1. Pré-processamento em lote dos candidatos (reutiliza cache)
        p_candidates = self._process_batch(candidates, preprocess=preprocess)

        # 2. Construir índice de acordo com a estratégia de indexação
        vectorizer = None
        cand_matrix = None
        bm25_index = None

        if self.indexing_strategy == "bm25":
            from text_similarity.core.bm25 import BM25Index

            bm25_index = BM25Index(k1=self.bm25_k1, b=self.bm25_b)
            bm25_index.fit(p_candidates)
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer

            vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
            try:
                cand_matrix = vectorizer.fit_transform(p_candidates)
            except ValueError:
                return [[] for _ in queries]

        # 3. Estratégia de execução
        if strategy == "parallel":
            from text_similarity.pipeline.parallel import run_parallel_queries

            # Extrair pesos do algoritmo para serialização
            alg_weights: dict[str, float] = {}
            if hasattr(self.algorithm, "weights"):
                alg_weights = self.algorithm.weights  # type: ignore[union-attr]

            return run_parallel_queries(
                queries=queries,
                candidates=list(candidates),
                p_candidates=p_candidates,
                cand_matrix=cand_matrix,
                vectorizer=vectorizer,
                mode=self.mode,
                entities=self.entities,
                algorithm_weights=alg_weights,
                top_n=top_n,
                min_cosine=min_cosine,
                n_workers=n_workers,
                fusion_strategy=self.fusion_strategy,
                rrf_k=self.rrf_k,
                rrf_weights=self.rrf_weights,
                preprocess=preprocess,
                indexing_strategy=self.indexing_strategy,
                bm25_index=bm25_index,
            )

        # Estratégia sequencial (vectorized)
        all_results: List[List[dict[str, Any]]] = []

        for query in queries:
            p_query = self._process(query, preprocess=preprocess)

            try:
                if self.indexing_strategy == "bm25":
                    assert bm25_index is not None
                    cosine_scores = bm25_index.get_scores_normalized(p_query)
                else:
                    from sklearn.metrics.pairwise import (
                        cosine_similarity as sklearn_cosine_similarity,
                    )

                    assert vectorizer is not None
                    query_vec = vectorizer.transform([p_query])
                    cosine_scores = sklearn_cosine_similarity(query_vec, cand_matrix)[0]
            except ValueError:
                all_results.append([])
                continue

            # 4. Filtrar pelo score e pegar top-N
            top_candidates = self._filter_by_cosine(
                candidates, p_candidates, cosine_scores, min_cosine, top_n
            )

            # 5. Scoring híbrido completo
            results = self._score_candidates(p_query, top_candidates)
            all_results.append(results)

        return all_results

    async def compare_batch_async(
        self,
        text: str,
        candidates: List[str],
        top_n: int = 50,
        min_cosine: float = 0.1,
        n_workers: int | None = None,
        preprocess: bool = True,
    ) -> List[dict[str, Any]]:
        """Versão assíncrona de :meth:`compare_batch`.

        Offloads o trabalho CPU-bound para um ``ProcessPoolExecutor``
        via ``loop.run_in_executor()``, mantendo o event loop livre
        para atender outras requisições concorrentes.

        Ideal para integração com frameworks async como FastAPI,
        aiohttp e Starlette.

        Args:
            text: Texto principal para buscar.
            candidates: Lista de textos candidatos.
            top_n: Número máximo de candidatos filtrados para a etapa final.
            min_cosine: Limiar mínimo de cosseno para descartar ruidosos.
            n_workers: Número de processos. Se None, usa ``os.cpu_count()``.
            preprocess: Se False, bypassa o pipeline de pré-processamento.

        Returns:
            Lista de dicionários, ordenados do maior score final para o menor.
        """
        import asyncio
        import functools

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            functools.partial(
                self.compare_batch,
                text,
                candidates,
                top_n=top_n,
                min_cosine=min_cosine,
                strategy="parallel",
                n_workers=n_workers,
                preprocess=preprocess,
            ),
        )

    async def compare_many_to_many_async(
        self,
        queries: List[str],
        candidates: List[str],
        top_n: int = 50,
        min_cosine: float = 0.1,
        n_workers: int | None = None,
        preprocess: bool = True,
    ) -> List[List[dict[str, Any]]]:
        """Versão assíncrona de :meth:`compare_many_to_many`.

        Offloads o trabalho CPU-bound para um ``ProcessPoolExecutor``
        via ``loop.run_in_executor()``, mantendo o event loop livre.

        Args:
            queries: Lista de textos de busca.
            candidates: Lista de textos candidatos.
            top_n: Número máximo de candidatos por query na etapa final.
            min_cosine: Limiar mínimo de cosseno para descartar ruidosos.
            n_workers: Número de processos. Se None, usa ``os.cpu_count()``.
            preprocess: Se False, bypassa o pipeline de pré-processamento.

        Returns:
            Lista de listas de dicionários — uma lista de resultados para
            cada query, ordenados do maior score final para o menor.
        """
        import asyncio
        import functools

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            functools.partial(
                self.compare_many_to_many,
                queries,
                candidates,
                top_n=top_n,
                min_cosine=min_cosine,
                strategy="parallel",
                n_workers=n_workers,
                preprocess=preprocess,
            ),
        )

    # -----------------------------------------------------------------
    # Re-ranking de resultados de bancos vetoriais
    # -----------------------------------------------------------------

    def rerank_vector_results(
        self,
        query: str,
        vector_candidates: List[dict[str, Any]],
        preprocess_query: bool = True,
        preprocess_candidates: bool = False,
    ) -> List[dict[str, Any]]:
        """Re-rankeia resultados de um banco vetorial usando HybridSimilarity.

        Recebe candidatos já retornados por um banco vetorial (Pinecone,
        Qdrant, Milvus, PGVector, Elasticsearch, etc.) e re-ordena
        aplicando os algoritmos linguísticos do ``HybridSimilarity``
        (edição, fonética, entidades), usando o score vetorial original
        como ``cos_score``.

        Pula completamente o TF-IDF local e o filtro por cosseno — o
        banco vetorial já fez essa etapa.

        Args:
            query: Texto de busca do usuário.
            vector_candidates: Lista de dicionários com os resultados do
                banco vetorial. Cada dict deve conter ao menos:

                - ``"text"`` (str): Texto do candidato.
                - ``"score"`` (float): Score de similaridade do banco
                  (0.0 a 1.0).
                - ``"id"`` (str, opcional): Identificador do documento.

                Exemplo::

                    [
                        {"id": "doc1", "text": "Galaxy S22", "score": 0.82},
                        {"id": "doc2", "text": "iPhone 15", "score": 0.71},
                    ]

            preprocess_query: Se True, aplica o pipeline na query.
                Padrão True pois a query geralmente é texto bruto do
                usuário.
            preprocess_candidates: Se True, aplica o pipeline nos textos
                dos candidatos. Padrão False pois textos vindos de
                bancos vetoriais geralmente já estão normalizados.

        Returns:
            Lista de dicionários ordenados por score final descendente,
            contendo ``"id"`` (se presente no input), ``"candidate"``,
            ``"score"`` (final), ``"vector_score"`` (original do banco)
            e ``"details"`` (dict por algoritmo).

        Raises:
            ValueError: Se algum candidato não tiver ``"text"`` ou
                ``"score"``.
        """
        if not vector_candidates:
            return []

        # Validação do formato de entrada
        for i, cand in enumerate(vector_candidates):
            if "text" not in cand:
                raise ValueError(f"Candidato na posição {i} não possui o campo 'text'.")
            if "score" not in cand:
                raise ValueError(
                    f"Candidato na posição {i} não possui o campo 'score'."
                )

        # 1. Pré-processar query e textos dos candidatos
        p_query = self._process(query, preprocess=preprocess_query)
        cand_texts = [c["text"] for c in vector_candidates]
        p_texts = self._process_batch(cand_texts, preprocess=preprocess_candidates)

        # 2. Montar top_candidates no formato esperado por _score_candidates
        #    O score vetorial do banco é mapeado como cos_score
        top_candidates: List[dict[str, Any]] = [
            {
                "candidate": cand["text"],
                "p_candidate": p_text,
                "cos_score": float(cand["score"]),
            }
            for cand, p_text in zip(vector_candidates, p_texts)
        ]

        # 3. Scoring híbrido (reutiliza linear ou RRF conforme configurado)
        scored = self._score_candidates(p_query, top_candidates)

        # 4. Enriquecer com id e vector_score originais
        # Mapear candidate text → dados originais para lookup
        original_map: dict[str, dict[str, Any]] = {
            c["text"]: c for c in vector_candidates
        }

        enriched: List[dict[str, Any]] = []
        for result in scored:
            original = original_map.get(result["candidate"], {})
            entry: dict[str, Any] = {}

            if "id" in original:
                entry["id"] = original["id"]

            entry["candidate"] = result["candidate"]
            entry["score"] = result["score"]
            entry["vector_score"] = original.get("score", 0.0)
            entry["details"] = result["details"]

            enriched.append(entry)

        return enriched
