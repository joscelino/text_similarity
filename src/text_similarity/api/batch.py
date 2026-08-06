"""Mixin de operações batch e reranking do :class:`Comparator`.

Isola do módulo principal (SEC-STD-001):

- :meth:`BatchMixin._filter_by_cosine` — utilidade comum.
- :meth:`BatchMixin.compare_batch` — wrapper single-query.
- :meth:`BatchMixin.compare_many_to_many` — pipeline batch completo;
  usa :meth:`_get_or_build_index` (SEC-DUP-003) para reutilizar índices.
- :meth:`BatchMixin.compare_batch_async` / :meth:`compare_many_to_many_async`
  — offloading via ``ProcessPoolExecutor``.
- :meth:`BatchMixin.rerank_vector_results` — re-ranking de resultados
  vindos de bancos vetoriais externos.
"""

from __future__ import annotations

from typing import Any, List, Literal


class BatchMixin:
    """Mixin com operações batch e reranking do :class:`Comparator`."""

    # ------------------------------------------------------------------
    # Utilidade compartilhada
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Batch: single-query wrapper
    # ------------------------------------------------------------------
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

        Otimiza o processo construindo o índice de todos os candidatos uma
        única vez (TF-IDF, BM25 ou Dense, conforme ``indexing_strategy``
        configurado em :meth:`Comparator.smart`) e extraindo os candidatos
        que passam num limiar mínimo de cosseno para só então aplicar as
        similaridades mais custosas (fonética, distância de edição).

        Args:
            text: Texto principal para buscar.
            candidates: Lista de textos candidatos.
            top_n: Número máximo de candidatos filtrados para a etapa final.
            min_cosine: Limiar mínimo de cosseno para descartar ruidosos.
            strategy: ``"vectorized"`` (padrão) ou ``"parallel"``.
            n_workers: Número de processos para ``strategy="parallel"``.
            preprocess: Se False, bypassa o pipeline.

        Returns:
            Lista de dicionários ordenados por score descendente.

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

    # ------------------------------------------------------------------
    # Batch: many-to-many
    # ------------------------------------------------------------------
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

        Otimiza cenários multi-query pré-computando o índice dos candidatos
        **uma única vez** e reutilizando-o para cada query. Em cenários como
        1.500 queries × 100k candidatos, isso elimina o recálculo redundante.

        A estratégia de indexação (TF-IDF, BM25 ou Dense) vem do parâmetro
        ``indexing_strategy`` de :meth:`Comparator.smart`. Por padrão usa
        TF-IDF; para BM25/Dense a construção/reuso do índice é delegada a
        :meth:`IndexManagerMixin._get_or_build_index` — sem ``if/elif``
        duplicado (SEC-DUP-003).

        Args:
            queries: Lista de textos de busca.
            candidates: Lista de textos candidatos.
            top_n: Máximo de candidatos por query na etapa final.
            min_cosine: Limiar mínimo de cosseno.
            strategy: ``"vectorized"`` (padrão) ou ``"parallel"``.
            n_workers: Número de processos para ``strategy="parallel"``.
            preprocess: Se False, bypassa o pipeline.

        Returns:
            Lista de listas de dicionários — uma por query.
        """
        if not queries:
            return []
        if not candidates:
            return [[] for _ in queries]

        # 1. Pré-processamento em lote dos candidatos (reutiliza cache)
        # ``_process_batch`` vem do Comparator.
        p_candidates = self._process_batch(  # type: ignore[attr-defined]
            candidates, preprocess=preprocess
        )

        # 2. Construir/recuperar índice conforme a estratégia
        vectorizer = None
        cand_matrix = None
        bm25_index = None
        dense_index = None

        if self.indexing_strategy in ("bm25", "dense"):  # type: ignore[attr-defined]
            # SEC-DUP-003: um único ponto de entrada para BM25 e Dense.
            active = self._get_or_build_index(  # type: ignore[attr-defined]
                self.indexing_strategy,  # type: ignore[attr-defined]
                p_candidates,
            )
            if self.indexing_strategy == "bm25":  # type: ignore[attr-defined]
                bm25_index = active
            else:
                dense_index = active
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

            alg_weights: dict[str, float] = {}
            if hasattr(self.algorithm, "weights"):  # type: ignore[attr-defined]
                alg_weights = self.algorithm.weights  # type: ignore[attr-defined]

            return run_parallel_queries(
                queries=queries,
                candidates=list(candidates),
                p_candidates=p_candidates,
                cand_matrix=cand_matrix,
                vectorizer=vectorizer,
                mode=self.mode,  # type: ignore[attr-defined]
                entities=self.entities,  # type: ignore[attr-defined]
                algorithm_weights=alg_weights,
                top_n=top_n,
                min_cosine=min_cosine,
                n_workers=n_workers,
                fusion_strategy=self.fusion_strategy,  # type: ignore[attr-defined]
                rrf_k=self.rrf_k,  # type: ignore[attr-defined]
                rrf_weights=self.rrf_weights,  # type: ignore[attr-defined]
                preprocess=preprocess,
                indexing_strategy=self.indexing_strategy,  # type: ignore[attr-defined]
                bm25_index=bm25_index,
                dense_index=dense_index,
                dense_model_name=self.dense_model_name,  # type: ignore[attr-defined]
                dense_model_revision=self.dense_model_revision,  # type: ignore[attr-defined]
            )

        # Estratégia sequencial (vectorized)
        all_results: List[List[dict[str, Any]]] = []

        for query in queries:
            p_query = self._process(query, preprocess=preprocess)  # type: ignore[attr-defined]

            try:
                if self.indexing_strategy == "dense":  # type: ignore[attr-defined]
                    assert dense_index is not None
                    cosine_scores = dense_index.get_scores_normalized(p_query)
                elif self.indexing_strategy == "bm25":  # type: ignore[attr-defined]
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

            top_candidates = self._filter_by_cosine(
                candidates, p_candidates, cosine_scores, min_cosine, top_n
            )
            results = self._score_candidates(p_query, top_candidates)  # type: ignore[attr-defined]
            all_results.append(results)

        return all_results

    # ------------------------------------------------------------------
    # Wrappers assíncronos
    # ------------------------------------------------------------------
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
        via ``loop.run_in_executor()``. Ideal para integração com
        FastAPI, aiohttp e Starlette.
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
        """Versão assíncrona de :meth:`compare_many_to_many`."""
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

    # ------------------------------------------------------------------
    # Re-ranking de resultados de bancos vetoriais
    # ------------------------------------------------------------------
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
        aplicando os algoritmos linguísticos do :class:`HybridSimilarity`
        (edição, fonética, entidades), usando o score vetorial original
        como ``cos_score``.

        Pula o TF-IDF local e o filtro por cosseno — o banco vetorial já
        fez essa etapa.

        Args:
            query: Texto de busca do usuário.
            vector_candidates: Lista de dicts com ``"text"``, ``"score"``
                e opcionalmente ``"id"``.
            preprocess_query: Aplica o pipeline na query (padrão True).
            preprocess_candidates: Aplica o pipeline nos textos dos
                candidatos (padrão False — geralmente já normalizados).

        Returns:
            Lista ordenada por score final descendente, com ``candidate``,
            ``score``, ``vector_score``, ``details`` (e ``id`` se presente).

        Raises:
            ValueError: Se algum candidato não tiver ``"text"`` ou ``"score"``.
        """
        if not vector_candidates:
            return []

        for i, cand in enumerate(vector_candidates):
            if "text" not in cand:
                raise ValueError(f"Candidato na posição {i} não possui o campo 'text'.")
            if "score" not in cand:
                raise ValueError(
                    f"Candidato na posição {i} não possui o campo 'score'."
                )

        p_query = self._process(query, preprocess=preprocess_query)  # type: ignore[attr-defined]
        cand_texts = [c["text"] for c in vector_candidates]
        p_texts = self._process_batch(  # type: ignore[attr-defined]
            cand_texts, preprocess=preprocess_candidates
        )

        top_candidates: List[dict[str, Any]] = [
            {
                "candidate": cand["text"],
                "p_candidate": p_text,
                "cos_score": float(cand["score"]),
            }
            for cand, p_text in zip(vector_candidates, p_texts)
        ]

        scored = self._score_candidates(p_query, top_candidates)  # type: ignore[attr-defined]

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


__all__ = ["BatchMixin"]
