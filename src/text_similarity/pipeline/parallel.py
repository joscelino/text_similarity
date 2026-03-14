"""Módulo de paralelismo para comparação multi-query via multiprocessing.

Implementa a distribuição de queries entre múltiplos processos utilizando
``concurrent.futures.ProcessPoolExecutor``. Cada worker recria seu próprio
``Comparator`` internamente para evitar problemas de serialização.
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Tuple


def _worker_process_queries(
    args: Tuple[
        List[str],  # chunk_queries
        List[str],  # candidates (originais)
        List[str],  # p_candidates (pré-processados)
        Any,  # cand_matrix (scipy sparse) ou None
        Any,  # vectorizer (TfidfVectorizer) ou None
        str,  # mode
        Optional[List[str]],  # entities
        Dict[str, float],  # algorithm_weights
        int,  # top_n
        float,  # min_cosine
        str,  # fusion_strategy
        int,  # rrf_k
        Optional[Dict[str, float]],  # rrf_weights
        bool,  # preprocess
        str,  # indexing_strategy
        Any,  # bm25_index (BM25Index) ou None
        Any,  # dense_index (DenseIndex) ou None
        str,  # dense_model_name
    ],
) -> List[List[Dict[str, Any]]]:
    """Worker function executada em cada processo filho.

    Recria o Comparator localmente e processa um chunk de queries
    contra o índice pré-computado (TF-IDF, BM25 ou Dense).

    Args:
        args: Tupla com todos os dados necessários para o worker.

    Returns:
        Lista de resultados para cada query no chunk.
    """
    (
        chunk_queries,
        candidates,
        p_candidates,
        cand_matrix,
        vectorizer,
        mode,
        entities,
        algorithm_weights,
        top_n,
        min_cosine,
        fusion_strategy,
        rrf_k,
        rrf_weights,
        preprocess,
        indexing_strategy,
        bm25_index,
        dense_index,
        dense_model_name,
    ) = args

    from text_similarity.api import Comparator

    # Checa se o "semantic" estava habilitado na classe mãe
    use_embeddings = algorithm_weights.get("semantic", 0.0) > 0.0

    # Recria o Comparator local (cada processo tem sua própria instância)
    comp = Comparator(
        mode=mode,
        entities=entities,
        use_cache=True,
        use_embeddings=use_embeddings,
        fusion_strategy=fusion_strategy,
        rrf_k=rrf_k,
        rrf_weights=rrf_weights,
    )

    # Sobrescreve os pesos do algoritmo para manter consistência
    if hasattr(comp.algorithm, "weights"):
        comp.algorithm.weights = algorithm_weights  # type: ignore[union-attr]

    chunk_results: List[List[Dict[str, Any]]] = []

    for query in chunk_queries:
        p_query = comp._process(query, preprocess=preprocess)

        try:
            if indexing_strategy == "dense":
                cosine_scores = dense_index.get_scores_normalized(p_query)
            elif indexing_strategy == "bm25":
                cosine_scores = bm25_index.get_scores_normalized(p_query)
            else:
                from sklearn.metrics.pairwise import (
                    cosine_similarity as sklearn_cosine_similarity,
                )

                query_vec = vectorizer.transform([p_query])
                cosine_scores = sklearn_cosine_similarity(query_vec, cand_matrix)[0]
        except ValueError:
            chunk_results.append([])
            continue

        # Filtrar pelo score e pegar top-N
        top_candidates = comp._filter_by_cosine(
            candidates, p_candidates, cosine_scores, min_cosine, top_n
        )

        # Scoring híbrido completo
        results = comp._score_candidates(p_query, top_candidates)
        chunk_results.append(results)

    return chunk_results


def run_parallel_queries(
    queries: List[str],
    candidates: List[str],
    p_candidates: List[str],
    cand_matrix: Any,
    vectorizer: Any,
    mode: str,
    entities: Optional[List[str]],
    algorithm_weights: Dict[str, float],
    top_n: int,
    min_cosine: float,
    n_workers: Optional[int] = None,
    fusion_strategy: str = "linear",
    rrf_k: int = 60,
    rrf_weights: Optional[Dict[str, float]] = None,
    preprocess: bool = True,
    indexing_strategy: str = "tfidf",
    bm25_index: Any = None,
    dense_index: Any = None,
    dense_model_name: str = ("paraphrase-multilingual-MiniLM-L12-v2"),
) -> List[List[Dict[str, Any]]]:
    """Orquestra a execução paralela via ProcessPoolExecutor.

    Particiona as queries em chunks e distribui entre N processos.
    Cada processo recria internamente o ``Comparator`` e processa
    seu chunk contra o índice pré-computado (TF-IDF, BM25 ou Dense).

    Args:
        queries: Lista completa de queries.
        candidates: Textos originais dos candidatos.
        p_candidates: Textos pré-processados dos candidatos.
        cand_matrix: Matriz TF-IDF esparsa (pickle-safe).
            None quando não é ``"tfidf"``.
        vectorizer: TfidfVectorizer já ajustado (pickle-safe).
            None quando não é ``"tfidf"``.
        mode: Modo do Comparator ('basic' ou 'smart').
        entities: Lista de entidades ativas (ou None).
        algorithm_weights: Pesos do algoritmo híbrido.
        top_n: Número máximo de candidatos por query.
        min_cosine: Limiar mínimo de cosseno.
        n_workers: Número de processos. Se None, usa ``cpu_count()``.
        fusion_strategy: Estratégia de fusão.
        rrf_k: Constante de suavização do RRF.
        rrf_weights: Pesos por algoritmo para o RRF.
        preprocess: Se False, bypassa o pipeline nos workers.
        indexing_strategy: ``"tfidf"``, ``"bm25"`` ou ``"dense"``.
        bm25_index: ``BM25Index`` já ajustada (pickle-safe).
        dense_index: ``DenseIndex`` já ajustado (pickle-safe).
        dense_model_name: Nome do modelo sentence-transformers.

    Returns:
        Lista de listas de resultados — uma para cada query,
        na mesma ordem das queries de entrada.
    """
    if n_workers is None:
        n_workers = os.cpu_count() or 1

    # Garantir que não criamos mais workers do que queries
    n_workers = min(n_workers, len(queries))

    # Caso degenerado: apenas 1 worker — sem overhead
    if n_workers <= 1:
        return _worker_process_queries(
            (
                queries,
                candidates,
                p_candidates,
                cand_matrix,
                vectorizer,
                mode,
                entities,
                algorithm_weights,
                top_n,
                min_cosine,
                fusion_strategy,
                rrf_k,
                rrf_weights,
                preprocess,
                indexing_strategy,
                bm25_index,
                dense_index,
                dense_model_name,
            )
        )

    # Particionar queries em chunks
    chunk_size = math.ceil(len(queries) / n_workers)
    chunks = [queries[i : i + chunk_size] for i in range(0, len(queries), chunk_size)]

    # Montar argumentos para cada worker
    worker_args = [
        (
            chunk,
            candidates,
            p_candidates,
            cand_matrix,
            vectorizer,
            mode,
            entities,
            algorithm_weights,
            top_n,
            min_cosine,
            fusion_strategy,
            rrf_k,
            rrf_weights,
            preprocess,
            indexing_strategy,
            bm25_index,
            dense_index,
            dense_model_name,
        )
        for chunk in chunks
    ]

    # Executar em paralelo
    all_results: List[List[Dict[str, Any]]] = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for chunk_results in executor.map(_worker_process_queries, worker_args):
            all_results.extend(chunk_results)

    return all_results
