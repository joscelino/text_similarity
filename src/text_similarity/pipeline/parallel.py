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
        List[str],          # chunk_queries
        List[str],          # candidates (originais)
        List[str],          # p_candidates (pré-processados)
        Any,                # cand_matrix (scipy sparse — pickle-safe)
        Any,                # vectorizer (TfidfVectorizer — pickle-safe)
        str,                # mode
        Optional[List[str]],  # entities
        Dict[str, float],   # algorithm_weights
        int,                # top_n
        float,              # min_cosine
    ],
) -> List[List[Dict[str, Any]]]:
    """Worker function executada em cada processo filho.

    Recria o Comparator localmente e processa um chunk de queries
    contra a matriz TF-IDF pré-computada dos candidatos.

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
    ) = args

    from sklearn.metrics.pairwise import (
        cosine_similarity as sklearn_cosine_similarity,
    )

    from text_similarity.api import Comparator

    # Recria o Comparator local (cada processo tem sua própria instância)
    comp = Comparator(mode=mode, entities=entities, use_cache=True)

    # Sobrescreve os pesos do algoritmo para manter consistência
    if hasattr(comp.algorithm, "weights"):
        comp.algorithm.weights = algorithm_weights  # type: ignore[union-attr]

    chunk_results: List[List[Dict[str, Any]]] = []

    for query in chunk_queries:
        p_query = comp._process(query)

        try:
            query_vec = vectorizer.transform([p_query])
            cosine_scores = sklearn_cosine_similarity(
                query_vec, cand_matrix
            )[0]
        except ValueError:
            chunk_results.append([])
            continue

        # Filtrar pelo cosseno e pegar top-N
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
) -> List[List[Dict[str, Any]]]:
    """Orquestra a execução paralela de queries via ProcessPoolExecutor.

    Particiona as queries em chunks e distribui entre N processos.
    Cada processo recria internamente o ``Comparator`` e processa
    seu chunk de queries contra a matriz TF-IDF pré-computada.

    Args:
        queries: Lista completa de queries.
        candidates: Textos originais dos candidatos.
        p_candidates: Textos pré-processados dos candidatos.
        cand_matrix: Matriz TF-IDF esparsa dos candidatos (pickle-safe).
        vectorizer: TfidfVectorizer já ajustado (pickle-safe).
        mode: Modo do Comparator ('basic' ou 'smart').
        entities: Lista de entidades ativas (ou None).
        algorithm_weights: Pesos do algoritmo híbrido.
        top_n: Número máximo de candidatos por query.
        min_cosine: Limiar mínimo de cosseno.
        n_workers: Número de processos. Se None, usa ``os.cpu_count()``.

    Returns:
        Lista de listas de resultados — uma para cada query,
        na mesma ordem das queries de entrada.
    """
    if n_workers is None:
        n_workers = os.cpu_count() or 1

    # Garantir que não criamos mais workers do que queries
    n_workers = min(n_workers, len(queries))

    # Caso degenerado: apenas 1 worker — sem overhead de multiprocessing
    if n_workers <= 1:
        return _worker_process_queries((
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
        ))

    # Particionar queries em chunks
    chunk_size = math.ceil(len(queries) / n_workers)
    chunks = [
        queries[i: i + chunk_size]
        for i in range(0, len(queries), chunk_size)
    ]

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
        )
        for chunk in chunks
    ]

    # Executar em paralelo
    all_results: List[List[Dict[str, Any]]] = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for chunk_results in executor.map(
            _worker_process_queries, worker_args
        ):
            all_results.extend(chunk_results)

    return all_results
