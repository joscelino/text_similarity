"""Módulo de paralelismo para pré-processamento em lote de candidatos.

Distribui o pré-processamento de textos entre múltiplos processos
usando ``concurrent.futures.ProcessPoolExecutor``.
Segue o mesmo pattern de ``parallel.py`` para compatibilidade com Windows (spawn).
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor
from typing import List, Optional


def _preprocess_worker(
    args: tuple,
) -> List[str]:
    """Worker top-level que recria pipeline local e processa chunk.

    Deve ser top-level function para ser pickle-safe no Windows (spawn).

    Args:
        args: Tupla (chunk_texts, mode, entity_names).

    Returns:
        Lista de textos pré-processados.
    """
    chunk_texts, mode, entity_names = args

    from text_similarity.api import Comparator

    comp = Comparator(mode=mode, entities=entity_names, use_cache=False)
    return [comp._process(text) for text in chunk_texts]


def run_parallel_preprocess(
    texts: List[str],
    mode: str,
    entity_names: Optional[List[str]],
    n_workers: Optional[int] = None,
    threshold: int = 1000,
) -> List[str]:
    """Distribui pré-processamento em paralelo se len(texts) > threshold.

    Args:
        texts: Lista de textos para pré-processar.
        mode: Modo do Comparator ('basic' ou 'smart').
        entity_names: Lista de entidades ativas (ou None).
        n_workers: Número de processos. Se None, usa ``os.cpu_count()``.
        threshold: Mínimo de textos para ativar paralelismo.

    Returns:
        Lista de textos pré-processados, na mesma ordem da entrada.
    """
    if len(texts) <= threshold:
        from text_similarity.api import Comparator

        comp = Comparator(mode=mode, entities=entity_names, use_cache=False)
        return [comp._process(text) for text in texts]

    if n_workers is None:
        n_workers = os.cpu_count() or 1

    # Não criar mais workers do que necessário
    n_workers = min(n_workers, max(1, len(texts) // threshold))

    # Particionar em chunks
    chunk_size = math.ceil(len(texts) / n_workers)
    chunks = [texts[i : i + chunk_size] for i in range(0, len(texts), chunk_size)]

    worker_args = [(chunk, mode, entity_names) for chunk in chunks]

    all_processed: List[str] = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for chunk_result in executor.map(_preprocess_worker, worker_args):
            all_processed.extend(chunk_result)

    return all_processed
