"""Implementação BM25 (Okapi BM25) para ranking de candidatos.

Índice BM25 otimizado para textos curtos em PT-BR (produtos, modelos,
descrições). Implementação pura sem dependências externas, pickle-safe
para compatibilidade com ``ProcessPoolExecutor`` no Windows (spawn).
"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np


class BM25Index:
    """Índice BM25 (Okapi BM25) para ranking de candidatos.

    Implementação pura sem dependências externas, otimizada para
    textos curtos (produtos, modelos, descrições).

    Args:
        k1: Parâmetro de saturação de term frequency. Valores maiores
            tornam o score mais sensível à frequência do termo.
            Padrão 1.2 (Okapi original).
        b: Parâmetro de normalização por comprimento do documento.
            0.0 = sem normalização, 1.0 = normalização total.
            Padrão 0.75 (Okapi original).
    """

    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        """Inicializa configurações do algoritmo BM25."""
        self.k1 = k1
        self.b = b
        self._corpus_size: int = 0
        self._avgdl: float = 0.0
        self._doc_freqs: Dict[str, int] = {}
        self._doc_lens: List[int] = []
        self._term_freqs: List[Dict[str, int]] = []

    def fit(self, documents: List[str]) -> "BM25Index":
        """Indexa corpus de candidatos (textos já pré-processados).

        Cada documento é tokenizado por espaço (bag-of-words), consistente
        com a saída do pipeline de pré-processamento.

        Args:
            documents: Lista de textos pré-processados.

        Returns:
            Self para encadeamento.
        """
        self._corpus_size = len(documents)
        self._term_freqs = []
        self._doc_lens = []
        self._doc_freqs = {}

        for doc in documents:
            tokens = doc.split()
            self._doc_lens.append(len(tokens))
            tf: Dict[str, int] = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            self._term_freqs.append(tf)
            for term in tf:
                self._doc_freqs[term] = self._doc_freqs.get(term, 0) + 1

        total_len = sum(self._doc_lens)
        self._avgdl = total_len / max(self._corpus_size, 1)
        return self

    def get_scores(self, query: str) -> np.ndarray:
        """Calcula scores BM25 do query contra todo o corpus.

        Args:
            query: Texto da query (pré-processado, tokenizado por espaço).

        Returns:
            Array de scores (não-normalizado) com shape ``(n_candidates,)``.
        """
        query_tokens = query.split()
        scores = np.zeros(self._corpus_size, dtype=np.float64)

        for token in query_tokens:
            if token not in self._doc_freqs:
                continue
            df = self._doc_freqs[token]
            idf = math.log((self._corpus_size - df + 0.5) / (df + 0.5) + 1.0)
            for i, tf_dict in enumerate(self._term_freqs):
                if token not in tf_dict:
                    continue
                tf = tf_dict[token]
                dl = self._doc_lens[i]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * dl / max(self._avgdl, 1e-10)
                )
                scores[i] += idf * numerator / denominator

        return scores

    def get_scores_normalized(self, query: str) -> np.ndarray:
        """Scores normalizados para ``[0, 1]`` via min-max scaling.

        Args:
            query: Texto da query (pré-processado, tokenizado por espaço).

        Returns:
            Array de scores normalizados com shape ``(n_candidates,)``.
        """
        scores = self.get_scores(query)
        max_score = scores.max()
        if max_score > 0:
            return scores / max_score
        return scores
