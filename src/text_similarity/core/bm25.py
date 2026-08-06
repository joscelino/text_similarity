"""Implementação BM25 (Okapi BM25) para ranking de candidatos.

Índice BM25 otimizado para textos curtos em PT-BR (produtos, modelos,
descrições). Implementação pura sem dependências externas, pickle-safe
para compatibilidade com ``ProcessPoolExecutor`` no Windows (spawn).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
from numpy.typing import NDArray

from text_similarity.core import _serialization

INDEX_FORMAT_VERSION = _serialization.INDEX_FORMAT_VERSION


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

    def get_scores(self, query: str) -> NDArray[np.float64]:
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

    def save(
        self,
        path: Union[str, "os.PathLike[str]"],
        *,
        hmac_key: Union[bytes, str, None] = None,
    ) -> None:
        """Serializa o índice BM25 no formato ``tsbr-index-v2``.

        Args:
            path: Caminho do arquivo de saída (ex: ``"idx.tsbr-index"``).
            hmac_key: Chave HMAC-SHA256 opcional (``bytes`` ou ``str``).
                Se não fornecida, tenta a variável de ambiente
                ``TSBR_HMAC_KEY``. Sem chave, o arquivo é gravado sem
                autenticação e um aviso é emitido.
        """
        data = {
            "k1": self.k1,
            "b": self.b,
            "corpus_size": self._corpus_size,
            "avgdl": self._avgdl,
            "doc_freqs": self._doc_freqs,
            "doc_lens": self._doc_lens,
            "term_freqs": self._term_freqs,
        }
        integrity_hash = hashlib.sha256(
            json.dumps(data, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        payload = {
            "type": "BM25Index",
            "version": INDEX_FORMAT_VERSION,
            "data": data,
            "integrity_hash": integrity_hash,
        }
        _serialization.dump_index(payload, Path(path), hmac_key=hmac_key)

    @classmethod
    def load(
        cls,
        path: Union[str, "os.PathLike[str]"],
        *,
        hmac_key: Union[bytes, str, None] = None,
        allow_legacy_pickle: bool = False,
    ) -> "BM25Index":
        """Carrega e valida um índice BM25 do disco.

        Suporta o formato seguro ``tsbr-index-v2`` (JSON + HMAC) e,
        mediante opt-in explícito, arquivos legados em pickle/joblib.

        Args:
            path: Caminho do arquivo gerado por ``save()``.
            hmac_key: Chave HMAC para validar autenticação. Se o arquivo
                foi gravado sem HMAC, o load prossegue com aviso.
            allow_legacy_pickle: Se ``True``, permite carregar arquivos
                antigos em pickle/joblib, emitindo ``SecurityWarning``.

        Returns:
            Instância ``BM25Index`` pronta para uso.

        Raises:
            ValueError: Se a versão, tipo, integridade ou HMAC não
                baterem; ou se um arquivo legado for encontrado sem
                ``allow_legacy_pickle=True``.
        """
        payload = _serialization.load_index(
            Path(path),
            hmac_key=hmac_key,
            allow_legacy_pickle=allow_legacy_pickle,
        )

        def _validate_integrity(data: Dict[str, Any], stored_hash: str) -> None:
            expected_hash = hashlib.sha256(
                json.dumps(data, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()
            if stored_hash != expected_hash:
                raise ValueError(
                    "Arquivo de índice corrompido (hash de integridade inválido)."
                )

        # Formato legado (pickle/joblib) retorna o dict original.
        if payload.get("version") == "1.0":
            data = payload["data"]
            _validate_integrity(data, payload.get("integrity_hash", ""))
        elif payload.get("version") == INDEX_FORMAT_VERSION:
            if payload.get("type") != "BM25Index":
                raise ValueError(
                    f"Tipo inválido: esperado 'BM25Index', "
                    f"encontrado {payload.get('type')!r}"
                )
            data = payload["data"]
            _validate_integrity(data, payload.get("integrity_hash", ""))
        else:
            raise ValueError(
                f"Versão incompatível: esperada "
                f"{INDEX_FORMAT_VERSION!r} ou '1.0' (legado), "
                f"encontrada {payload.get('version')!r}"
            )

        idx = cls(k1=data["k1"], b=data["b"])
        idx._corpus_size = data["corpus_size"]
        idx._avgdl = data["avgdl"]
        idx._doc_freqs = data["doc_freqs"]
        idx._doc_lens = data["doc_lens"]
        idx._term_freqs = data["term_freqs"]
        return idx

    def get_scores_normalized(self, query: str) -> NDArray[np.float32]:
        """Scores normalizados para ``[0, 1]`` via min-max scaling.

        Args:
            query: Texto da query (pré-processado, tokenizado por espaço).

        Returns:
            Array de scores normalizados com shape ``(n_candidates,)``.
        """
        scores = self.get_scores(query)
        max_score = scores.max()
        if max_score > 0:
            return np.asarray(scores / max_score, dtype=np.float32)
        return np.asarray(scores, dtype=np.float32)
