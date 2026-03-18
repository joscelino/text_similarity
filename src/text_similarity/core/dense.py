"""Índice denso baseado em sentence-transformers para filtragem semântica.

Índice de embeddings densos para candidatos, otimizado para recall
semântico em PT-BR. Utiliza similaridade de cosseno entre vetores
densos como filtro de primeiro estágio em operações batch.

Compatível com ``ProcessPoolExecutor`` (pickle-safe): armazena apenas
arrays numpy, sem referências ao modelo de embedding.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from pathlib import Path
from typing import Any, List, Union

import numpy as np

logger = logging.getLogger(__name__)

# Cache global lazy — mesmo padrão de semantic.py.
# O modelo NÃO é armazenado na instância (não é pickle-safe).
# Workers recriam o modelo localmente via este cache global.
_DENSE_MODEL: Any = None
_DENSE_MODEL_NAME: str | None = None
_DENSE_LOCK = threading.Lock()

_INDEX_VERSION = "1.0"


def _ensure_dense_model(
    model_name: str,
    device: str | None = None,
) -> Any:
    """Carrega o modelo de embedding globalmente (lazy, thread-safe).

    Utiliza Double-Checked Locking para evitar contenção após
    a primeira carga.

    Args:
        model_name: Nome/path do modelo no HuggingFace.
        device: Dispositivo ('cpu', 'cuda', etc). Se None, auto.

    Returns:
        Instância de ``SentenceTransformer`` carregada.
    """
    global _DENSE_MODEL, _DENSE_MODEL_NAME

    if _DENSE_MODEL is not None and _DENSE_MODEL_NAME == model_name:
        return _DENSE_MODEL

    with _DENSE_LOCK:
        if _DENSE_MODEL is not None and _DENSE_MODEL_NAME == model_name:
            return _DENSE_MODEL

        logger.info(
            "Carregando modelo denso para indexação: %s",
            model_name,
        )
        try:
            from sentence_transformers import SentenceTransformer

            kwargs: dict[str, Any] = {}
            if device:
                kwargs["device"] = device

            _DENSE_MODEL = SentenceTransformer(model_name, **kwargs)
            _DENSE_MODEL_NAME = model_name
            return _DENSE_MODEL

        except ImportError as e:
            raise ImportError(
                "DenseIndex requer sentence-transformers. "
                "Instale com: pip install text-similarity-br[semantic]  "
                "ou: uv add text-similarity-br[semantic]"
            ) from e


class DenseIndex:
    """Índice de embeddings densos para filtragem semântica.

    Utiliza ``sentence-transformers`` para codificar documentos
    em vetores densos e computar similaridade de cosseno como
    filtro de primeiro estágio em operações batch.

    Pickle-safe: armazena apenas ``np.ndarray`` e metadados
    escalares. O modelo de embedding é carregado globalmente por
    processo via cache lazy.

    Args:
        model_name: Nome/path do modelo no HuggingFace.
            Padrão: modelo multilíngue leve com suporte a PT-BR.
        device: Dispositivo ('cpu', 'cuda', etc). Se None, auto.
        precision: Precisão dos embeddings armazenados.
            ``"float32"`` (padrão) — qualidade máxima, ~4 bytes/dim.
            ``"int8"`` — quantização escalar, ~1 byte/dim (~75% menor).
            ``"binary"`` — bits empacotados, ~0.125 byte/dim (~97% menor).
    """

    def __init__(
        self,
        model_name: str = ("paraphrase-multilingual-MiniLM-L12-v2"),
        device: str | None = None,
        precision: str = "float32",
    ) -> None:
        """Configura identificação do modelo de embedding."""
        if precision not in ("float32", "int8", "binary"):
            raise ValueError(
                f"precision deve ser 'float32', 'int8' ou 'binary', "
                f"recebido: {precision!r}"
            )
        self.model_name = model_name
        self.device = device
        self.precision = precision
        self._embeddings: np.ndarray | None = None
        self.n_documents: int = 0
        self.embedding_dim: int = 0

    def fit(self, documents: List[str]) -> "DenseIndex":
        """Codifica todos os documentos e armazena os embeddings.

        Args:
            documents: Lista de textos (pré-processados ou raw).

        Returns:
            Self para encadeamento.
        """
        model = _ensure_dense_model(self.model_name, self.device)
        emb_f32: np.ndarray = model.encode(
            documents,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

        self.n_documents = len(documents)
        self.embedding_dim = emb_f32.shape[1] if emb_f32.ndim == 2 else 0

        if self.precision == "float32":
            # Normalizar vetores para cosseno via dot product
            norms = np.linalg.norm(emb_f32, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            self._embeddings = emb_f32 / norms
        elif self.precision == "int8":
            try:
                from sentence_transformers.quantization import quantize_embeddings

                self._embeddings = quantize_embeddings(emb_f32, precision="int8")
            except ImportError:
                # Fallback: quantização manual se API não disponível
                self._embeddings = np.clip(np.round(emb_f32 * 127.0), -128, 127).astype(
                    np.int8
                )
        else:  # binary
            try:
                from sentence_transformers.quantization import quantize_embeddings

                self._embeddings = quantize_embeddings(emb_f32, precision="binary")
            except ImportError:
                # Fallback: binarização manual via sinal
                packed = np.packbits((emb_f32 > 0).astype(np.uint8), axis=1)
                self._embeddings = packed

        return self

    def get_scores_normalized(self, query: str) -> np.ndarray:
        """Similaridade da query contra o corpus, normalizada em ``[0, 1]``.

        Args:
            query: Texto da query.

        Returns:
            Array de scores com shape ``(n_candidates,)``.
        """
        if self._embeddings is None:
            return np.array([], dtype=np.float32)

        model = _ensure_dense_model(self.model_name, self.device)
        q_emb: np.ndarray = model.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

        if self.precision == "float32":
            # Normalizar query e usar dot product
            q_norm = np.linalg.norm(q_emb)
            if q_norm > 1e-10:
                q_emb = q_emb / q_norm
            scores = self._embeddings @ q_emb.flatten()
            return np.clip(scores, 0.0, 1.0)

        elif self.precision == "int8":
            # Dequantizar e computar cosseno
            doc_f32 = self._embeddings.astype(np.float32)
            norms = np.linalg.norm(doc_f32, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            doc_norm = doc_f32 / norms
            q_norm = np.linalg.norm(q_emb)
            if q_norm > 1e-10:
                q_emb = q_emb / q_norm
            scores = doc_norm @ q_emb.flatten()
            return np.clip(scores.astype(np.float32), 0.0, 1.0)

        else:  # binary
            # Similaridade Hamming via XOR + popcount
            # quantize_embeddings retorna int8 com bits empacotados
            q_bits = np.packbits((q_emb.flatten() > 0).astype(np.uint8))
            emb_uint8 = self._embeddings.view(np.uint8)
            n_bits = emb_uint8.shape[1] * 8
            xor = np.bitwise_xor(emb_uint8, q_bits)
            hamming = np.unpackbits(xor, axis=1).sum(axis=1).astype(np.float32)
            similarity = 1.0 - hamming / max(n_bits, 1)
            return np.clip(similarity, 0.0, 1.0)

    def save(self, path: Union[str, Path]) -> None:
        """Serializa o índice denso para disco via joblib.

        Args:
            path: Caminho do arquivo de saída (ex: ``"idx.pkl"``).
        """
        import joblib

        embeddings_bytes = (
            self._embeddings.tobytes() if self._embeddings is not None else b""
        )
        integrity_hash = hashlib.sha256(embeddings_bytes).hexdigest()

        payload = {
            "version": _INDEX_VERSION,
            "type": "DenseIndex",
            "data": {
                "model_name": self.model_name,
                "device": self.device,
                "precision": self.precision,
                "n_documents": self.n_documents,
                "embedding_dim": self.embedding_dim,
                "embeddings": self._embeddings,
            },
            "integrity_hash": integrity_hash,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "DenseIndex":
        """Carrega e valida um índice denso do disco.

        Args:
            path: Caminho do arquivo gerado por ``save()``.

        Returns:
            Instância ``DenseIndex`` pronta para uso.

        Raises:
            ValueError: Se a versão, tipo ou integridade não bater.
        """
        import joblib

        payload = joblib.load(path)
        if payload.get("version") != _INDEX_VERSION:
            raise ValueError(
                f"Versão incompatível: esperada {_INDEX_VERSION!r}, "
                f"encontrada {payload.get('version')!r}"
            )
        if payload.get("type") != "DenseIndex":
            raise ValueError(
                f"Tipo inválido: esperado 'DenseIndex', "
                f"encontrado {payload.get('type')!r}"
            )
        data = payload["data"]
        embeddings: np.ndarray | None = data.get("embeddings")
        embeddings_bytes = embeddings.tobytes() if embeddings is not None else b""
        expected_hash = hashlib.sha256(embeddings_bytes).hexdigest()
        if payload.get("integrity_hash") != expected_hash:
            raise ValueError(
                "Arquivo de índice corrompido (hash de integridade inválido)."
            )

        idx = cls(
            model_name=data["model_name"],
            device=data.get("device"),
            precision=data.get("precision", "float32"),
        )
        idx._embeddings = embeddings
        idx.n_documents = data.get("n_documents", 0)
        idx.embedding_dim = data.get("embedding_dim", 0)
        return idx
