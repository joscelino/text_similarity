"""Índice denso baseado em sentence-transformers para filtragem semântica.

Índice de embeddings densos para candidatos, otimizado para recall
semântico em PT-BR. Utiliza similaridade de cosseno entre vetores
densos como filtro de primeiro estágio em operações batch.

Compatível com ``ProcessPoolExecutor`` (pickle-safe): armazena apenas
arrays numpy, sem referências ao modelo de embedding.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, List

import numpy as np

logger = logging.getLogger(__name__)

# Cache global lazy — mesmo padrão de semantic.py.
# O modelo NÃO é armazenado na instância (não é pickle-safe).
# Workers recriam o modelo localmente via este cache global.
_DENSE_MODEL: Any = None
_DENSE_MODEL_NAME: str | None = None
_DENSE_LOCK = threading.Lock()


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
                "O índice denso requer `sentence-transformers`. "
                "Instale com: pip install "
                "text_similarity[semantic]"
            ) from e


class DenseIndex:
    """Índice de embeddings densos para filtragem semântica.

    Utiliza ``sentence-transformers`` para codificar documentos
    em vetores densos e computar similaridade de cosseno como
    filtro de primeiro estágio em operações batch.

    Pickle-safe: armazena apenas ``np.ndarray`` (float32) e
    metadados escalares. O modelo de embedding é carregado
    globalmente por processo via cache lazy.

    Args:
        model_name: Nome/path do modelo no HuggingFace.
            Padrão: modelo multilíngue leve com suporte a PT-BR.
        device: Dispositivo ('cpu', 'cuda', etc). Se None, auto.
    """

    def __init__(
        self,
        model_name: str = ("paraphrase-multilingual-MiniLM-L12-v2"),
        device: str | None = None,
    ) -> None:
        """Configura identificação do modelo de embedding."""
        self.model_name = model_name
        self.device = device
        self._embeddings: np.ndarray | None = None

    def fit(self, documents: List[str]) -> "DenseIndex":
        """Codifica todos os documentos e armazena os embeddings.

        Args:
            documents: Lista de textos (pré-processados ou raw).

        Returns:
            Self para encadeamento.
        """
        model = _ensure_dense_model(self.model_name, self.device)
        self._embeddings = model.encode(
            documents,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

        # Normalizar vetores para cosseno via dot product
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        self._embeddings = self._embeddings / norms

        return self

    def get_scores_normalized(self, query: str) -> np.ndarray:
        """Similaridade de cosseno da query contra o corpus.

        Retorna scores já normalizados em ``[0, 1]``.

        Args:
            query: Texto da query.

        Returns:
            Array de scores com shape ``(n_candidates,)``.
        """
        if self._embeddings is None:
            return np.array([], dtype=np.float32)

        model = _ensure_dense_model(self.model_name, self.device)
        q_emb = model.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

        # Normalizar query
        q_norm = np.linalg.norm(q_emb)
        if q_norm > 1e-10:
            q_emb = q_emb / q_norm

        # Cosseno = dot product (vetores já normalizados)
        scores = self._embeddings @ q_emb.flatten()

        # Clip para [0, 1] (cosseno pode ser negativo)
        return np.clip(scores, 0.0, 1.0)
