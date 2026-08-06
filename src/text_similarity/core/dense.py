"""Índice denso baseado em sentence-transformers para filtragem semântica.

Índice de embeddings densos para candidatos, otimizado para recall
semântico em PT-BR. Utiliza similaridade de cosseno entre vetores
densos como filtro de primeiro estágio em operações batch.

Compatível com ``ProcessPoolExecutor`` (pickle-safe): armazena apenas
arrays numpy, sem referências ao modelo de embedding.
"""

from __future__ import annotations

import io
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
from numpy.typing import NDArray

from text_similarity.core import _serialization

logger = logging.getLogger(__name__)

# Cache global lazy — mesmo padrão de semantic.py.
# O modelo NÃO é armazenado na instância (não é pickle-safe).
# Workers recriam o modelo localmente via este cache global.
_DENSE_MODEL: Any = None
_DENSE_MODEL_KEY: tuple[str, str | None, str | None] | None = None
_DENSE_LOCK = threading.Lock()

INDEX_FORMAT_VERSION = _serialization.INDEX_FORMAT_VERSION


def _ensure_dense_model(
    model_name: str,
    device: str | None = None,
    revision: str | None = None,
) -> Any:
    """Carrega o modelo de embedding globalmente (lazy, thread-safe).

    Utiliza Double-Checked Locking para evitar contenção após
    a primeira carga. A chave de cache inclui ``model_name``, ``device``
    e ``revision`` para evitar reuso incorreto quando algum desses
    parâmetros muda (SEC-LOGIC-005).

    Args:
        model_name: Nome/path do modelo no HuggingFace.
        device: Dispositivo ('cpu', 'cuda', etc). Se None, auto.
        revision: Revisão (SHA) do modelo. Se None, usa a revisão padrão.

    Returns:
        Instância de ``SentenceTransformer`` carregada.
    """
    global _DENSE_MODEL, _DENSE_MODEL_KEY

    model_key = (model_name, device, revision)

    if _DENSE_MODEL is not None and _DENSE_MODEL_KEY == model_key:
        return _DENSE_MODEL

    with _DENSE_LOCK:
        if _DENSE_MODEL is not None and _DENSE_MODEL_KEY == model_key:
            return _DENSE_MODEL

        logger.info(
            "Carregando modelo denso para indexação: %s (device=%s, revision=%s)",
            model_name,
            device,
            revision,
        )
        try:
            from sentence_transformers import SentenceTransformer

            kwargs: dict[str, Any] = {}
            if device:
                kwargs["device"] = device
            if revision:
                kwargs["revision"] = revision

            _DENSE_MODEL = SentenceTransformer(model_name, **kwargs)
            _DENSE_MODEL_KEY = model_key
            return _DENSE_MODEL

        except ImportError as e:
            raise ImportError(
                _serialization.sentence_transformers_install_hint("DenseIndex")
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
        revision: Revisão (SHA) do modelo. Se informado, propagado como
            ``revision=<sha>`` para ``SentenceTransformer``.
        precision: Precisão dos embeddings armazenados.
            ``"float32"`` (padrão) — qualidade máxima, ~4 bytes/dim.
            ``"int8"`` — quantização escalar, ~1 byte/dim (~75% menor).
            ``"binary"`` — bits empacotados, ~0.125 byte/dim (~97% menor).
    """

    def __init__(
        self,
        model_name: str = ("paraphrase-multilingual-MiniLM-L12-v2"),
        device: str | None = None,
        revision: str | None = None,
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
        self.revision = revision
        self.precision = precision
        self._embeddings: NDArray[Any] | None = None
        self.n_documents: int = 0
        self.embedding_dim: int = 0

    def fit(self, documents: List[str]) -> "DenseIndex":
        """Codifica todos os documentos e armazena os embeddings.

        Args:
            documents: Lista de textos (pré-processados ou raw).

        Returns:
            Self para encadeamento.
        """
        model = _ensure_dense_model(self.model_name, self.device, self.revision)
        emb_f32: NDArray[np.float32] = model.encode(
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

    def get_scores_normalized(self, query: str) -> NDArray[np.float32]:
        """Similaridade da query contra o corpus, normalizada em ``[0, 1]``.

        Args:
            query: Texto da query.

        Returns:
            Array de scores com shape ``(n_candidates,)``.
        """
        if self._embeddings is None:
            return np.array([], dtype=np.float32)

        model = _ensure_dense_model(self.model_name, self.device, self.revision)
        q_emb: NDArray[np.float32] = model.encode(
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
            return np.asarray(np.clip(scores, 0.0, 1.0), dtype=np.float32)

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
            return np.asarray(np.clip(scores, 0.0, 1.0), dtype=np.float32)

        else:  # binary
            # Similaridade Hamming via XOR + popcount
            # quantize_embeddings retorna int8 com bits empacotados
            q_bits = np.packbits((q_emb.flatten() > 0).astype(np.uint8))
            emb_uint8 = self._embeddings.view(np.uint8)
            n_bits = emb_uint8.shape[1] * 8
            xor = np.bitwise_xor(emb_uint8, q_bits)
            hamming = np.unpackbits(xor, axis=1).sum(axis=1).astype(np.float32)
            similarity = 1.0 - hamming / max(n_bits, 1)
            return np.asarray(np.clip(similarity, 0.0, 1.0), dtype=np.float32)

    def save(
        self,
        path: Union[str, "os.PathLike[str]"],
        *,
        hmac_key: Union[bytes, str, None] = None,
    ) -> None:
        """Serializa o índice denso no formato ``tsbr-index-v2`` (NPZ + HMAC).

        Args:
            path: Caminho do arquivo de saída (ex: ``"dense.tsbr-index"``).
            hmac_key: Chave HMAC-SHA256 opcional (``bytes`` ou ``str``).
                Se não fornecida, tenta a variável de ambiente
                ``TSBR_HMAC_KEY``. Sem chave, o arquivo é gravado sem
                autenticação e um aviso é emitido.
        """
        meta: Dict[str, Any] = {
            "model_name": self.model_name,
            "device": self.device,
            "revision": self.revision,
            "precision": self.precision,
            "n_documents": self.n_documents,
            "embedding_dim": self.embedding_dim,
        }
        meta_bytes = json.dumps(meta, sort_keys=True, ensure_ascii=False).encode(
            "utf-8"
        )
        len_meta = len(meta_bytes).to_bytes(4, "big")

        buffer = io.BytesIO()
        if self._embeddings is not None:
            np.savez_compressed(buffer, embeddings=self._embeddings)
        else:
            np.savez_compressed(buffer, embeddings=np.array([], dtype=np.float32))
        npz_bytes = buffer.getvalue()

        payload_bytes = len_meta + meta_bytes + npz_bytes
        _serialization.dump_authenticated_bytes(
            payload_bytes,
            Path(path),
            type_name="DenseIndex",
            hmac_key=hmac_key,
        )

    @classmethod
    def load(
        cls,
        path: Union[str, "os.PathLike[str]"],
        *,
        hmac_key: Union[bytes, str, None] = None,
        allow_legacy_pickle: bool = False,
    ) -> "DenseIndex":
        """Carrega e valida um índice denso do disco.

        Suporta o formato seguro ``tsbr-index-v2`` (NPZ + HMAC) e,
        mediante opt-in explícito, arquivos legados em pickle/joblib.

        Args:
            path: Caminho do arquivo gerado por ``save()``.
            hmac_key: Chave HMAC para validar autenticação. Se o arquivo
                foi gravado sem HMAC, o load prossegue com aviso.
            allow_legacy_pickle: Se ``True``, permite carregar arquivos
                antigos em pickle/joblib, emitindo ``SecurityWarning``.

        Returns:
            Instância ``DenseIndex`` pronta para uso.

        Raises:
            ValueError: Se a versão, tipo, integridade ou HMAC não
                baterem; ou se um arquivo legado for encontrado sem
                ``allow_legacy_pickle=True``.
        """
        header, payload_bytes = _serialization.load_authenticated_bytes(
            Path(path),
            hmac_key=hmac_key,
            allow_legacy_pickle=allow_legacy_pickle,
            expected_type="DenseIndex",
        )

        if header.get("format") == "legacy-pickle":
            import joblib

            payload = joblib.load(path)
            data = payload["data"]
            embeddings: NDArray[Any] | None = data.get("embeddings")
            idx = cls(
                model_name=data["model_name"],
                device=data.get("device"),
                revision=data.get("revision"),
                precision=data.get("precision", "float32"),
            )
            idx._embeddings = embeddings
            idx.n_documents = data.get("n_documents", 0)
            idx.embedding_dim = data.get("embedding_dim", 0)
            return idx

        # Formato novo: [len_meta:uint32 BE][meta_json][npz_bytes]
        if len(payload_bytes) < 4:
            raise ValueError("Payload do índice denso está vazio ou corrompido.")
        len_meta = int.from_bytes(payload_bytes[:4], "big")
        meta_end = 4 + len_meta
        if meta_end > len(payload_bytes):
            raise ValueError("Metadados do índice denso estão truncados.")

        meta = json.loads(payload_bytes[4:meta_end].decode("utf-8"))
        npz_bytes = payload_bytes[meta_end:]

        buffer = io.BytesIO(npz_bytes)
        with np.load(buffer, allow_pickle=False) as npz:
            embeddings = npz["embeddings"]

        idx = cls(
            model_name=meta["model_name"],
            device=meta.get("device"),
            revision=meta.get("revision"),
            precision=meta.get("precision", "float32"),
        )
        idx._embeddings = embeddings
        idx.n_documents = meta.get("n_documents", 0)
        idx.embedding_dim = meta.get("embedding_dim", 0)
        return idx
