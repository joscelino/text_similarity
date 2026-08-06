"""Implementação do cálculo de similaridade semântica (Word Embeddings)."""

from __future__ import annotations

import logging
import threading
from typing import Any

from text_similarity.core import _serialization
from text_similarity.core.base import SimilarityAlgorithm
from text_similarity.exceptions import StageProcessingError

logger = logging.getLogger(__name__)


class SemanticSimilarityError(Exception):
    """Falha esperada durante o cálculo de similaridade semântica.

    Encapsula erros previsíveis do backend de embeddings (ex:
    ``RuntimeError`` do PyTorch, ``torch.cuda.OutOfMemoryError``)
    quando :class:`SemanticSimilarity` opera em modo ``strict=True``.

    Em modo ``strict=True`` (padrão, recomendado para produção), erros
    esperados são convertidos nesta exceção e propagados ao chamador,
    permitindo estratégias de retry/fallback deliberadas. Em modo
    ``strict=False`` (comportamento legado tolerante) o algoritmo apenas
    loga o stacktrace via ``logger.error(..., exc_info=True)`` e retorna
    ``0.0``, mascarando o problema — útil somente para pipelines batch
    onde uma comparação individual falha silenciosamente não invalida
    o lote todo.
    """


# Cache Global Lazy Initialization
# Esse padrão garante que workers não tentem serializar o modelo de dezenas/centenas
# de MB durante a paralelização (multiprocessing) e instanciem localmente o peso ao
# inicializar.
#
# _MODEL_LOCK protege contra race conditions em ambientes multithreaded (ex: FastAPI
# com run_in_executor usando o ThreadPoolExecutor padrão). O padrão Double-Checked
# Locking garante que o lock só é adquirido na primeira carga — após isso, o
# fast-path retorna sem contenção.
_GLOBAL_MODEL: Any = None
_CURRENT_MODEL_KEY: tuple[str, str | None, str | None] | None = None
_SENTENCE_UTIL: Any = None
_MODEL_LOCK = threading.Lock()


class SemanticSimilarity(SimilarityAlgorithm):
    """Algoritmo de Similaridade Baseado em Vetores Densos.

    Utiliza o `sentence-transformers` nativamente para extrair
    características semânticas que o TF-IDF desconhece (sinônimos, contexto).

    Esta classe instancializa os modelos por Demanda ("Lazy Init"), para não penalizar
    a latência de inicialização para clientes da biblioteca que utilizem o modo 'basic'.
    """

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        device: str | None = None,
        revision: str | None = None,
        strict: bool = True,
    ) -> None:
        """Configura a identificação do modelo de Embedding.

        Args:
            model_name: O nome/path no HuggingFace. Por default, utiliza
                um modelo leve multilinguístico (inclui Português).
            device: Dispositivo ('cpu', 'cuda', etc). Se None, o
                pytorch/sentence_transformers detecta e usa o melhor hardware
                disponível localmente.
            revision: Revisão (SHA) do modelo. Se informado, propagado como
                ``revision=<sha>`` para ``SentenceTransformer``.
            strict: Se ``True`` (padrão, recomendado para produção), erros
                esperados do backend (``RuntimeError``,
                ``torch.cuda.OutOfMemoryError``) durante ``compare`` são
                re-lançados como :class:`SemanticSimilarityError`, e
                exceções inesperadas propagam livremente sem serem
                engolidas. Se ``False``, mantém o comportamento legado
                tolerante: qualquer falha resulta em ``0.0`` e um
                ``logger.error(..., exc_info=True)`` para diagnóstico.
        """
        self.model_name = model_name
        self.device = device
        self.revision = revision
        self.strict = strict

    def _ensure_model_loaded(self) -> Any:
        """Carrega o modelo lazy, armazenando globalmente por processo worker.

        Utiliza Double-Checked Locking para garantir thread-safety sem penalizar
        a performance após a carga inicial: o lock só é adquirido quando o modelo
        ainda não está disponível. A chave de cache inclui ``model_name``,
        ``device`` e ``revision`` para evitar reuso incorreto (SEC-LOGIC-005).
        """
        global _GLOBAL_MODEL, _CURRENT_MODEL_KEY, _SENTENCE_UTIL

        model_key = (self.model_name, self.device, self.revision)

        # Fast-path: sem lock — modelo já carregado para este processo/thread
        if _GLOBAL_MODEL is not None and _CURRENT_MODEL_KEY == model_key:
            return _GLOBAL_MODEL

        with _MODEL_LOCK:
            # Segundo check dentro do lock: outra thread pode ter carregado
            # o modelo enquanto esta esperava para adquirir o lock.
            if _GLOBAL_MODEL is not None and _CURRENT_MODEL_KEY == model_key:
                return _GLOBAL_MODEL

            logger.info(
                "Carregando e inicializando o modelo semântico: %s "
                "(device=%s, revision=%s)",
                self.model_name,
                self.device,
                self.revision,
            )
            try:
                # Import Local (Lazy Import) para evitar gargalos na biblioteca inteira
                # para quem depende apenas de Lexical/Phonetic
                from sentence_transformers import SentenceTransformer
                from sentence_transformers import util as st_util

                kwargs: dict[str, Any] = {}
                if self.device:
                    kwargs["device"] = self.device
                if self.revision:
                    kwargs["revision"] = self.revision

                _GLOBAL_MODEL = SentenceTransformer(self.model_name, **kwargs)
                _CURRENT_MODEL_KEY = model_key
                _SENTENCE_UTIL = st_util
                return _GLOBAL_MODEL

            except ImportError as e:
                raise ImportError(
                    _serialization.sentence_transformers_install_hint(
                        "SemanticSimilarity"
                    )
                ) from e
            except Exception as e:
                raise StageProcessingError("SemanticSimilarity", e) from e

    def compare(self, text1: str, text2: str) -> float:
        """Gera vetores densos e computa a dissimilaridade do cosseno.

        Args:
            text1: Primeiro texto
            text2: Segundo texto

        Returns:
            Float estritamente de 0.0 a 1.0 (onde 1.0 é semanticamente idêntico).

        Raises:
            SemanticSimilarityError: Se ``strict=True`` e o backend
                lançar ``RuntimeError`` (inclui ``torch.cuda.OutOfMemoryError``)
                durante o encode/cos_sim. Outras exceções propagam livremente
                sem serem engolidas.
        """
        if not text1 or not text2:
            return 0.0

        model = self._ensure_model_loaded()

        # Tuple de tipos "esperados" — falhas conhecidas do backend PyTorch.
        # torch.cuda.OutOfMemoryError herda de RuntimeError, então uma
        # única entrada cobre ambos casos sem exigir import condicional.
        expected_errors: tuple[type[BaseException], ...] = (RuntimeError,)

        if self.strict:
            try:
                emb1 = model.encode(text1, convert_to_tensor=True)
                emb2 = model.encode(text2, convert_to_tensor=True)
                cosine_scores = _SENTENCE_UTIL.cos_sim(emb1, emb2)
                score = float(cosine_scores[0][0])  # pyright: ignore
                return max(0.0, min(1.0, score))
            except expected_errors as e:
                # Erro esperado: encapsula em SemanticSimilarityError
                # preservando o traceback original via ``raise ... from e``.
                raise SemanticSimilarityError(
                    f"Falha ao computar similaridade semântica: {e}"
                ) from e
            # Demais exceções propagam SEM captura silenciosa (sem
            # ``except Exception``) — comportamento recomendado para
            # produção: bugs inesperados devem falhar visivelmente.

        # Modo tolerante (strict=False): mantém retorno 0.0 mas registra
        # stacktrace completo para diagnóstico posterior.
        try:
            emb1 = model.encode(text1, convert_to_tensor=True)
            emb2 = model.encode(text2, convert_to_tensor=True)
            cosine_scores = _SENTENCE_UTIL.cos_sim(emb1, emb2)
            score = float(cosine_scores[0][0])  # pyright: ignore
            return max(0.0, min(1.0, score))
        except Exception:
            logger.error(
                "Erro ao inferir Similaridade Semântica (strict=False, "
                "retornando 0.0). Considere strict=True em produção.",
                exc_info=True,
            )
            return 0.0

    def unload(self) -> None:
        """Libera o modelo semântico e o módulo util da memória global.

        Útil para liberar RAM/VRAM após uma sessão de inferência intensa,
        ou antes de trocar para um modelo diferente.
        """
        global _GLOBAL_MODEL, _CURRENT_MODEL_KEY, _SENTENCE_UTIL
        with _MODEL_LOCK:
            _GLOBAL_MODEL = None
            _CURRENT_MODEL_KEY = None
            _SENTENCE_UTIL = None
        logger.info("Modelo semântico descarregado da memória.")
