"""Implementação do cálculo de similaridade semântica (Word Embeddings)."""

from __future__ import annotations

import logging
from typing import Any

from text_similarity.core.base import SimilarityAlgorithm
from text_similarity.exceptions import StageProcessingError

logger = logging.getLogger(__name__)

# Cache Global Lazy Initialization
# Esse padrão garante que workers não tentem serializar o modelo de dezenas/centenas de MB
# durante a paralelização (multiprocessing) e instanciem localmente o peso ao inicializar.
_GLOBAL_MODEL: Any = None
_CURRENT_MODEL_NAME: str | None = None


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
    ) -> None:
        """Configura a identificação do modelo de Embedding.

        Args:
            model_name: O nome/path no HuggingFace. Por default, utiliza
                um modelo leve multilinguístico (inclui Português).
            device: Dispositivo ('cpu', 'cuda', etc). Se None, o pytorch/sentence_transformers
                detecta e usa o melhor hardware disponível localmente.
        """
        self.model_name = model_name
        self.device = device
        self._model_ref = None

    def _ensure_model_loaded(self) -> Any:
        """Carrega o modelo lazy, armazenando globalmente por processo worker."""
        global _GLOBAL_MODEL, _CURRENT_MODEL_NAME

        # Fast path
        if _GLOBAL_MODEL is not None and _CURRENT_MODEL_NAME == self.model_name:
            return _GLOBAL_MODEL

        logger.info(f"Carregando e inicializando o modelo semântico: {self.model_name}")
        try:
            # Import Local (Lazy Import) para evitar gargalos na biblioteca inteira
            # para quem depende apenas de Lexical/Phonetic
            from sentence_transformers import SentenceTransformer

            # Configura um dictionary para passar device apenas se fornecido
            kwargs = {}
            if self.device:
                kwargs["device"] = self.device

            _GLOBAL_MODEL = SentenceTransformer(self.model_name, **kwargs)
            _CURRENT_MODEL_NAME = self.model_name
            return _GLOBAL_MODEL

        except ImportError as e:
            raise ImportError(
                "A computação Semântica requer a lib `sentence-transformers`. "
                "Para instalá-la, rode: pip install text_similarity[semantic]"
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
        """
        if not text1 or not text2:
            return 0.0

        model = self._ensure_model_loaded()

        try:
            from sentence_transformers import util

            # encode() processa a frase e devolve o tensor Numpy (PyTorch cpu/cuda Tensor)
            # convert_to_tensor=True garante que util.cos_sim opere em PyTorch nativo
            emb1 = model.encode(text1, convert_to_tensor=True)
            emb2 = model.encode(text2, convert_to_tensor=True)

            # Cosine_Similarity pode retornar tensores bidimensionais de 1x1 nestes casos
            cosine_scores = util.cos_sim(emb1, emb2)
            score = float(cosine_scores[0][0])  # pyright: ignore

            # Mantém no bucket da interface Base
            return max(0.0, min(1.0, score))

        except Exception as e:
            logger.error(f"Erro ao inferir Similaridade Semântica: {e}")
            return 0.0
