"""Núcleo da classe :class:`Comparator` (SEC-STD-001).

A responsabilidade deste módulo é apenas *orquestrar* a inicialização
do :class:`Comparator` — pipeline, cache, algoritmo, configuração e
factories. Todo o comportamento comportamental é herdado dos mixins
em :mod:`text_similarity.api.batch`, :mod:`text_similarity.api.scoring`,
:mod:`text_similarity.api.index_manager` e
:mod:`text_similarity.api.dataframe_ops`.
"""

from __future__ import annotations

import copy
import threading
from typing import Any, Dict, List, Literal

from text_similarity.api.batch import BatchMixin
from text_similarity.api.dataframe_ops import DataFrameOpsMixin
from text_similarity.api.index_manager import IndexManagerMixin
from text_similarity.api.scoring import ScoringMixin
from text_similarity.config import (
    BASIC_WEIGHTS,
    BASIC_WEIGHTS_WITH_SEMANTIC,
    SMART_WEIGHTS,
    SMART_WEIGHTS_WITH_SEMANTIC,
    ComparatorConfig,
)
from text_similarity.core._index_protocol import IndexProtocol
from text_similarity.core.base import SimilarityAlgorithm
from text_similarity.core.fusion import RRFusion
from text_similarity.core.hybrid import HybridSimilarity
from text_similarity.pipeline.backends import (
    CleanTextStage,
    LemmatizeStage,
    NormalizeEntitiesStage,
    StopwordsStage,
    TokenizerStage,
)
from text_similarity.pipeline.cache import PipelineCache
from text_similarity.pipeline.pipeline import PreprocessingPipeline
from text_similarity.pipeline.stage import PipelineStage


class Comparator(
    BatchMixin,
    DataFrameOpsMixin,
    IndexManagerMixin,
    ScoringMixin,
):
    """Classe principal para comparação de similaridade de textos em português.

    Thread-safe: todas as operações de leitura/escrita no cache in-memory
    são protegidas por ``self._cache_lock``. O pré-processamento em lote
    pode ser distribuído via ``ThreadPoolExecutor`` quando o volume de
    textos excede ``parallel_threshold``.
    """

    def __init__(
        self,
        mode: str = "basic",
        entities: list[str] | None = None,
        use_cache: bool = True,
        use_embeddings: bool = False,
        fusion_strategy: Literal["linear", "rrf"] = "linear",
        rrf_k: int = 60,
        rrf_weights: dict[str, float] | None = None,
        indexing_strategy: Literal["tfidf", "bm25", "dense"] = "tfidf",
        bm25_k1: float = 1.2,
        bm25_b: float = 0.75,
        dense_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        dense_model_revision: str | None = None,
        dense_precision: str = "float32",
        parallel_threshold: int = 1000,
        strict: bool = True,
        weights: dict[str, float] | None = None,
        *,
        config: ComparatorConfig | None = None,
    ) -> None:
        """Inicializa a classe Comparator preparando o pipeline.

        A configuração pode ser fornecida de duas formas equivalentes:

        1. Passando um :class:`~text_similarity.config.ComparatorConfig`
           pronto via ``config=`` (recomendado para reuso).
        2. Passando os parâmetros individuais (retro-compatível com a
           API anterior). Nesse caso um ``ComparatorConfig`` é construído
           internamente.

        Args:
            mode: Modo de operação ('basic' ou 'smart').
            entities: Lista de entidades para extrair no modo smart.
            use_cache: Se True, habilita o cache in-memory.
            use_embeddings: Ativa Similaridade Semântica baseada em
                embeddings densos.
            fusion_strategy: Estratégia de fusão para operações batch
                (``"linear"`` ou ``"rrf"``).
            rrf_k: Parâmetro de suavização do RRF (padrão 60).
            rrf_weights: Pesos por algoritmo para o RRF.
            indexing_strategy: ``"tfidf"`` (padrão), ``"bm25"`` ou
                ``"dense"``.
            bm25_k1: Saturação de term frequency do BM25 (padrão 1.2).
            bm25_b: Normalização por comprimento do BM25 (padrão 0.75).
            dense_model_name: Nome do modelo sentence-transformers.
                Não aceite valores de usuários não confiáveis; aplique
                whitelist na aplicação hospedeira se necessário.
            dense_model_revision: Revisão (SHA) do modelo sentence-transformers
                a ser carregada. Permite pin de supply chain propagando
                ``revision=<sha>`` para ``SentenceTransformer``.
            dense_precision: ``"float32"`` (padrão), ``"int8"`` ou
                ``"binary"``.
            parallel_threshold: Número mínimo de textos para ativar
                pré-processamento paralelo em ``_process_batch``.
            strict: Modo estrito para
                :class:`~text_similarity.core.semantic.SemanticSimilarity`.
            weights: Override opcional do perfil de pesos padrão.
            config: :class:`ComparatorConfig` pronto — quando fornecido,
                demais parâmetros são ignorados.

        Raises:
            TypeError: Se um parâmetro desconhecido for passado.
            ValueError: Se ``weights`` não somar 1.0 ou ``mode`` inválido.
        """
        if config is None:
            config = ComparatorConfig(
                mode=mode,
                entities=entities,
                use_cache=use_cache,
                use_embeddings=use_embeddings,
                fusion_strategy=fusion_strategy,
                rrf_k=rrf_k,
                rrf_weights=rrf_weights,
                indexing_strategy=indexing_strategy,
                bm25_k1=bm25_k1,
                bm25_b=bm25_b,
                dense_model_name=dense_model_name,
                dense_model_revision=dense_model_revision,
                dense_precision=dense_precision,
                parallel_threshold=parallel_threshold,
                strict=strict,
                weights=weights,
            )

        self.config = config
        self.mode = config.mode
        self.entities = config.entities
        self.use_cache = config.use_cache
        self.use_embeddings = config.use_embeddings
        self.fusion_strategy = config.fusion_strategy
        self.rrf_k = config.rrf_k
        self.rrf_weights = config.rrf_weights
        self.indexing_strategy = config.indexing_strategy
        self.bm25_k1 = config.bm25_k1
        self.bm25_b = config.bm25_b
        self.dense_model_name = config.dense_model_name
        self.dense_model_revision = config.dense_model_revision
        self.dense_precision = config.dense_precision
        self.parallel_threshold = config.parallel_threshold
        self.strict = config.strict
        self._active_index: IndexProtocol | None = None
        self._rrf_fusion: RRFusion | None = (
            RRFusion(k=self.rrf_k, weights=self.rrf_weights)
            if self.fusion_strategy == "rrf"
            else None
        )

        # Cache in-memory: hash SHA-256 do texto → texto pré-processado
        self.cache: PipelineCache | None = PipelineCache() if self.use_cache else None
        self._cache_store: dict[str, str] = {}
        self._cache_lock = threading.Lock()

        # Pipeline
        stages: List[PipelineStage] = []
        if self.mode == "smart":
            from text_similarity.entities.normalizer import EntityNormalizer

            stages.append(
                NormalizeEntitiesStage(
                    normalizer=EntityNormalizer(entities=self.entities)
                )
            )

        stages.extend(
            [
                CleanTextStage(),
                TokenizerStage(),
                StopwordsStage(),
                LemmatizeStage(),
            ]
        )
        self.pipeline = PreprocessingPipeline(stages)

        # Algoritmo
        entity_types = [e.replace("_", "") for e in (self.entities or [])] or None

        resolved_weights = self._resolve_weights(config)
        self.algorithm: SimilarityAlgorithm = HybridSimilarity(
            weights=resolved_weights,
            target_entities=entity_types,
            semantic_strict=self.strict,
            semantic_model_name=self.dense_model_name,
            semantic_device=None,
            semantic_revision=self.dense_model_revision,
        )

    def __deepcopy__(self, memo: Dict[int, Any]) -> "Comparator":
        """Cópia profunda segura para calibração.

        Recria ``_cache_lock`` em vez de tentar copiar o objeto
        ``threading.Lock`` (não serializável via ``pickle``).
        """
        new_obj: Comparator = self.__class__.__new__(self.__class__)
        memo[id(self)] = new_obj
        for attr_name, attr_value in self.__dict__.items():
            if attr_name == "_cache_lock":
                setattr(new_obj, attr_name, threading.Lock())
            else:
                setattr(new_obj, attr_name, copy.deepcopy(attr_value, memo))
        return new_obj

    # ------------------------------------------------------------------
    # Config helpers e factories
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_weights(config: ComparatorConfig) -> dict[str, float]:
        """Seleciona o perfil de pesos do :class:`HybridSimilarity`.

        Ordem de precedência:
            1. ``config.weights`` (override do usuário).
            2. Perfil default de :mod:`text_similarity.config.default_weights`
               escolhido a partir de ``mode``/``use_embeddings``.
        """
        if config.weights is not None:
            return dict(config.weights)

        if config.mode == "smart":
            profile = (
                SMART_WEIGHTS_WITH_SEMANTIC if config.use_embeddings else SMART_WEIGHTS
            )
        else:
            profile = (
                BASIC_WEIGHTS_WITH_SEMANTIC if config.use_embeddings else BASIC_WEIGHTS
            )
        return dict(profile)

    @classmethod
    def basic(cls) -> "Comparator":
        """Instancia um Comparator no modo básico.

        Delega os defaults ao :class:`ComparatorConfig` — sem duplicação.
        """
        return cls(config=ComparatorConfig(mode="basic"))

    @classmethod
    def smart(
        cls,
        entities: list[str] | None = None,
        use_cache: bool = True,
        use_embeddings: bool = False,
        fusion_strategy: Literal["linear", "rrf"] = "linear",
        rrf_k: int = 60,
        rrf_weights: dict[str, float] | None = None,
        indexing_strategy: Literal["tfidf", "bm25", "dense"] = "tfidf",
        bm25_k1: float = 1.2,
        bm25_b: float = 0.75,
        dense_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        dense_model_revision: str | None = None,
        dense_precision: str = "float32",
        parallel_threshold: int = 1000,
        strict: bool = True,
    ) -> "Comparator":
        """Instancia um Comparator no modo inteligente (smart).

        Ativa a extração de entidades, unifica tokens, analisa a fonética
        PT-BR e cruza resultados de múltiplos algoritmos. Se
        ``use_embeddings=True``, ativa Similaridade Semântica.
        """
        config = ComparatorConfig(
            mode="smart",
            entities=entities,
            use_cache=use_cache,
            use_embeddings=use_embeddings,
            fusion_strategy=fusion_strategy,
            rrf_k=rrf_k,
            rrf_weights=rrf_weights,
            indexing_strategy=indexing_strategy,
            bm25_k1=bm25_k1,
            bm25_b=bm25_b,
            dense_model_name=dense_model_name,
            dense_model_revision=dense_model_revision,
            dense_precision=dense_precision,
            parallel_threshold=parallel_threshold,
            strict=strict,
        )
        return cls(config=config)

    # ------------------------------------------------------------------
    # Pré-processamento e cache
    # ------------------------------------------------------------------
    def _process(self, text: str, preprocess: bool = True) -> str:
        """Pré-processa o texto pelo pipeline, com cache in-memory.

        Quando ``preprocess=False``, retorna o texto sem alterações.
        """
        if not preprocess:
            return text

        if self.cache is not None:
            key = self.cache.hash_text(text)
            with self._cache_lock:
                if key in self._cache_store:
                    return self._cache_store[key]
        else:
            key = None

        processed, _ = self.pipeline.process(text)

        if self.cache is not None and key is not None:
            with self._cache_lock:
                self._cache_store[key] = processed

        return processed

    def clear_cache(self) -> None:
        """Limpa o cache in-memory e o cache em disco do Joblib."""
        with self._cache_lock:
            self._cache_store.clear()
        if self.cache is not None:
            self.cache.clear()

    @property
    def _entity_names(self) -> "list[str] | None":
        """Retorna a lista de entidades configuradas."""
        return self.entities

    def _process_batch(self, texts: List[str], preprocess: bool = True) -> List[str]:
        """Pré-processa uma lista de textos em lote, reutilizando cache.

        Para lotes grandes (>``parallel_threshold`` textos), distribui
        o trabalho entre múltiplos processos.
        """
        if not preprocess:
            return list(texts)

        if len(texts) > self.parallel_threshold:
            from text_similarity.pipeline.parallel_preprocess import (
                run_parallel_preprocess,
            )

            processed = run_parallel_preprocess(
                texts,
                self.mode,
                self._entity_names,
                threshold=self.parallel_threshold,
            )
        else:
            processed = [self._process(text, preprocess=preprocess) for text in texts]

        if self.cache is not None:
            with self._cache_lock:
                for text, p_text in zip(texts, processed):
                    key = self.cache.hash_text(text)
                    self._cache_store[key] = p_text

        return processed

    # ------------------------------------------------------------------
    # Comparação pair-wise
    # ------------------------------------------------------------------
    def compare(self, text1: str, text2: str, preprocess: bool = True) -> float:
        """Compara dois textos e retorna um valor global de similaridade.

        Returns:
            Score entre 0.0 (diferentes) e 1.0 (idênticos).
        """
        p_text1 = self._process(text1, preprocess=preprocess)
        p_text2 = self._process(text2, preprocess=preprocess)
        return self.algorithm.compare(p_text1, p_text2)

    def explain(
        self, text1: str, text2: str, preprocess: bool = True
    ) -> dict[str, Any]:
        """Retorna as predições individuais de cada algoritmo.

        Returns:
            Dicionário com 'score' e 'details' por algoritmo.
        """
        p_text1 = self._process(text1, preprocess=preprocess)
        p_text2 = self._process(text2, preprocess=preprocess)

        if isinstance(self.algorithm, HybridSimilarity):
            return self.algorithm.explain(p_text1, p_text2)

        score = self.algorithm.compare(p_text1, p_text2)
        return {"score": score, "details": {"algorithm": score}}


__all__ = ["Comparator"]
