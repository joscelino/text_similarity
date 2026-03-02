"""Módulo principal da API pública da biblioteca.

A interface central para comparação é através da classe `Comparator`.
"""
from __future__ import annotations

from typing import Any, List

from text_similarity.core.base import SimilarityAlgorithm
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


class Comparator:
    """Classe principal para comparação de similaridade de textos em português."""

    def __init__(
        self,
        mode: str = "basic",
        entities: list[str] | None = None,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> None:
        """Inicializa a classe Comparator preparando o pipeline.

        Args:
            mode: Modo de operação ('basic' ou 'smart').
            entities: Lista de entidades para extrair no modo smart.
            use_cache: Se True, habilita o cache in-memory de textos processados.
            **kwargs: Argumentos arbitrários reservados para extensões futuras.
        """
        self.mode = mode
        self.entities = entities
        self.use_cache = use_cache

        # Cache in-memory: hash SHA-256 do texto original → texto pré-processado
        self.cache: PipelineCache | None = PipelineCache() if use_cache else None
        self._cache_store: dict[str, str] = {}

        # Configuração do Pipeline
        stages: List[PipelineStage] = []
        if self.mode == "smart":
            # Smart habilita a normalização de entidades antes da limpeza
            from text_similarity.entities.normalizer import EntityNormalizer

            stages.append(NormalizeEntitiesStage(
                normalizer=EntityNormalizer(entities=self.entities)
            ))

        stages.extend([
            CleanTextStage(),
            TokenizerStage(),
            StopwordsStage(),
            LemmatizeStage(),
        ])
        self.pipeline = PreprocessingPipeline(stages)

        # Configuração do Algoritmo
        if self.mode == "smart":
            # Dá peso maior para a fonética e entidades exatas (tokens)
            self.algorithm: SimilarityAlgorithm = HybridSimilarity(
                weights={"cosine": 0.45, "edit": 0.35, "phonetic": 0.20}
            )
        else:
            self.algorithm = HybridSimilarity(
                weights={"cosine": 0.5, "edit": 0.5, "phonetic": 0.0}
            )

    @classmethod
    def basic(cls) -> "Comparator":
        """Instancia um Comparator no modo básico.

        Sem detecção algorítmica de entidades e focado em Bag of Words + Levenshtein.
        """
        return cls(mode="basic")

    @classmethod
    def smart(
        cls,
        entities: list[str] | None = None,
        use_cache: bool = True,
    ) -> "Comparator":
        """Instancia um Comparator no modo inteligente.

        Ativa a extração de entidades, unifica tokens, analisa a fonética PT-BR
        e cruza resultados de múltiplos algoritmos.
        """
        return cls(mode="smart", entities=entities, use_cache=use_cache)

    def _process(self, text: str) -> str:
        """Pré-processa o texto pelo pipeline, com cache in-memory.

        Na primeira chamada para um texto, executa o pipeline completo e
        armazena o resultado. Chamadas subsequentes com o mesmo texto retornam
        o resultado cacheado diretamente.

        Args:
            text: Texto bruto de entrada.

        Returns:
            Texto pré-processado como bag of words.
        """
        if self.cache is not None:
            key = self.cache.hash_text(text)
            if key in self._cache_store:
                return self._cache_store[key]

        processed, _ = self.pipeline.process(text)

        if self.cache is not None:
            self._cache_store[key] = processed

        return processed

    def clear_cache(self) -> None:
        """Limpa o cache in-memory e o cache em disco do Joblib."""
        self._cache_store.clear()
        if self.cache is not None:
            self.cache.clear()

    def compare(self, text1: str, text2: str) -> float:
        """Compara dois textos e retorna um valor global de similaridade.

        Args:
            text1: Primeiro texto para comparação.
            text2: Segundo texto para comparação.

        Returns:
            Score entre 0.0 (completamente diferentes) e 1.0 (idênticos).
        """
        p_text1 = self._process(text1)
        p_text2 = self._process(text2)
        return self.algorithm.compare(p_text1, p_text2)

    def explain(self, text1: str, text2: str) -> dict[str, Any]:
        """Retorna as predições individuais de todos os algoritmos rodados no texto.

        Args:
            text1: Primeiro texto para comparação.
            text2: Segundo texto para comparação.

        Returns:
            Dicionário com 'score' (float) e 'details' (dict por algoritmo).
        """
        p_text1 = self._process(text1)
        p_text2 = self._process(text2)

        if isinstance(self.algorithm, HybridSimilarity):
            return self.algorithm.explain(p_text1, p_text2)

        score = self.algorithm.compare(p_text1, p_text2)
        return {"score": score, "details": {"algorithm": score}}
