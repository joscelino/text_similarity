"""Módulo principal da API pública da biblioteca.

A interface central para comparação é através da classe `Comparator`.
"""
from __future__ import annotations
from typing import Any

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
            use_cache: Se True, instancia o cache em disco.
            **kwargs: Argumentos arbitrários passados ao algoritmo subjacente.
        """
        self.mode = mode
        self.entities = entities
        self.use_cache = use_cache
        self.cache = PipelineCache() if use_cache else None

        # Configuração do Pipeline
        stages = []
        if self.mode == "smart":
            # Smart habilita a normalização de entidades
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
             # Dá um peso maior para a fonética e entidades exatas (tokens)
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

        Ativa a extração de entidades, unifica tokens, analisa a fonética PT-BR e cruza resultados.
        """
        return cls(mode="smart", entities=entities, use_cache=use_cache)

    def _process(self, text: str) -> str:
        if self.cache:
            _ = self.cache.hash_text(text)
            # Simula a recuperação usando a Memory do Joblib
            # Joblib Memory geralmente decora funções. Para manter simples:
            # Como instanciamos o Pipeline via métodos de classe dinâmicos,
            # não usamos o decorador estático para permitir injetar entidades variadas,
            # então o Joblib LRU decoraria um método de instância,
            # o que requereria cuidado com self no hash.
            # Implementação simples sem cache ativado em disco,
            # delegamos para o chamador.
            pass

        processed, _ = self.pipeline.process(text)
        return processed

    def compare(self, text1: str, text2: str) -> float:
        """Compara dois textos e retorna um valor global de similaridade."""
        p_text1 = self._process(text1)
        p_text2 = self._process(text2)
        return self.algorithm.compare(p_text1, p_text2)

    def explain(self, text1: str, text2: str) -> dict[str, Any]:
        """Retorna as predições individuais de todos os algoritmos rodados no texto."""
        p_text1 = self._process(text1)
        p_text2 = self._process(text2)

        # Como o base similarity algorithm só tem compare() formalmente,
        # fazemos cast se for a implementação híbrida.
        if isinstance(self.algorithm, HybridSimilarity):
            return self.algorithm.explain(p_text1, p_text2)

        score = self.algorithm.compare(p_text1, p_text2)
        return {"score": score, "details": {"algorithm": score}}

