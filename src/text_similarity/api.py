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
    """
    Classe principal para comparação de similaridade de textos em português.
    """

    def __init__(
        self,
        mode: str = "basic",
        entities: list[str] | None = None,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> None:
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
        """
        Retorna o Comparator básico: sem detecção algorítmica de entidades e
        focado em Bag of Words + Levenshtein.
        """
        return cls(mode="basic")

    @classmethod
    def smart(
        cls,
        entities: list[str] | None = None,
        use_cache: bool = True,
    ) -> "Comparator":
        """
        Retorna o Comparator inteligente: extrai entidades, unifica tokens,
        analisa a fonética br e cruza resultados.
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
        """
        Compara dois textos e retorna um valor de similaridade global.
        """
        p_text1 = self._process(text1)
        p_text2 = self._process(text2)
        return self.algorithm.compare(p_text1, p_text2)

    def explain(self, text1: str, text2: str) -> dict[str, Any]:
        """
        Retorna o score e detalhes de cada algoritmo aplicado.
        """
        p_text1 = self._process(text1)
        p_text2 = self._process(text2)

        # Como o base similarity algorithm só tem compare() formalmente,
        # fazemos cast se for a implementação híbrida.
        if isinstance(self.algorithm, HybridSimilarity):
            return self.algorithm.explain(p_text1, p_text2)

        score = self.algorithm.compare(p_text1, p_text2)
        return {"score": score, "details": {"algorithm": score}}

