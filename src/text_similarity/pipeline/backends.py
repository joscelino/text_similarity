from __future__ import annotations

from typing import Any

from text_similarity.entities.normalizer import EntityNormalizer
from text_similarity.pipeline.stage import PipelineStage
from text_similarity.preprocessing.lemmatization import Lemmatizer
from text_similarity.preprocessing.stopwords import StopwordsFilter
from text_similarity.preprocessing.text_cleaner import TextCleaner
from text_similarity.preprocessing.tokenizer import Tokenizer


class CleanTextStage(PipelineStage):
    def __init__(self, cleaner: TextCleaner | None = None) -> None:
        self.cleaner = cleaner or TextCleaner()

    def process(self, text: str, context: dict[str, Any]) -> str:
        return self.cleaner.clean(text)


class NormalizeEntitiesStage(PipelineStage):
    def __init__(self, normalizer: EntityNormalizer | None = None) -> None:
        self.normalizer = normalizer or EntityNormalizer()

    def process(self, text: str, context: dict[str, Any]) -> str:
        # A API Normalizer atual não retorna as entidades extraídas se
        # usarmos apenas normalize().
        # Precisaríamos rodar extract() para guardar no context.
        return self.normalizer.normalize(text)


class TokenizerStage(PipelineStage):
    def __init__(self, tokenizer: Tokenizer | None = None) -> None:
        self.tokenizer = tokenizer or Tokenizer()

    def process(self, text: str, context: dict[str, Any]) -> str:
        # Aqui violamos levemente a tipagem se passarmos List[str] pra frente como str
        # A Pipeline envia o `current_text` para frente, então temos que codificar
        # os tokens unidos por espaço, e depois o próximo recupera.
        # Mas para a bag of words isso já resolve.
        tokens = self.tokenizer.tokenize(text)
        context["tokens"] = tokens
        return " ".join(tokens)


class StopwordsStage(PipelineStage):
    def __init__(self, filter: StopwordsFilter | None = None) -> None:
        self.filter = filter or StopwordsFilter()

    def process(self, text: str, context: dict[str, Any]) -> str:
        # Usa os tokens passados no context se existirem
        tokens = context.get("tokens", text.split())
        filtered = self.filter.filter(tokens)
        context["tokens"] = filtered
        return " ".join(filtered)


class LemmatizeStage(PipelineStage):
    def __init__(self, lemmatizer: Lemmatizer | None = None) -> None:
        self.lemmatizer = lemmatizer or Lemmatizer()

    def process(self, text: str, context: dict[str, Any]) -> str:
        tokens = context.get("tokens", text.split())
        lemmas = self.lemmatizer.lemmatize(tokens)
        context["tokens"] = lemmas
        # Retorno de bag of words
        return " ".join(lemmas)
