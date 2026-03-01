"""Módulo contendo os adaptadores (backends) que envelopam os processadores de texto na interface PipelineStage."""

from __future__ import annotations

from typing import Any

from text_similarity.entities.normalizer import EntityNormalizer
from text_similarity.pipeline.stage import PipelineStage
from text_similarity.preprocessing.lemmatization import Lemmatizer
from text_similarity.preprocessing.stopwords import StopwordsFilter
from text_similarity.preprocessing.text_cleaner import TextCleaner
from text_similarity.preprocessing.tokenizer import Tokenizer


class CleanTextStage(PipelineStage):
    """Estágio responsável por limpar o texto removendo acentos e pontuações extras."""

    def __init__(self, cleaner: TextCleaner | None = None) -> None:
        """Inicializa o estágio de limpeza básica."""
        self.cleaner = cleaner or TextCleaner()

    def process(self, text: str, context: dict[str, Any]) -> str:
        """Aplica a limpeza baseada no TextCleaner ao texto."""
        return self.cleaner.clean(text)


class NormalizeEntitiesStage(PipelineStage):
    """Estágio de pipeline que detecta entidades numéricas e as protege contra deturpação."""

    def __init__(self, normalizer: EntityNormalizer | None = None) -> None:
        """Inicializa o estágio de normalização de entidades."""
        self.normalizer = normalizer or EntityNormalizer()

    def process(self, text: str, context: dict[str, Any]) -> str:
        """Varre e substitui entidades no texto injetando suas tags correspondentes no lugar dos numerais."""
        # A API Normalizer atual não retorna as entidades extraídas se
        # usarmos apenas normalize().
        # Precisaríamos rodar extract() para guardar no context.
        return self.normalizer.normalize(text)


class TokenizerStage(PipelineStage):
    """Estágio que quebra strings limpas e padronizadas em Arrays de tokens, mantendo as tags HTML-like ilesas."""

    def __init__(self, tokenizer: Tokenizer | None = None) -> None:
        """Inicializa o tokenizador."""
        self.tokenizer = tokenizer or Tokenizer()

    def process(self, text: str, context: dict[str, Any]) -> str:
        """Corta os textos e envia pelo `context['tokens']` à fase de Stopwords."""
        # Aqui violamos levemente a tipagem se passarmos List[str] pra frente como str
        # A Pipeline envia o `current_text` para frente, então temos que codificar
        # os tokens unidos por espaço, e depois o próximo recupera.
        # Mas para a bag of words isso já resolve.
        tokens = self.tokenizer.tokenize(text)
        context["tokens"] = tokens
        return " ".join(tokens)


class StopwordsStage(PipelineStage):
    """Estágio intermediário para remoção de Stopwords do dicionário NLTK em Português."""

    def __init__(self, filter: StopwordsFilter | None = None) -> None:
        """Inicia o filtro de StopWords parametrizado para ler e reescrever o `context['tokens']`."""
        self.filter = filter or StopwordsFilter()

    def process(self, text: str, context: dict[str, Any]) -> str:
        """Elimina conjunções e elipses irrelevantes, devolvendo no context."""
        # Usa os tokens passados no context se existirem
        tokens = context.get("tokens", text.split())
        filtered = self.filter.filter(tokens)
        context["tokens"] = filtered
        return " ".join(filtered)


class LemmatizeStage(PipelineStage):
    """Último estágio opcional que corta plural/verbos flexionados pelo radical/lemma via SpaCy."""

    def __init__(self, lemmatizer: Lemmatizer | None = None) -> None:
        """Disponibiliza o SpaCy ou fallback como backend de Lemmatização."""
        self.lemmatizer = lemmatizer or Lemmatizer()

    def process(self, text: str, context: dict[str, Any]) -> str:
        """Puxa a listagem remanescente de `context['tokens']` e aglomera a string unida reduzida."""
        tokens = context.get("tokens", text.split())
        lemmas = self.lemmatizer.lemmatize(tokens)
        context["tokens"] = lemmas
        # Retorno de bag of words
        return " ".join(lemmas)
