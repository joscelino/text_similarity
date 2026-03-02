"""Módulo contendo os adaptadores (backends) que envelopam os processadores de texto na interface PipelineStage."""

from __future__ import annotations

from text_similarity.entities.normalizer import EntityNormalizer
from text_similarity.pipeline.stage import PipelineContext, PipelineStage
from text_similarity.preprocessing.lemmatization import Lemmatizer
from text_similarity.preprocessing.stopwords import StopwordsFilter
from text_similarity.preprocessing.text_cleaner import TextCleaner
from text_similarity.preprocessing.tokenizer import Tokenizer


class CleanTextStage(PipelineStage):
    """Estágio responsável por limpar o texto removendo acentos e pontuações extras."""

    def __init__(self, cleaner: TextCleaner | None = None) -> None:
        """Inicializa o estágio de limpeza básica."""
        self.cleaner = cleaner or TextCleaner()

    def process(self, ctx: PipelineContext) -> PipelineContext:
        """Aplica a limpeza baseada no TextCleaner ao texto do contexto."""
        ctx.text = self.cleaner.clean(ctx.text)
        return ctx


class NormalizeEntitiesStage(PipelineStage):
    """Estágio de pipeline que detecta entidades numéricas e as protege contra deturpação."""

    def __init__(self, normalizer: EntityNormalizer | None = None) -> None:
        """Inicializa o estágio de normalização de entidades."""
        self.normalizer = normalizer or EntityNormalizer()

    def process(self, ctx: PipelineContext) -> PipelineContext:
        """Varre e substitui entidades no texto injetando suas tags no contexto."""
        ctx.text = self.normalizer.normalize(ctx.text)
        return ctx


class TokenizerStage(PipelineStage):
    """Estágio que quebra strings limpas em listas de tokens, mantendo as tags de entidade ilesas."""

    def __init__(self, tokenizer: Tokenizer | None = None) -> None:
        """Inicializa o tokenizador."""
        self.tokenizer = tokenizer or Tokenizer()

    def process(self, ctx: PipelineContext) -> PipelineContext:
        """Tokeniza o texto e armazena os tokens diretamente em `ctx.tokens`."""
        ctx.tokens = self.tokenizer.tokenize(ctx.text)
        ctx.text = " ".join(ctx.tokens)
        return ctx


class StopwordsStage(PipelineStage):
    """Estágio intermediário para remoção de Stopwords do dicionário NLTK em Português."""

    def __init__(self, filter: StopwordsFilter | None = None) -> None:
        """Inicia o filtro de StopWords parametrizado para ler e reescrever `ctx.tokens`."""
        self.filter = filter or StopwordsFilter()

    def process(self, ctx: PipelineContext) -> PipelineContext:
        """Elimina conjunções e elipses irrelevantes, atualizando tokens e texto no contexto."""
        tokens = ctx.tokens if ctx.tokens else ctx.text.split()
        ctx.tokens = self.filter.filter(tokens)
        ctx.text = " ".join(ctx.tokens)
        return ctx


class LemmatizeStage(PipelineStage):
    """Último estágio opcional que reduz tokens ao radical/lemma via SpaCy."""

    def __init__(self, lemmatizer: Lemmatizer | None = None) -> None:
        """Disponibiliza o SpaCy ou fallback como backend de Lematização."""
        self.lemmatizer = lemmatizer or Lemmatizer()

    def process(self, ctx: PipelineContext) -> PipelineContext:
        """Lematiza os tokens em `ctx.tokens` e reconstrói `ctx.text` como bag of words."""
        tokens = ctx.tokens if ctx.tokens else ctx.text.split()
        ctx.tokens = self.lemmatizer.lemmatize(tokens)
        ctx.text = " ".join(ctx.tokens)
        return ctx
