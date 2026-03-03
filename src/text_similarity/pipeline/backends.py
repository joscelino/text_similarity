"""Módulo de adaptadores envelopando processadores na interface PipelineStage."""

from __future__ import annotations

import re

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
    """Estágio que detecta e protege entidades numéricas contra deturpação."""

    # Padrão: sigla curta de fabricante (2–4 letras) + espaço + sufixo alfanumérico.
    # Limite de 4 letras exclui palavras PT-BR comuns (BALAO=5, CATETER=7)
    # enquanto cobre siglas típicas: RFX, APC, HP, LG, 3M, S (Galaxy S).
    # O sufixo deve conter obrigatoriamente letras E dígitos.
    # Ex válidos  : "RFX 765J9", "S 22Ultra", "HP Z2G9", "GN 500"
    # Ex inválidos: "BALAO RFX765J9" (5 letras), "Custou 30" (sufixo só numérico)
    _MODEL_SPACE_RE = re.compile(
        r"\b([A-Za-z]{2,4})"  # grupo 1: sigla curta (2-4 letras)
        r"\s+"  # um ou mais espaços
        r"("  # grupo 2: sufixo longo que OBRIGATORIAMENTE contenha números
        r"(?=[A-Za-z\d]*\d)[A-Za-z\d]+(?:[-][A-Za-z\d]+)*"
        r")"
        r"\b"
    )

    def __init__(self, normalizer: EntityNormalizer | None = None) -> None:
        """Inicializa o estágio de normalização de entidades."""
        self.normalizer = normalizer or EntityNormalizer()

    @staticmethod
    def _collapse_model_spaces(text: str) -> str:
        """Cola siglas de modelos que têm espaço interno, antes do extrator rodar.

        Converte padrões como "RFX 765J9" em "RFX765J9" para que o
        `ProductModelExtractor` capture a entidade completa como um único token.
        Apenas casos onde a sigla tem 2–6 letras e o sufixo é alfanumérico
        misto (letras + dígitos) são normalizados, minimizando falsos positivos.
        """
        return NormalizeEntitiesStage._MODEL_SPACE_RE.sub(r"\1\2", text)

    def process(self, ctx: PipelineContext) -> PipelineContext:
        """Colapsa espaços internos de modelos e normaliza entidades no contexto."""
        ctx.text = self._collapse_model_spaces(ctx.text)
        ctx.text = self.normalizer.normalize(ctx.text)
        return ctx


class TokenizerStage(PipelineStage):
    """Quebra strings limpas em tokens, mantendo tags de entidade ilesas."""

    def __init__(self, tokenizer: Tokenizer | None = None) -> None:
        """Inicializa o tokenizador."""
        self.tokenizer = tokenizer or Tokenizer()

    def process(self, ctx: PipelineContext) -> PipelineContext:
        """Tokeniza o texto e armazena os tokens diretamente em `ctx.tokens`."""
        ctx.tokens = self.tokenizer.tokenize(ctx.text)
        ctx.text = " ".join(ctx.tokens)
        return ctx


class StopwordsStage(PipelineStage):
    """Estágio para remoção de Stopwords do dicionário NLTK em Português."""

    def __init__(self, filter: StopwordsFilter | None = None) -> None:
        """Inicia filtro de StopWords para ler e reescrever `ctx.tokens`."""
        self.filter = filter or StopwordsFilter()

    def process(self, ctx: PipelineContext) -> PipelineContext:
        """Elimina conjunções irrelevantes, atualizando tokens e texto no contexto."""
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
        """Lematiza `ctx.tokens` e reconstrói `ctx.text` como bag of words."""
        tokens = ctx.tokens if ctx.tokens else ctx.text.split()
        ctx.tokens = self.lemmatizer.lemmatize(tokens)
        ctx.text = " ".join(ctx.tokens)
        return ctx
