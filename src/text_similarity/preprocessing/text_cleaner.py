"""Módulo para saneamento básico de string, desativando pontuações e acentos textuais."""

from __future__ import annotations
import re
import unicodedata


class TextCleaner:
    """Limpa e normaliza textos em português.
    
    Responsabilidades: lowercase, remoção de acentos/pontuação extra,
    e expansão de contrações básicas.
    """

    # Principais contrações do português para expansão
    CONTRACOES = {
        " do ": " de o ",
        " da ": " de a ",
        " dos ": " de os ",
        " das ": " de as ",
        " no ": " em o ",
        " na ": " em a ",
        " nos ": " em os ",
        " nas ": " em as ",
        " num ": " em um ",
        " numa ": " em uma ",
        " nums ": " em uns ",
        " numas ": " em umas ",
        " ao ": " a o ",
        " à ": " a a ",
        " aos ": " a os ",
        " às ": " a as ",
        " pelo ": " por o ",
        " pela ": " por a ",
        " pelos ": " por os ",
        " pelas ": " por as ",
    }

    def __init__(
        self,
        remove_accents: bool = True,
        expand_contractions: bool = True,
        remove_punctuation: bool = True,
    ) -> None:
        """Inicializa configurações de limpeza baseadas em flags booleanas."""
        self._remove_accents = remove_accents
        self._expand_contractions = expand_contractions
        self._remove_punctuation = remove_punctuation

    def clean(self, text: str) -> str:
        """Limpa o texto conforme as predefinições de instância passadas no contrutor."""
        text = text.lower()

        if self._expand_contractions:
            # Adicionamos espaços nas bordas para replace de palavras soltas
            text = f" {text} "
            for contracao, expansao in self.CONTRACOES.items():
                text = text.replace(contracao, expansao)
            text = text.strip()

        # Encodar a string matará acentos de "comum", mas matará as tags
        # de entidades também, que estão formadas em ascii. Porém o Normalize
        # não deve destruir o <money...  Acontece que o replace dos entities
        # precisa ficar robusto ao ASCII ignore.
        if self._remove_accents:
            text = (
                unicodedata.normalize("NFKD", text)
                .encode("ASCII", "ignore")
                .decode("utf-8")
            )

        if self._remove_punctuation:
            # Remove pontuações que não compõem palavras.
            # Cuidado para não matar as marcações de entidade do tipo <money:10>
            # Vamos manter alfanuméricos, espaços, hífens,
            # pontos (decimais) e < > : (usados pelas entidades)
            text = re.sub(r"[^a-z0-9\s$<>\-:\.]", "", text)

        # Remover espaços duplicados
        text = re.sub(r"\s+", " ", text).strip()

        return text
