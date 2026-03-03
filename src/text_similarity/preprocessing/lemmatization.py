"""Módulo de Lematização usando NLTK e SpaCy opcional e como fallback."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class Lemmatizer:
    """Lematizador PT-BR.

    Tenta utilizar spaCy primeiramente, se disponível, para alta precisão.
    Senão, utiliza NLTK (RSLPStemmer) como fallback que provê Stemming.
    Se nenhum estiver disponível, atua como um pass-through e loga um aviso.
    """

    def __init__(self) -> None:
        """Tenta injetar backends de linguística."""
        self.backend = "none"
        self._nlp = None
        self._stemmer = None

        # 1. Tentar spaCy (Requer: spacy e pt_core_news_sm)
        try:
            import spacy

            # Tentar carregar o modelo PT
            self._nlp = spacy.load("pt_core_news_sm")
            self.backend = "spacy"
            return
        except ImportError:
            pass
        except OSError:
            # SpaCy instalado mas sem o modelo pt_core_news_sm baixado
            pass

        # 2. Tentar NLTK Stemmer
        try:
            from nltk.stem import RSLPStemmer

            # Tenta instanciar para ver se os dados estão baixados
            try:
                self._stemmer = RSLPStemmer()
                self.backend = "nltk"
                return
            except LookupError:
                # Necessita: nltk.download('rslp') e nltk.download('punkt')
                pass
        except ImportError:
            pass

        logger.warning(
            "Lemmatizer operando em modo pass-through (nenhum backend encotrado). "
            "Para lematização real, instale o spaCy e o modelo pt_core_news_sm."
        )

    def lemmatize(self, tokens: list[str]) -> list[str]:
        """Aplica stemming/lematização aos tokens.

        Tags de entidade são ignoradas e retornadas como estão.
        """
        if not tokens:
            return tokens

        if self.backend == "spacy" and self._nlp is not None:
            # Juntamos os tokens e usamos as quebras do spacy.
            # Mas tratamos token a token se n for entidade.
            lemmatized_tokens = []
            for t in tokens:
                if t.startswith("<"):
                    lemmatized_tokens.append(t)
                else:
                    doc = self._nlp(" ".join([t]))
                    # Pegamos as lemmas dos múltiplos tokens gerados.
                    lemmas = [token.lemma_ for token in doc]
                    if lemmas:
                        lemmatized_tokens.append("".join(lemmas))
                    else:
                        lemmatized_tokens.append(t)
            return lemmatized_tokens

        if self.backend == "nltk" and self._stemmer is not None:
            return [t if t.startswith("<") else self._stemmer.stem(t) for t in tokens]

        # Fallback none
        return tokens
