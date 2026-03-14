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
            import spacy  # pragma: no cover

            # Tentar carregar o modelo PT
            self._nlp = spacy.load("pt_core_news_sm")  # pragma: no cover
            self.backend = "spacy"  # pragma: no cover
            return  # pragma: no cover
        except ImportError:
            pass
        except OSError:
            # SpaCy instalado mas sem o modelo pt_core_news_sm baixado
            pass

        # 2. Tentar NLTK Stemmer
        try:
            from nltk.stem import RSLPStemmer  # pragma: no cover

            # Tenta instanciar para ver se os dados estão baixados
            try:
                self._stemmer = RSLPStemmer()  # pragma: no cover
                self.backend = "nltk"  # pragma: no cover
                return  # pragma: no cover
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
            lemmatized_tokens: list[str] = []
            entity_indices: set[int] = set()
            regular_tokens: list[str] = []

            for i, t in enumerate(tokens):
                if t.startswith("<"):
                    entity_indices.add(i)
                else:
                    regular_tokens.append(t)

            # Batch: uma chamada nlp.pipe() ao invés de N chamadas nlp()
            docs = list(self._nlp.pipe(regular_tokens, batch_size=256))

            reg_idx = 0
            for i, t in enumerate(tokens):
                if i in entity_indices:
                    lemmatized_tokens.append(t)
                else:
                    lemmas = [tok.lemma_ for tok in docs[reg_idx]]
                    lemmatized_tokens.append("".join(lemmas) if lemmas else t)
                    reg_idx += 1

            return lemmatized_tokens

        if self.backend == "nltk" and self._stemmer is not None:
            return [t if t.startswith("<") else self._stemmer.stem(t) for t in tokens]

        # Fallback none
        return tokens
