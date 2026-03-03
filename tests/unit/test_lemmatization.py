"""Testes unitários para o módulo de lematização (Lemmatizer)."""

from unittest.mock import MagicMock, patch


# --- Testes do branch pas-through (backend "none") ---


def test_lemmatizer_passthrough_empty_tokens():
    """Garante que lista vazia é retornada sem erro no modo passthrough."""
    from text_similarity.preprocessing.lemmatization import Lemmatizer

    with patch.dict("sys.modules", {"spacy": None, "nltk": None}):
        lem = Lemmatizer()
        lem.backend = "none"
        lem._nlp = None
        lem._stemmer = None

        result = lem.lemmatize([])
        assert result == []


def test_lemmatizer_passthrough_returns_tokens_unchanged():
    """No modo pass-through, tokens são devolvidos intactos."""
    from text_similarity.preprocessing.lemmatization import Lemmatizer

    lem = Lemmatizer()
    lem.backend = "none"
    lem._nlp = None
    lem._stemmer = None

    tokens = ["correr", "casa", "rapidamente"]
    result = lem.lemmatize(tokens)
    assert result == tokens


def test_lemmatizer_passthrough_preserves_entity_tags():
    """No modo pass-through, tags de entidade são preservadas."""
    from text_similarity.preprocessing.lemmatization import Lemmatizer

    lem = Lemmatizer()
    lem.backend = "none"

    tokens = ["<money:30.0>", "produto", "<dimension:2.0kg>"]
    result = lem.lemmatize(tokens)
    assert result == tokens


# --- Testes do branch NLTK ---


def test_lemmatizer_nltk_stems_tokens():
    """Com backend NLTK, tokens regulares recebem stemming."""
    from text_similarity.preprocessing.lemmatization import Lemmatizer

    # Criar um stemmer mock que retorna uma versão simplificada do token
    mock_stemmer = MagicMock()
    mock_stemmer.stem.side_effect = lambda t: t[:4]  # Simula stem: "correr" -> "corr"

    lem = Lemmatizer()
    lem.backend = "nltk"
    lem._stemmer = mock_stemmer

    tokens = ["correr", "casas", "rapidamente"]
    result = lem.lemmatize(tokens)

    assert result == ["corr", "casa", "rapi"]
    assert mock_stemmer.stem.call_count == 3


def test_lemmatizer_nltk_skips_entity_tags():
    """Com backend NLTK, tags de entidade (<...>) não são passadas ao stemmer."""
    from text_similarity.preprocessing.lemmatization import Lemmatizer

    mock_stemmer = MagicMock()
    mock_stemmer.stem.side_effect = lambda t: t[:3]

    lem = Lemmatizer()
    lem.backend = "nltk"
    lem._stemmer = mock_stemmer

    tokens = ["<money:30.0>", "produto", "<productmodel:S22>"]
    result = lem.lemmatize(tokens)

    # Tags de entidade devem passar sem stemming
    assert result[0] == "<money:30.0>"
    assert result[2] == "<productmodel:S22>"
    # Token regular recebeu stem
    assert result[1] == "pro"
    # Stemmer foi chamado apenas 1 vez (só para "produto")
    assert mock_stemmer.stem.call_count == 1


# --- Testes do branch spaCy ---


def test_lemmatizer_spacy_lematizes_tokens():
    """Com backend spaCy, tokens regulares recebem lematização."""
    from text_similarity.preprocessing.lemmatization import Lemmatizer

    # Simular um token spacy com atributo lemma_
    mock_token = MagicMock()
    mock_token.lemma_ = "correr"

    # Simular o retorno do nlp() que é um objeto doc iterável
    mock_doc = MagicMock()
    mock_doc.__iter__ = MagicMock(return_value=iter([mock_token]))

    mock_nlp = MagicMock()
    mock_nlp.return_value = mock_doc

    lem = Lemmatizer()
    lem.backend = "spacy"
    lem._nlp = mock_nlp

    tokens = ["correndo"]
    result = lem.lemmatize(tokens)

    assert result == ["correr"]
    mock_nlp.assert_called_once_with("correndo")


def test_lemmatizer_spacy_skips_entity_tags():
    """Com backend spaCy, tags de entidade não são enviadas ao nlp()."""
    from text_similarity.preprocessing.lemmatization import Lemmatizer

    mock_nlp = MagicMock()
    mock_token = MagicMock()
    mock_token.lemma_ = "produto"
    mock_doc = MagicMock()
    mock_doc.__iter__ = MagicMock(return_value=iter([mock_token]))
    mock_nlp.return_value = mock_doc

    lem = Lemmatizer()
    lem.backend = "spacy"
    lem._nlp = mock_nlp

    tokens = ["<money:30.0>", "produto"]
    result = lem.lemmatize(tokens)

    # Tag de entidade preservada intacta
    assert result[0] == "<money:30.0>"
    # Token regular lematizado
    assert result[1] == "produto"
    # nlp() chamado apenas para "produto"
    mock_nlp.assert_called_once_with("produto")


def test_lemmatizer_spacy_empty_lemmas_fallback():
    """Com backend spaCy, se doc retornar vazio, retorna o token original."""
    from text_similarity.preprocessing.lemmatization import Lemmatizer

    mock_doc = MagicMock()
    mock_doc.__iter__ = MagicMock(return_value=iter([]))  # Doc sem lemmas

    mock_nlp = MagicMock()
    mock_nlp.return_value = mock_doc

    lem = Lemmatizer()
    lem.backend = "spacy"
    lem._nlp = mock_nlp

    tokens = ["algumtexto"]
    result = lem.lemmatize(tokens)

    # Deve retornar o token original quando o doc estiver vazio
    assert result == ["algumtexto"]
