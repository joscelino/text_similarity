"""Testes de seleção de backend do Lemmatizer sem depender de SpaCy/NLTK."""

from __future__ import annotations

import logging
import sys
import types
from typing import Any, Iterator
from unittest.mock import MagicMock, patch

import pytest

from text_similarity.preprocessing.lemmatization import Lemmatizer


def _make_spacy_module(
    load_side_effect: BaseException | None = None,
    nlp_mock: Any | None = None,
) -> Any:
    """Cria um módulo spaCy mockado."""
    spacy = types.ModuleType("spacy")
    if load_side_effect is not None:
        setattr(spacy, "load", MagicMock(side_effect=load_side_effect))
    else:
        setattr(spacy, "load", MagicMock(return_value=nlp_mock or MagicMock()))
    return spacy


def _make_nltk_module(
    stemmer_side_effect: BaseException | None = None,
    stemmer_mock: Any | None = None,
) -> Any:
    """Cria um módulo nltk com submódulo stem mockado."""
    nltk = types.ModuleType("nltk")
    stem = types.ModuleType("nltk.stem")
    if stemmer_side_effect is not None:
        setattr(stem, "RSLPStemmer", MagicMock(side_effect=stemmer_side_effect))
    else:
        setattr(
            stem,
            "RSLPStemmer",
            MagicMock(return_value=stemmer_mock or MagicMock()),
        )
    setattr(nltk, "stem", stem)
    sys.modules["nltk.stem"] = stem
    return nltk


@pytest.fixture(autouse=True)
def _clean_modules() -> Iterator[None]:
    """Remove entradas mockadas de sys.modules após cada teste."""
    yield
    for name in ("spacy", "nltk", "nltk.stem"):
        module = sys.modules.get(name)
        if module is not None and getattr(module, "__spec__", None) is None:
            del sys.modules[name]


def test_spacy_backend_selected_when_available(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """SpaCy disponível e modelo baixado -> backend='spacy'."""
    caplog.set_level(logging.INFO)
    nlp_mock = MagicMock()
    spacy = _make_spacy_module(nlp_mock=nlp_mock)

    with patch.dict(sys.modules, {"spacy": spacy}, clear=False):
        lemmatizer = Lemmatizer()

    assert lemmatizer.backend == "spacy"
    assert lemmatizer._nlp is nlp_mock
    assert "Backend de lematização selecionado: spacy" in caplog.text


def test_spacy_missing_model_falls_back_to_nltk(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """SpaCy instalado sem modelo -> fallback para NLTK."""
    caplog.set_level(logging.DEBUG)
    spacy = _make_spacy_module(load_side_effect=OSError("model not found"))
    nltk = _make_nltk_module()

    with patch.dict(sys.modules, {"spacy": spacy, "nltk": nltk}, clear=False):
        lemmatizer = Lemmatizer()

    assert lemmatizer.backend == "nltk"
    assert lemmatizer._stemmer is nltk.stem.RSLPStemmer.return_value
    assert "python -m spacy download pt_core_news_sm" in caplog.text
    assert "Backend de lematização selecionado: nltk" in caplog.text


def test_nltk_backend_selected_when_spacy_unavailable(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """SpaCy não instalado mas NLTK ok -> backend='nltk'."""
    caplog.set_level(logging.INFO)
    nltk = _make_nltk_module()

    with patch.dict(sys.modules, {"spacy": None, "nltk": nltk}, clear=False):
        lemmatizer = Lemmatizer()

    assert lemmatizer.backend == "nltk"
    assert lemmatizer._stemmer is nltk.stem.RSLPStemmer.return_value
    assert "Backend de lematização selecionado: nltk" in caplog.text


def test_nltk_missing_data_falls_back_to_passthrough(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """NLTK instalado mas sem dados baixados -> pass-through."""
    caplog.set_level(logging.DEBUG)
    nltk = _make_nltk_module(stemmer_side_effect=LookupError("rslp data missing"))

    with patch.dict(sys.modules, {"spacy": None, "nltk": nltk}, clear=False):
        lemmatizer = Lemmatizer()

    assert lemmatizer.backend == "none"
    assert lemmatizer._stemmer is None
    assert lemmatizer._nlp is None
    assert "Dados do NLTK não estão baixados" in caplog.text
    assert "pass-through" in caplog.text.lower()


def test_passthrough_when_no_backends_available(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Nenhum backend disponível -> pass-through com aviso."""
    caplog.set_level(logging.WARNING)

    with patch.dict(sys.modules, {"spacy": None, "nltk": None}, clear=False):
        lemmatizer = Lemmatizer()

    assert lemmatizer.backend == "none"
    assert lemmatizer._stemmer is None
    assert lemmatizer._nlp is None
    assert "pass-through" in caplog.text.lower()
    assert "pt_core_news_sm" in caplog.text
