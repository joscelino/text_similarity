"""Testes de logging do fallback do Lemmatizer."""

from __future__ import annotations

import inspect
import logging
import sys
import types
from typing import Any, Iterator
from unittest.mock import MagicMock, patch

import pytest

from text_similarity.preprocessing import lemmatization as lem_module
from text_similarity.preprocessing.lemmatization import Lemmatizer


@pytest.fixture(autouse=True)
def _clean_modules() -> Iterator[None]:
    """Remove entradas mockadas de sys.modules após cada teste."""
    yield
    for name in ("spacy", "nltk", "nltk.stem"):
        module = sys.modules.get(name)
        if module is not None and getattr(module, "__spec__", None) is None:
            del sys.modules[name]


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


def test_spacy_import_error_logs_debug(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """ImportError do spaCy gera logger.debug com a exceção."""
    caplog.set_level(logging.DEBUG)

    with patch.dict(sys.modules, {"spacy": None}, clear=False):
        Lemmatizer()

    assert "Backend spaCy indisponível" in caplog.text


def test_spacy_missing_model_logs_download_hint(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Modelo spaCy ausente gera hint de download."""
    caplog.set_level(logging.DEBUG)
    spacy = _make_spacy_module(load_side_effect=OSError("model not found"))

    with patch.dict(sys.modules, {"spacy": spacy, "nltk": None}, clear=False):
        Lemmatizer()

    assert "python -m spacy download pt_core_news_sm" in caplog.text
    assert "pass-through" in caplog.text.lower()


def test_nltk_import_error_logs_debug(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """ImportError do NLTK gera logger.debug com a exceção."""
    caplog.set_level(logging.DEBUG)

    with patch.dict(sys.modules, {"spacy": None, "nltk": None}, clear=False):
        Lemmatizer()

    assert "Backend NLTK indisponível" in caplog.text


def test_nltk_missing_data_logs_debug(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """LookupError do RSLPStemmer gera logger.debug com a exceção."""
    caplog.set_level(logging.DEBUG)
    nltk = _make_nltk_module(stemmer_side_effect=LookupError("rslp missing"))

    with patch.dict(sys.modules, {"spacy": None, "nltk": nltk}, clear=False):
        Lemmatizer()

    assert "Dados do NLTK não estão baixados" in caplog.text
    assert "rslp missing" in caplog.text


def test_selected_backend_logs_info(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Backend selecionado gera logger.info."""
    caplog.set_level(logging.INFO)
    nltk = _make_nltk_module()

    with patch.dict(sys.modules, {"spacy": None, "nltk": nltk}, clear=False):
        Lemmatizer()

    assert "Backend de lematização selecionado: nltk" in caplog.text


def test_no_remaining_except_pass_in_source() -> None:
    """Não há ``except ... : pass`` no módulo de lemmatização."""
    source = inspect.getsource(lem_module)
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        assert "except" not in stripped or ": pass" not in stripped, (
            f"Linha ainda contém except:pass: {line}"
        )


def test_no_pragma_no_cover_in_backend_selection() -> None:
    """A lógica de seleção de backend não usa pragma: no cover."""
    source = inspect.getsource(lem_module)
    assert "# pragma: no cover" not in source
