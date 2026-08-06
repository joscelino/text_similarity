"""Testes para a mensagem unificada de instalação do extra [semantic]."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from text_similarity.core import _serialization
from text_similarity.core.dense import DenseIndex
from text_similarity.core.semantic import SemanticSimilarity


@pytest.mark.parametrize(
    "caller",
    ["DenseIndex", "SemanticSimilarity"],
)
def test_install_hint_message_contains_expected_commands(caller: str) -> None:
    """A mensagem de hint contém os comandos pip e uv esperados."""
    message = _serialization.sentence_transformers_install_hint(caller)

    assert caller in message
    assert "pip install text-similarity-br[semantic]" in message
    assert "uv add text-similarity-br[semantic]" in message


def test_dense_index_raises_import_error_with_hint() -> None:
    """DenseIndex levanta ImportError com hint unificado sem sentence-transformers."""
    with patch.dict("sys.modules", {"sentence_transformers": None}):
        with pytest.raises(
            ImportError,
            match="pip install text-similarity-br\\[semantic\\]",
        ):
            DenseIndex(model_name="dummy-model").fit(["texto exemplo"])


def test_semantic_similarity_raises_import_error_with_hint() -> None:
    """SemanticSimilarity levanta ImportError com hint unificado."""
    import text_similarity.core.semantic as semantic_module

    semantic_module._GLOBAL_MODEL = None
    semantic_module._CURRENT_MODEL_KEY = None

    with patch.dict("sys.modules", {"sentence_transformers": None}):
        with pytest.raises(
            ImportError,
            match="uv add text-similarity-br\\[semantic\\]",
        ):
            SemanticSimilarity(model_name="dummy-model").compare("a", "b")
