"""Testes para SEC-LOGIC-005: chave de cache do modelo inclui device e revision.

Verifica que tanto :class:`DenseIndex` quanto :class:`SemanticSimilarity`
usam uma chave de cache composta por ``(model_name, device, revision)`` e
recarregam o modelo quando o ``device`` (ou ``revision``) muda.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from text_similarity.core.dense import DenseIndex
from text_similarity.core.semantic import SemanticSimilarity


@pytest.fixture(autouse=True)
def _reset_global_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Limpa os caches globais antes de cada teste para isolamento."""
    import text_similarity.core.dense as dense_module
    import text_similarity.core.semantic as semantic_module

    monkeypatch.setattr(dense_module, "_DENSE_MODEL", None, raising=False)
    monkeypatch.setattr(dense_module, "_DENSE_MODEL_KEY", None, raising=False)
    monkeypatch.setattr(semantic_module, "_GLOBAL_MODEL", None, raising=False)
    monkeypatch.setattr(semantic_module, "_CURRENT_MODEL_KEY", None, raising=False)
    monkeypatch.setattr(semantic_module, "_SENTENCE_UTIL", None, raising=False)


class TestDenseModelCacheKey:
    """Cache key do DenseIndex leva device e revision em conta."""

    def test_sentence_transformer_receives_revision(self) -> None:
        """Quando informado, ``revision`` é propagado para SentenceTransformer."""
        fake_model = MagicMock(name="fake_dense_model")
        fake_model.encode.return_value = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)

        with patch(
            "sentence_transformers.SentenceTransformer",
            return_value=fake_model,
        ) as mock_st:
            idx = DenseIndex(
                model_name="dummy-model",
                device="cpu",
                revision="abc123",
            )
            idx.fit(["texto exemplo"])

        mock_st.assert_called_once_with(
            "dummy-model",
            device="cpu",
            revision="abc123",
        )

    def test_device_change_reloads_model(self) -> None:
        """Mudança de device invalida o cache e recarrega o modelo."""
        cpu_model = MagicMock(name="cpu_model")
        cpu_model.encode.return_value = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
        cuda_model = MagicMock(name="cuda_model")
        cuda_model.encode.return_value = np.array([[0.4, 0.5, 0.6]], dtype=np.float32)

        with patch(
            "sentence_transformers.SentenceTransformer",
            side_effect=[cpu_model, cuda_model],
        ) as mock_st:
            idx_cpu = DenseIndex(model_name="dummy-model", device="cpu")
            idx_cpu.fit(["texto um"])

            idx_cuda = DenseIndex(model_name="dummy-model", device="cuda")
            idx_cuda.fit(["texto dois"])

        assert mock_st.call_count == 2
        assert mock_st.call_args_list[0] == (
            ("dummy-model",),
            {"device": "cpu"},
        )
        assert mock_st.call_args_list[1] == (
            ("dummy-model",),
            {"device": "cuda"},
        )

        # As instâncias retornadas devem refletir o device solicitado
        assert idx_cpu.device == "cpu"
        assert idx_cuda.device == "cuda"


class TestSemanticModelCacheKey:
    """Cache key do SemanticSimilarity leva device e revision em conta."""

    def test_sentence_transformer_receives_revision(self) -> None:
        """Quando informado, ``revision`` é propagado para SentenceTransformer."""
        fake_model = MagicMock(name="fake_semantic_model")
        fake_util = MagicMock(name="fake_util")
        fake_util.cos_sim.return_value = [[0.99]]

        with (
            patch(
                "sentence_transformers.SentenceTransformer",
                return_value=fake_model,
            ) as mock_st,
            patch(
                "sentence_transformers.util",
                fake_util,
            ),
        ):
            sem = SemanticSimilarity(
                model_name="dummy-model",
                device="cpu",
                revision="sha789",
            )
            _ = sem.compare("carro", "veículo")

        mock_st.assert_called_once_with(
            "dummy-model",
            device="cpu",
            revision="sha789",
        )

    def test_device_change_reloads_model(self) -> None:
        """Após carregar em cuda, pedir em cpu recarrega no device correto."""
        cpu_model = MagicMock(name="cpu_model")
        cpu_model.encode.return_value = [[0.1, 0.2, 0.3]]
        cuda_model = MagicMock(name="cuda_model")
        cuda_model.encode.return_value = [[0.4, 0.5, 0.6]]
        fake_util = MagicMock(name="fake_util")
        fake_util.cos_sim.return_value = [[0.88]]

        with (
            patch(
                "sentence_transformers.SentenceTransformer",
                side_effect=[cuda_model, cpu_model],
            ) as mock_st,
            patch(
                "sentence_transformers.util",
                fake_util,
            ),
        ):
            sem_cuda = SemanticSimilarity(model_name="dummy-model", device="cuda")
            _ = sem_cuda.compare("carro", "automóvel")

            sem_cpu = SemanticSimilarity(model_name="dummy-model", device="cpu")
            _ = sem_cpu.compare("carro", "veículo")

        assert mock_st.call_count == 2
        assert mock_st.call_args_list[0] == (
            ("dummy-model",),
            {"device": "cuda"},
        )
        assert mock_st.call_args_list[1] == (
            ("dummy-model",),
            {"device": "cpu"},
        )

        # A instância retornada deve refletir o device solicitado
        assert sem_cpu.device == "cpu"
