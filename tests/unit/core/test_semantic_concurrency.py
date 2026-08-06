"""Testes de thread-safety para o cache global de ``SemanticSimilarity``.

Estes testes exercitam ``SemanticSimilarity._ensure_model_loaded`` sob alta
concorrência para verificar que o padrão Double-Checked Locking protege o
carregamento do modelo SentenceTransformer sem race conditions.

Cenários cobertos:
    * Stress test com 8 threads x 200 comparações verificando:
        - ausência de ``RuntimeError`` / exceções concorrentes;
        - carregamento do modelo ocorre uma única vez (DCL);
        - chave de cache global reflete ``(model_name, device, revision)``.

Ver: SEC-LOGIC-006 (Sprint "Sincronizar worktree elastic-bartik com DCL do main").
"""

from __future__ import annotations

import threading
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
from unittest.mock import MagicMock, patch

import pytest

import text_similarity.core.semantic as semantic_module
from text_similarity.core.semantic import SemanticSimilarity


@pytest.fixture(autouse=True)
def _reset_global_caches() -> None:
    """Limpa os caches globais antes de cada teste para isolamento."""
    semantic_module._GLOBAL_MODEL = None
    semantic_module._CURRENT_MODEL_KEY = None
    semantic_module._SENTENCE_UTIL = None


@pytest.fixture()
def mocked_semantic_backend() -> Generator[MagicMock, None, None]:
    """Mocka SentenceTransformer e util.cos_sim para testes de concorrência."""
    fake_model = MagicMock(name="fake_sentence_transformer")
    fake_model.encode.return_value = [[0.9, 0.1, 0.0]]
    fake_util = MagicMock(name="fake_sentence_util")
    fake_util.cos_sim.return_value = [[0.88]]

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
        yield mock_st


# ---------------------------------------------------------------------------
# Testes de atributos / documentação (contrato do lock)
# ---------------------------------------------------------------------------


class TestSemanticLockContract:
    """Garante que o contrato de thread-safety está em vigor no módulo."""

    def test_module_has_model_lock(self) -> None:
        """O módulo semantic deve expor ``_MODEL_LOCK`` (threading.Lock)."""
        assert hasattr(threading, "Lock")
        assert callable(getattr(semantic_module._MODEL_LOCK, "acquire", None))
        assert callable(getattr(semantic_module._MODEL_LOCK, "release", None))
        with semantic_module._MODEL_LOCK:
            pass

    def test_ensure_model_loaded_uses_double_checked_locking(
        self, mocked_semantic_backend: MagicMock
    ) -> None:
        """Duas chamadas concorrentes devem resultar em exatamente um carregamento."""
        sem = SemanticSimilarity(model_name="dummy-model", device="cpu", revision="abc")

        # Precondição: caches limpos
        assert semantic_module._GLOBAL_MODEL is None
        assert semantic_module._CURRENT_MODEL_KEY is None

        sem._ensure_model_loaded()
        sem._ensure_model_loaded()

        mocked_semantic_backend.assert_called_once()
        assert semantic_module._CURRENT_MODEL_KEY == ("dummy-model", "cpu", "abc")


# ---------------------------------------------------------------------------
# Stress test principal
# ---------------------------------------------------------------------------


class TestSemanticStressUnderThreads:
    """Stress tests de concorrência sobre o cache global do modelo semântico."""

    THREADS = 8
    ITERATIONS_PER_THREAD = 200

    def test_stress_ensure_model_loaded_no_race(
        self, mocked_semantic_backend: MagicMock
    ) -> None:
        """8 threads x 200 iterações não devem lançar exceção nem recarregar modelo.

        Verifica:
        * ausência de ``RuntimeError`` ou outras exceções concorrentes;
        * ``SentenceTransformer`` é instanciado uma única vez (DCL);
        * ``_CURRENT_MODEL_KEY`` é atribuída ANTES de ``_GLOBAL_MODEL`` e
          reflete a tupla ``(model_name, device, revision)``.
        """
        errors: List[BaseException] = []
        errors_lock = threading.Lock()
        barrier = threading.Barrier(self.THREADS)

        def worker(thread_id: int) -> int:
            """Executa ``ITERATIONS_PER_THREAD`` comparações."""
            processed_count = 0
            try:
                barrier.wait(timeout=10.0)
                sem = SemanticSimilarity(
                    model_name="stress-model",
                    device="cpu",
                    revision="stress-rev",
                    strict=False,
                )
                for i in range(self.ITERATIONS_PER_THREAD):
                    score = sem.compare(
                        f"texto A {thread_id}-{i}", f"texto B {thread_id}-{i}"
                    )
                    # Sanity check: score deve estar em [0, 1].
                    assert 0.0 <= score <= 1.0
                    processed_count += 1
            except BaseException as exc:  # noqa: BLE001 - queremos capturar tudo
                with errors_lock:
                    errors.append(exc)
            return processed_count

        with ThreadPoolExecutor(max_workers=self.THREADS) as pool:
            futures = [pool.submit(worker, tid) for tid in range(self.THREADS)]
            results = [f.result(timeout=60.0) for f in as_completed(futures)]

        assert not errors, (
            f"Threads levantaram exceções durante o stress: "
            f"{[type(e).__name__ + ': ' + str(e) for e in errors]}"
        )
        assert sum(results) == self.THREADS * self.ITERATIONS_PER_THREAD

        # O modelo deve ter sido carregado exatamente uma vez graças ao DCL.
        assert mocked_semantic_backend.call_count == 1, (
            f"DCL falhou: SentenceTransformer foi chamado "
            f"{mocked_semantic_backend.call_count} vez(es) em vez de 1."
        )

        # A chave de cache deve refletir a configuração solicitada.
        assert semantic_module._CURRENT_MODEL_KEY == (
            "stress-model",
            "cpu",
            "stress-rev",
        ), f"Chave de cache incorreta após stress: {semantic_module._CURRENT_MODEL_KEY}"

        # Invariante crítico do DCL: _CURRENT_MODEL_KEY nunca deve estar
        # atribuída com _GLOBAL_MODEL ainda None (evita outras threads
        # retornarem um modelo de chave errada no fast-path).
        assert (semantic_module._GLOBAL_MODEL is None) == (
            semantic_module._CURRENT_MODEL_KEY is None
        ), (
            "Invariante DCL violado: _GLOBAL_MODEL e _CURRENT_MODEL_KEY "
            "devem ser ambos None ou ambos definidos."
        )
