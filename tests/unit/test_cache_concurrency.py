"""Testes de thread-safety para o cache in-memory do ``Comparator``.

Estes testes exercitam ``Comparator._process`` (e indiretamente ``compare()``)
sob alta concorrência para verificar que ``self._cache_store`` está protegido
por ``self._cache_lock`` e não sofre race conditions.

Cenários cobertos:
    * Stress test com 8 threads x 200 comparações verificando:
        - ausência de ``RuntimeError`` (ex.: "dictionary changed size during
          iteration");
        - integridade final do cache (chaves consistentes com os textos);
        - ausência de duplicação de trabalho de pipeline (o valor associado a
          cada chave deve ser único e determinístico).
    * ``clear_cache()`` invocado concorrentemente a ``compare()`` não deve
      lançar exceção nem corromper o cache.

Ver: SEC-LOGIC-001 (Sprint "Thread-safety do cache do Comparator").
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import pytest

from text_similarity.api import Comparator

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sample_texts() -> List[str]:
    """Conjunto pequeno de textos PT-BR reutilizados entre threads.

    Um conjunto compacto (10 textos) maximiza a chance de duas threads
    processarem o mesmo texto simultaneamente — expondo qualquer race
    condition no padrão check-then-set de ``_process``.
    """
    return [
        "comprei dois quilos de arroz por trinta reais",
        "adquiri 2kg de arroz pagando R$ 30,00",
        "livro sobre inteligência artificial e aprendizado de máquina",
        "o carro vermelho estava estacionado na rua",
        "reunião marcada para segunda-feira às quatorze horas",
        "receita de bolo de chocolate com cobertura de brigadeiro",
        "python é uma linguagem de programação de alto nível",
        "a temperatura hoje está em vinte e cinco graus celsius",
        "notebook novo com dezesseis gigabytes de memória ram",
        "restaurante italiano no centro da cidade servindo massas frescas",
    ]


@pytest.fixture()
def comparator() -> Comparator:
    """Instância padrão do ``Comparator`` (modo basic + cache habilitado)."""
    return Comparator.basic()


# ---------------------------------------------------------------------------
# Testes de atributos / documentação (contrato do lock)
# ---------------------------------------------------------------------------


class TestCacheLockContract:
    """Garante que o contrato de thread-safety está em vigor na instância."""

    def test_init_creates_cache_lock(self, comparator: Comparator) -> None:
        """``Comparator.__init__`` deve criar ``self._cache_lock``."""
        assert hasattr(comparator, "_cache_lock")
        # ``threading.Lock`` é uma factory function; o tipo interno é
        # ``_thread.lock``. Verificamos comportamento (context manager) em
        # vez do tipo exato para manter portabilidade.
        assert callable(getattr(comparator._cache_lock, "acquire", None))
        assert callable(getattr(comparator._cache_lock, "release", None))
        # Deve funcionar como context manager.
        with comparator._cache_lock:
            pass

    def test_docstring_mentions_thread_safety(self) -> None:
        """A docstring da classe deve documentar thread-safety."""
        doc = Comparator.__doc__ or ""
        assert "thread-safe" in doc.lower()
        assert "threadpoolexecutor" in doc.lower()


# ---------------------------------------------------------------------------
# Stress test principal
# ---------------------------------------------------------------------------


class TestCacheStressUnderThreads:
    """Stress tests de concorrência sobre ``_cache_store``."""

    THREADS = 8
    ITERATIONS_PER_THREAD = 200
    # Janela do cenário clear_cache/compare concorrente, em segundos.
    CONCURRENCY_SECONDS = 3.0

    def test_stress_compare_no_runtime_error_and_cache_integrity(
        self, comparator: Comparator, sample_texts: List[str]
    ) -> None:
        """8 threads x 200 iterações não devem lançar ``RuntimeError``.

        Verifica também que o cache final está íntegro:
        * cada chave presente corresponde a exatamente um texto do
          ``sample_texts``;
        * o valor armazenado é idêntico ao produzido pelo pipeline
          (sem duplicação/corrupção);
        * a barrier garante que todas as threads começam ao mesmo tempo,
          maximizando a probabilidade de contention.
        """
        errors: List[BaseException] = []
        errors_lock = threading.Lock()
        barrier = threading.Barrier(self.THREADS)
        num_texts = len(sample_texts)

        def worker(thread_id: int) -> int:
            """Executa ``ITERATIONS_PER_THREAD`` comparações round-robin."""
            processed_count = 0
            try:
                barrier.wait(timeout=10.0)
                for i in range(self.ITERATIONS_PER_THREAD):
                    idx_a = (thread_id + i) % num_texts
                    idx_b = (thread_id + i + 1) % num_texts
                    text_a = sample_texts[idx_a]
                    text_b = sample_texts[idx_b]
                    score = comparator.compare(text_a, text_b)
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

        # Nenhuma exceção deve ter escapado (em especial RuntimeError por
        # mutação concorrente de dict).
        assert not errors, (
            f"Threads levantaram exceções durante o stress: "
            f"{[type(e).__name__ + ': ' + str(e) for e in errors]}"
        )
        assert sum(results) == self.THREADS * self.ITERATIONS_PER_THREAD

        # Integridade do cache: cada texto exercitado deve estar mapeado
        # para o mesmo resultado produzido por uma execução isolada
        # (sem duplicação de trabalho / valores divergentes).
        assert comparator.cache is not None, "Cache deveria estar habilitado"
        with comparator._cache_lock:
            snapshot = dict(comparator._cache_store)

        # Todos os textos exercitados devem estar no cache.
        expected_keys = {comparator.cache.hash_text(t) for t in sample_texts}
        cached_keys = set(snapshot.keys())
        missing = expected_keys - cached_keys
        assert not missing, (
            f"{len(missing)} chave(s) esperada(s) ausente(s) do cache "
            f"após o stress — indica escrita perdida."
        )

        # Nenhuma chave "fantasma" (chaves só existem via hash_text dos textos).
        extraneous = cached_keys - expected_keys
        assert not extraneous, (
            f"{len(extraneous)} chave(s) inesperada(s) no cache: {extraneous}"
        )

        # O valor armazenado deve bater com o valor determinístico do pipeline.
        # Como o Comparator do stress tem cache preenchido, criamos uma
        # instância limpa para gerar o "ground truth".
        ground_truth_comp = Comparator.basic()
        for text in sample_texts:
            key = ground_truth_comp.cache.hash_text(text)  # type: ignore[union-attr]
            expected = ground_truth_comp._process(text)
            assert snapshot[key] == expected, (
                f"Valor cacheado divergente para o texto {text!r}: "
                f"esperado={expected!r}, cacheado={snapshot[key]!r}"
            )

    def test_clear_cache_concurrent_with_compare_is_safe(
        self, comparator: Comparator, sample_texts: List[str]
    ) -> None:
        """``clear_cache()`` chamado em paralelo com ``compare()`` não deve falhar.

        O objetivo é apenas garantir ausência de ``RuntimeError`` /
        ``KeyError`` decorrentes de mutação concorrente sobre o dict —
        o conteúdo final do cache é intencionalmente não-determinístico
        neste cenário.
        """
        stop_flag = threading.Event()
        errors: List[BaseException] = []
        errors_lock = threading.Lock()
        clears_done = 0
        clears_lock = threading.Lock()

        # Cenário time-boxed em vez de contado. Um alvo de N clears depende da
        # thread do clearer conseguir GIL, e ela disputa com THREADS-1 threads
        # CPU-bound dentro do sklearn; em runner de 2 cores essa thread pode
        # ficar minutos sem avançar. O deadline em wall-clock torna a duração
        # do teste determinística sem enfraquecer a corrida exercitada.
        #
        # A barrier sincroniza a largada: sem ela as threads de compare saem na
        # frente e monopolizam o GIL, e a do clearer pode não ser escalonada
        # nenhuma vez dentro da janela.
        deadline = 0.0
        barrier = threading.Barrier(self.THREADS, timeout=30.0)

        def start_clock() -> None:
            nonlocal deadline
            deadline = time.monotonic() + self.CONCURRENCY_SECONDS

        def keep_running() -> bool:
            return not stop_flag.is_set() and time.monotonic() < deadline

        def compare_worker() -> None:
            try:
                barrier.wait()
                i = 0
                while keep_running():
                    a = sample_texts[i % len(sample_texts)]
                    b = sample_texts[(i + 1) % len(sample_texts)]
                    comparator.compare(a, b)
                    i += 1
            except BaseException as exc:  # noqa: BLE001
                with errors_lock:
                    errors.append(exc)

        def clearer_worker() -> None:
            nonlocal clears_done
            try:
                # Arma o relógio antes de liberar a barrier, para que a janela
                # só comece quando as THREADS threads já estiverem vivas.
                start_clock()
                barrier.wait()
                # Um clear garantido antes de entrar na disputa por GIL: mantém
                # a asserção de cobertura livre de flake sem enfraquecê-la (se
                # esta thread não rodasse, a barrier estouraria e viraria erro).
                comparator.clear_cache()
                with clears_lock:
                    clears_done += 1
                while keep_running():
                    comparator.clear_cache()
                    with clears_lock:
                        clears_done += 1
            except BaseException as exc:  # noqa: BLE001
                with errors_lock:
                    errors.append(exc)

        with ThreadPoolExecutor(max_workers=self.THREADS) as pool:
            futures = [pool.submit(compare_worker) for _ in range(self.THREADS - 1)]
            futures.append(pool.submit(clearer_worker))
            try:
                for fut in futures:
                    fut.result(timeout=self.CONCURRENCY_SECONDS + 30.0)
            finally:
                # Sinaliza a parada em qualquer caminho de saída. Sem isto, um
                # timeout aqui escaparia para o __exit__ do pool, que chama
                # shutdown(wait=True) e faria join em threads ainda girando —
                # travando o processo sem nenhum timeout capaz de interromper.
                stop_flag.set()

        assert not errors, (
            f"Concorrência clear_cache/compare produziu exceções: "
            f"{[type(e).__name__ + ': ' + str(e) for e in errors]}"
        )
        assert clears_done > 0, (
            "clear_cache() nunca executou durante a janela de concorrência; "
            "o cenário de corrida não chegou a ser exercitado."
        )
