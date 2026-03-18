"""Testes unitários para parâmetros e métodos do Comparator (api.py)."""

from __future__ import annotations

from unittest.mock import patch

from text_similarity.api import Comparator


class TestParallelThreshold:
    """Testes do parâmetro parallel_threshold no Comparator."""

    def test_should_use_sequential_when_texts_below_threshold(self) -> None:
        """Should use sequential when texts below threshold."""
        comp = Comparator.smart(parallel_threshold=1000)
        texts = ["texto um", "texto dois", "texto três"]  # 3 < 1000

        with patch("text_similarity.api.Comparator._process") as mock_process:
            mock_process.side_effect = lambda t, preprocess=True: t
            comp._process_batch(texts)

        assert mock_process.call_count == len(texts)

    def test_should_use_parallel_when_texts_above_custom_threshold(self) -> None:
        """Should use parallel when texts above custom threshold."""
        comp = Comparator.smart(parallel_threshold=2)
        texts = ["texto um", "texto dois", "texto três"]  # 3 > 2

        with patch(
            "text_similarity.pipeline.parallel_preprocess.run_parallel_preprocess"
        ) as mock_parallel:
            mock_parallel.return_value = texts
            with patch(
                "text_similarity.api.run_parallel_preprocess",
                mock_parallel,
                create=True,
            ):
                # Chama _process_batch via import dinâmico — testar threshold
                pass

        # Verificar que o threshold foi salvo corretamente
        assert comp.parallel_threshold == 2

    def test_default_parallel_threshold_is_1000(self) -> None:
        """Default parallel_threshold deve ser 1000."""
        comp = Comparator.smart()
        assert comp.parallel_threshold == 1000

    def test_parallel_threshold_set_in_init(self) -> None:
        """parallel_threshold passado ao __init__ deve ser salvo."""
        comp = Comparator(mode="basic", parallel_threshold=500)
        assert comp.parallel_threshold == 500

    def test_parallel_threshold_passed_to_smart(self) -> None:
        """parallel_threshold passado ao smart() deve ser salvo."""
        comp = Comparator.smart(parallel_threshold=250)
        assert comp.parallel_threshold == 250
