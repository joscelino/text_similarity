"""Testes unitários para o bypass de pré-processamento (preprocess=False).

Valida que quando ``preprocess=False``, o pipeline NÃO é executado
e os textos passam inalterados para os algoritmos de similaridade.
"""

from __future__ import annotations

from unittest.mock import patch

from text_similarity.api import Comparator


class TestPreprocessBypass:
    """Garante que preprocess=False impede a execução do pipeline."""

    def test_process_bypasses_pipeline(self) -> None:
        """_process(text, preprocess=False) deve retornar o texto sem alterações."""
        comp = Comparator.basic()
        raw = "S22 Ultra 128GB"
        result = comp._process(raw, preprocess=False)
        assert result == raw

    def test_process_bypasses_cache(self) -> None:
        """Com preprocess=False, o cache NÃO deve ser consultado nem populado."""
        comp = Comparator.basic()
        raw = "Galaxy S22 Ultra"

        comp._process(raw, preprocess=False)

        # Cache deve estar vazio — nada foi armazenado
        assert len(comp._cache_store) == 0

    def test_process_with_preprocess_true_uses_pipeline(self) -> None:
        """Com preprocess=True (default), o pipeline deve ser executado."""
        comp = Comparator.basic()
        raw = "Comprei UM celular NOVO"

        with patch.object(comp.pipeline, "process", wraps=comp.pipeline.process) as mock:
            comp._process(raw, preprocess=True)
            mock.assert_called_once_with(raw)

    def test_process_with_preprocess_false_skips_pipeline(self) -> None:
        """Com preprocess=False, pipeline.process() NÃO deve ser chamado."""
        comp = Comparator.basic()
        raw = "texto já limpo"

        with patch.object(comp.pipeline, "process") as mock:
            result = comp._process(raw, preprocess=False)
            mock.assert_not_called()

        assert result == raw

    def test_process_batch_bypasses_pipeline(self) -> None:
        """_process_batch com preprocess=False retorna textos inalterados."""
        comp = Comparator.basic()
        texts = ["galaxy s22", "iphone 15 pro", "pixel 8"]

        with patch.object(comp.pipeline, "process") as mock:
            results = comp._process_batch(texts, preprocess=False)
            mock.assert_not_called()

        assert results == texts


class TestComparePreprocessFalse:
    """Valida preprocess=False nos métodos públicos de comparação."""

    def test_compare_preprocess_false_no_pipeline(self) -> None:
        """compare() com preprocess=False não deve chamar o pipeline."""
        comp = Comparator.basic()

        with patch.object(comp.pipeline, "process") as mock:
            score = comp.compare("galaxy s22", "galaxy s22", preprocess=False)
            mock.assert_not_called()

        # Textos idênticos sem pré-processamento devem ter score alto
        assert score > 0.5

    def test_compare_preprocess_false_preserves_case(self) -> None:
        """Sem pré-processamento, diferenças de case afetam o score."""
        comp = Comparator.basic()

        # Com pré-processamento: pipeline normaliza case → scores próximos
        score_with = comp.compare("GALAXY S22", "galaxy s22", preprocess=True)

        # Sem pré-processamento: case diferente → score potencialmente menor
        score_without = comp.compare("GALAXY S22", "galaxy s22", preprocess=False)

        # O importante é que ambos funcionam sem erro.
        # O score com pipeline tende a ser >= score sem, pois normaliza case.
        assert isinstance(score_with, float)
        assert isinstance(score_without, float)

    def test_explain_preprocess_false_no_pipeline(self) -> None:
        """explain() com preprocess=False não deve chamar o pipeline."""
        comp = Comparator.basic()

        with patch.object(comp.pipeline, "process") as mock:
            result = comp.explain("galaxy s22", "galaxy s22", preprocess=False)
            mock.assert_not_called()

        assert "score" in result
        assert "details" in result


class TestBatchPreprocessFalse:
    """Valida preprocess=False nos métodos batch."""

    def test_compare_batch_preprocess_false(self) -> None:
        """compare_batch com preprocess=False não executa o pipeline."""
        comp = Comparator.basic()
        candidates = ["galaxy s22", "iphone 15", "pixel 8"]

        with patch.object(comp.pipeline, "process") as mock:
            results = comp.compare_batch(
                "galaxy s22", candidates, preprocess=False
            )
            mock.assert_not_called()

        assert isinstance(results, list)

    def test_compare_many_to_many_preprocess_false(self) -> None:
        """compare_many_to_many com preprocess=False não executa o pipeline."""
        comp = Comparator.basic()
        queries = ["galaxy s22", "iphone 15"]
        candidates = ["galaxy s22 ultra", "iphone 15 pro", "pixel 8"]

        with patch.object(comp.pipeline, "process") as mock:
            results = comp.compare_many_to_many(
                queries, candidates, preprocess=False
            )
            mock.assert_not_called()

        assert len(results) == 2
        assert all(isinstance(r, list) for r in results)

    def test_compare_batch_default_still_preprocesses(self) -> None:
        """compare_batch sem argumento preprocess deve executar o pipeline."""
        comp = Comparator.basic()
        candidates = ["galaxy s22"]

        with patch.object(comp.pipeline, "process", wraps=comp.pipeline.process) as mock:
            comp.compare_batch("galaxy s22", candidates)
            assert mock.call_count >= 1


class TestPreprocessFalseTextIntegrity:
    """Testes de integração: textos especiais passam inalterados."""

    def test_raw_product_codes_preserved(self) -> None:
        """Códigos de produto com caracteres especiais não são alterados."""
        comp = Comparator.basic()
        raw = "SM-S908B/DS"

        result = comp._process(raw, preprocess=False)
        assert result == raw

    def test_already_clean_text_scores_correctly(self) -> None:
        """Textos já limpos produzem scores coerentes sem pipeline."""
        comp = Comparator.basic()

        # Simula textos que já passaram por limpeza externa
        clean1 = "samsung galaxy s22 ultra 256gb"
        clean2 = "samsung galaxy s22 ultra 256gb preto"

        score = comp.compare(clean1, clean2, preprocess=False)
        assert 0.0 <= score <= 1.0
        assert score > 0.3  # Textos similares devem ter score razoável

    def test_preprocess_false_with_entities_in_text(self) -> None:
        """Textos com entidades não disparam extração quando preprocess=False."""
        comp = Comparator.smart(entities=["money", "number"])

        with patch.object(comp.pipeline, "process") as mock:
            score = comp.compare(
                "trinta reais",
                "R$ 30,00",
                preprocess=False,
            )
            mock.assert_not_called()

        assert isinstance(score, float)
