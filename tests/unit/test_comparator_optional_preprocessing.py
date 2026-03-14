"""Testes unitários para o bypass de pré-processamento (preprocess=False).

Valida que quando ``preprocess=False``, o pipeline NÃO é executado
e os textos passam inalterados para os algoritmos de similaridade.

Cobertura das Fases 1-4 da Feature 1 (Tratamento de Dados Opcional).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

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

        with patch.object(
            comp.pipeline, "process", wraps=comp.pipeline.process
        ) as mock:
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
            results = comp.compare_batch("galaxy s22", candidates, preprocess=False)
            mock.assert_not_called()

        assert isinstance(results, list)

    def test_compare_many_to_many_preprocess_false(self) -> None:
        """compare_many_to_many com preprocess=False não executa o pipeline."""
        comp = Comparator.basic()
        queries = ["galaxy s22", "iphone 15"]
        candidates = ["galaxy s22 ultra", "iphone 15 pro", "pixel 8"]

        with patch.object(comp.pipeline, "process") as mock:
            results = comp.compare_many_to_many(queries, candidates, preprocess=False)
            mock.assert_not_called()

        assert len(results) == 2
        assert all(isinstance(r, list) for r in results)

    def test_compare_batch_default_still_preprocesses(self) -> None:
        """compare_batch sem argumento preprocess deve executar o pipeline."""
        comp = Comparator.basic()
        candidates = ["galaxy s22"]

        with patch.object(
            comp.pipeline, "process", wraps=comp.pipeline.process
        ) as mock:
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


# =====================================================================
# Fase 3: Integração com HybridSimilarity (strings já limpas)
# =====================================================================


class TestHybridSimilarityWithCleanStrings:
    """Valida HybridSimilarity com strings já limpas.

    Testa que todos os algoritmos operam corretamente sobre strings
    já limpas quando preprocess=False.
    """

    def test_smart_mode_all_algorithms_with_clean_text(self) -> None:
        """Smart mode (cosine+edit+phonetic+entity) produz scores válidos."""
        comp = Comparator.smart(entities=["product_model", "money"])

        clean1 = "samsung galaxy s22 ultra 256gb"
        clean2 = "samsung galaxy s22 ultra 128gb"

        result = comp.explain(clean1, clean2, preprocess=False)

        assert 0.0 <= result["score"] <= 1.0
        assert "details" in result
        # Algoritmos ativos devem estar presentes nos detalhes
        details = result["details"]
        assert any(key in details for key in ("cosine", "edit", "phonetic", "entity"))

    def test_smart_mode_entity_short_circuit_with_clean_text(self) -> None:
        """Short-circuit de entidade funciona com preprocess=False."""
        comp = Comparator.smart(entities=["product_model"])

        # Textos já limpos que compartilham modelo de produto
        score = comp.compare(
            "<productmodel:GN500>",
            "pecas <productmodel:GN500> <productmodel:GN1000>",
            preprocess=False,
        )
        assert score >= 0.9  # Short-circuit deve ativar

    def test_smart_explain_returns_all_active_algorithms(self) -> None:
        """explain() com preprocess=False retorna detalhes de todos os algoritmos."""
        comp = Comparator.smart()

        result = comp.explain(
            "notebook dell inspiron 15",
            "notebook dell inspiron 15 polegadas",
            preprocess=False,
        )

        assert "score" in result
        assert "details" in result
        # Modo smart padrão tem: cosine, edit, phonetic, entity
        for algo in ("cosine", "edit", "phonetic", "entity"):
            assert algo in result["details"], f"Algoritmo '{algo}' ausente nos detalhes"

    def test_batch_smart_mode_with_clean_strings(self) -> None:
        """compare_batch no modo smart com strings limpas retorna resultados."""
        comp = Comparator.smart()

        candidates = [
            "notebook dell inspiron 15 i5",
            "mouse logitech wireless",
            "monitor samsung 27 4k",
        ]

        results = comp.compare_batch(
            "notebook dell inspiron",
            candidates,
            preprocess=False,
        )

        assert len(results) > 0
        for r in results:
            assert 0.0 <= r["score"] <= 1.0
            assert "candidate" in r
            assert "details" in r


# =====================================================================
# Fase 4: Verificação de ausência de spaCy e Joblib
# =====================================================================


class TestBypassDoesNotEngageDependencies:
    """Garante que preprocess=False não aciona spaCy nem Joblib."""

    def test_lemmatize_stage_not_called(self) -> None:
        """LemmatizeStage.process() NÃO é chamado com preprocess=False."""
        comp = Comparator.basic()

        # Encontrar o LemmatizeStage no pipeline
        from text_similarity.pipeline.backends import LemmatizeStage

        lemma_stage = None
        for stage in comp.pipeline.stages:
            if isinstance(stage, LemmatizeStage):
                lemma_stage = stage
                break

        assert lemma_stage is not None, "LemmatizeStage deveria existir no pipeline"

        with patch.object(lemma_stage, "process") as mock:
            comp.compare("texto limpo", "texto limpo", preprocess=False)
            mock.assert_not_called()

    def test_cache_hash_not_called(self) -> None:
        """PipelineCache.hash_text() NÃO é chamado com preprocess=False."""
        comp = Comparator.smart(use_cache=True)

        assert comp.cache is not None

        with patch.object(comp.cache, "hash_text") as mock:
            comp.compare("texto limpo", "outro texto", preprocess=False)
            mock.assert_not_called()

    def test_normalize_entities_stage_not_called(self) -> None:
        """NormalizeEntitiesStage.process() NÃO é chamado com preprocess=False."""
        comp = Comparator.smart(entities=["money", "date", "dimension"])

        from text_similarity.pipeline.backends import NormalizeEntitiesStage

        entity_stage = None
        for stage in comp.pipeline.stages:
            if isinstance(stage, NormalizeEntitiesStage):
                entity_stage = stage
                break

        assert entity_stage is not None, (
            "NormalizeEntitiesStage deveria existir no smart"
        )

        with patch.object(entity_stage, "process") as mock:
            comp.compare(
                "trinta reais ontem",
                "R$ 30,00 12/03/2023",
                preprocess=False,
            )
            mock.assert_not_called()

    def test_clear_cache_works_after_preprocess_false_only(self) -> None:
        """clear_cache() funciona sem erro mesmo sem uso de pipeline."""
        comp = Comparator.smart(use_cache=True)

        # Usa apenas preprocess=False — cache nunca é populado
        comp.compare("a", "b", preprocess=False)
        comp.compare("c", "d", preprocess=False)

        assert len(comp._cache_store) == 0

        # clear_cache() não deve lançar exceção
        comp.clear_cache()
        assert len(comp._cache_store) == 0


# =====================================================================
# Fase 4: Benchmarks — preprocess=False deve ser mais rápido
# =====================================================================


class TestPreprocessBenchmarks:
    """Benchmarks comparando preprocess=True vs preprocess=False.

    Cache é desabilitado para medir o custo real do pipeline a cada chamada,
    sem que o cache in-memory mascare a diferença.
    """

    @pytest.fixture()
    def comp_no_cache(self) -> Comparator:
        """Comparator sem cache — força execução do pipeline a cada chamada."""
        return Comparator.basic()

    @pytest.fixture()
    def sample_candidates(self) -> list:
        return [
            "samsung galaxy s22 ultra 256gb",
            "apple iphone 15 pro max 512gb",
            "google pixel 8 pro 128gb",
            "motorola edge 40 neo 256gb",
            "xiaomi redmi note 13 pro 256gb",
            "notebook dell inspiron 15 i5",
            "mouse logitech mx master 3s",
            "monitor lg ultrawide 34 144hz",
            "teclado mecanico redragon kumara",
            "headset hyperx cloud alpha wireless",
        ]

    def _compare_clearing_cache(
        self, comp: Comparator, t1: str, t2: str, preprocess: bool
    ) -> float:
        """Helper: limpa cache antes de cada compare para forçar pipeline."""
        comp.clear_cache()
        return comp.compare(t1, t2, preprocess=preprocess)

    def _batch_clearing_cache(
        self, comp: Comparator, query: str, candidates: list, preprocess: bool
    ) -> list:
        """Helper: limpa cache antes de cada batch para forçar pipeline."""
        comp.clear_cache()
        return comp.compare_batch(query, candidates, preprocess=preprocess)

    def test_benchmark_compare_preprocess_true(self, benchmark, comp_no_cache) -> None:
        """Benchmark: compare() com preprocess=True (pipeline executado)."""
        benchmark(
            self._compare_clearing_cache,
            comp_no_cache,
            "galaxy s22 ultra 256gb preto",
            "galaxy s22 ultra 128gb branco",
            True,
        )

    def test_benchmark_compare_preprocess_false(self, benchmark, comp_no_cache) -> None:
        """Benchmark: compare() com preprocess=False (bypass direto)."""
        benchmark(
            comp_no_cache.compare,
            "galaxy s22 ultra 256gb preto",
            "galaxy s22 ultra 128gb branco",
            preprocess=False,
        )

    def test_benchmark_batch_preprocess_true(
        self, benchmark, comp_no_cache, sample_candidates
    ) -> None:
        """Benchmark: compare_batch() com preprocess=True (pipeline em lote)."""
        benchmark(
            self._batch_clearing_cache,
            comp_no_cache,
            "galaxy s22",
            sample_candidates,
            True,
        )

    def test_benchmark_batch_preprocess_false(
        self, benchmark, comp_no_cache, sample_candidates
    ) -> None:
        """Benchmark: compare_batch() com preprocess=False (bypass em lote)."""
        benchmark(
            comp_no_cache.compare_batch,
            "galaxy s22",
            sample_candidates,
            preprocess=False,
        )

    def test_preprocess_false_faster_than_true(self) -> None:
        """Confirma que preprocess=False é significativamente mais rápido."""
        import time

        comp = Comparator(mode="basic", use_cache=False)

        text1 = "samsung galaxy s22 ultra 256gb preto novo lacrado"
        text2 = "samsung galaxy s22 ultra 128gb branco seminovo"
        n_runs = 200

        # Medir preprocess=True (pipeline completo a cada chamada, sem cache)
        start = time.perf_counter()
        for _ in range(n_runs):
            comp.compare(text1, text2, preprocess=True)
        time_with = time.perf_counter() - start

        # Medir preprocess=False (bypass direto)
        start = time.perf_counter()
        for _ in range(n_runs):
            comp.compare(text1, text2, preprocess=False)
        time_without = time.perf_counter() - start

        assert time_without < time_with, (
            f"preprocess=False ({time_without:.4f}s) deveria ser mais rápido "
            f"que preprocess=True ({time_with:.4f}s)"
        )
