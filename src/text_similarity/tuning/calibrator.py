"""Módulo de otimização de pesos (Grid Search) para o Comparator."""

import copy
import time
from typing import Any, Dict, List

from text_similarity.api import Comparator
from text_similarity.tuning.report import CalibrationReport


class WeightCalibrator:
    """Motor de otimização de pesos para encontrar a melhor sintonia Híbrida.

    Executa um Grid Search em uma base Gold Standard previamente anotada,
    explorando diferentes configurações numéricas para o HybridSimilarity
    (`cosine`, `edit`, `phonetic`, `semantic`) e calculando o F1-Score
    e tempo de execução.
    """

    def __init__(
        self,
        comparator: Comparator,
        configurations: List[Dict[str, float]],
        match_threshold: float = 0.65,
    ) -> None:
        """Inicializa o Calibrador resguardando a classe base.

        Args:
            comparator: A instância já configurada (ideal: `Comparator.smart()`).
            configurations: Lista de pesos a serem iteradas.
            match_threshold: Limiar para considerar um teste binário como positivo.
        """
        # Protege contra mutação acidental da instância original
        self.comparator = copy.deepcopy(comparator)
        self.configurations = configurations
        self.match_threshold = match_threshold

    def evaluate(
        self, gold_standard: List[Dict[str, Any]]
    ) -> CalibrationReport:
        """Executa a calibração iterativa garantindo o pré-processamento único.

        Args:
            gold_standard: Dataset esperado no formato:
               [{"query": str, "target": str, "match": bool}]

        Returns:
            CalibrationReport final formatado em terminal ou Markdown.
        """
        if not gold_standard:
            raise ValueError("O Gold Standard não pode estar vazio.")

        # 1. OPTIMIZATION: Extração e Pré-Processamento Único
        # Varre a coleção e cria um cache das strings processadas
        # para evitar milhares de recálculos no SpaCy/Regex
        prep_cache: Dict[str, str] = {}
        for row in gold_standard:
            for key in ["query", "target"]:
                raw_text = row.get(key, "")
                if raw_text not in prep_cache:
                    # Proteção contra Empty Strings solicitada pelo usuário
                    clean_text = raw_text.strip()
                    if clean_text:
                        prep_cache[raw_text] = self.comparator._process(clean_text)
                    else:
                        prep_cache[raw_text] = ""

        # 2. Executar configurações
        all_results = []
        best_f1 = -1.0
        best_weights = {}
        best_metrics: Dict[str, float] = {}
        worst_errors_best_cfg = []

        from text_similarity.core.hybrid import HybridSimilarity

        for cfg in self.configurations:
            # Garante Imutabilidade injetando controladamente no algoritmo replicado
            if isinstance(self.comparator.algorithm, HybridSimilarity):
                self.comparator.algorithm.weights = cfg

            start_t = time.perf_counter()

            # Contadores de Métricas Binárias
            tp, fp, fn, tn = 0, 0, 0, 0
            cfg_errors = []

            for row in gold_standard:
                q_raw = row["query"]
                t_raw = row["target"]
                expected_match = bool(row["match"])

                p_q = prep_cache[q_raw]
                p_t = prep_cache[t_raw]

                # Comparação super rápida via Cache (Não repassa pelo _process)
                if not p_q or not p_t:
                    score = 0.0
                else:
                    score = self.comparator.algorithm.compare(p_q, p_t)

                predicted_match = score >= self.match_threshold

                if predicted_match and expected_match:
                    tp += 1
                elif predicted_match and not expected_match:
                    fp += 1
                elif not predicted_match and expected_match:
                    fn += 1
                    # Guardamos os Falsos Negativos para o Discrepancy Analyzer
                    explain_data = self.comparator.explain(q_raw, t_raw)
                    cfg_errors.append(
                        {
                            "query": q_raw,
                            "target": t_raw,
                            "predicted_score": score,
                            "expected_match": True,
                            "explain": explain_data
                        }
                    )
                else:
                    tn += 1

            total_ms = (time.perf_counter() - start_t) * 1000.0

            # Calculo F1-Score
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            metrics = {
                "f1_score": f1,
                "precision": precision,
                "recall": recall,
                "total_time_ms": total_ms,
            }

            all_results.append({"weights": cfg, "metrics": metrics})

            # Atualiza o Best
            if f1 > best_f1:
                best_f1 = f1
                best_weights = cfg
                best_metrics = metrics
                worst_errors_best_cfg = cfg_errors

        return CalibrationReport(
            best_weights=best_weights,
            best_metrics=best_metrics,
            all_results=all_results,
            worst_errors=worst_errors_best_cfg,
        )
