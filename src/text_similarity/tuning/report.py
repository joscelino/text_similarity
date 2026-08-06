"""Relatório e visualização dos resultados da calibração.

SEC-STD-004: substituiu chamadas de exibição no terminal por
``logger.info()`` para permitir integração com o pipeline padrão de
logging (handlers, formatters, níveis) e facilitar captura em testes via
``caplog``. Sequências de escape representando nova linha (que antes
surgiam como texto literal) foram corrigidas para quebras de linha
reais.
"""

import logging
from typing import Any, Dict, List

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    HAS_RICH = True
except ImportError:
    HAS_RICH = False

logger = logging.getLogger(__name__)


class CalibrationReport:
    """Relatório contendo os resultados consolidados do Grid Search."""

    def __init__(
        self,
        best_weights: Dict[str, float],
        best_metrics: Dict[str, float],
        all_results: List[Dict[str, Any]],
        worst_errors: List[Dict[str, Any]],
    ):
        """Inicializa o relatório.

        Args:
            best_weights: Os pesos da melhor configuração encontrada.
            best_metrics: F1-Score, Precisão, Recall, MAE e Tempo da melhor config.
            all_results: Lista detalhada de todos os testes iterados.
            worst_errors: Lista dos Falsos Negativos mais discrepantes da melhor config.
        """
        self.best_weights = best_weights
        self.best_metrics = best_metrics
        self.all_results = sorted(
            all_results, key=lambda x: x["metrics"].get("f1_score", 0), reverse=True
        )
        self.worst_errors = worst_errors

    def _print_fallback_table(self) -> None:
        """Fallback markdown se a library `rich` não estiver instalada."""
        logger.info("=== Dashboard de Calibração (Fallback) ===")
        logger.info("🏆 Melhor Configuração: %s", self.best_weights)
        f1 = self.best_metrics.get("f1_score", 0)
        t = self.best_metrics.get("total_time_ms", 0)
        logger.info("Resultados: F1-Score: %.3f | Tempo: %.1fms", f1, t)

        logger.info("--- Histórico de Custo-Benefício ---")
        for res in self.all_results:
            w = res["weights"]
            f1 = res["metrics"].get("f1_score", 0)
            t = res["metrics"].get("total_time_ms", 0)
            logger.info("Pesos: %s -> F1: %.3f | Tempo: %.1fms", w, f1, t)

    def show_time_profiling(self) -> None:
        """Exibe o comparativo de ganho de precisão vs custo de tempo."""
        if not HAS_RICH:
            self._print_fallback_table()
            return

        console = Console()
        table = Table(title="⏱️ Trade-Off: Precisão vs CPU Time", show_lines=True)
        table.add_column("Pesos (Config)", style="cyan", no_wrap=True)
        table.add_column("F1-Score", justify="right", style="green")
        table.add_column("Precisão", justify="right")
        table.add_column("Recall", justify="right")
        table.add_column("Tempo (ms)", justify="right", style="yellow")

        for res in self.all_results:
            w_str = ", ".join(f"{k}: {v:.2f}" for k, v in res["weights"].items())
            m = res["metrics"]
            # Destaque para a campeã
            is_best = res["weights"] == self.best_weights
            marker = "⭐ " if is_best else ""

            table.add_row(
                f"{marker}{w_str}",
                f"{m.get('f1_score', 0):.3f}",
                f"{m.get('precision', 0):.3f}",
                f"{m.get('recall', 0):.3f}",
                f"{m.get('total_time_ms', 0):.1f} ms",
            )

        console.print(table)

    def show_worst_errors(self) -> None:
        """Análise automatizada apontando o ofensor da discrepância."""
        if not self.worst_errors:
            logger.info(
                "Nenhum erro falso negativo detectado na melhor configuração! 🎉"
            )
            return

        if HAS_RICH:
            console = Console()
            msg_hdr = (
                "\n[bold red]🚨 Análise de Discrepância (Falsos Negativos)[/bold red]"
            )
            console.print(msg_hdr)
            for err in self.worst_errors:
                q, t = err["query"], err["target"]
                score = err["predicted_score"]
                explain = err["explain"]

                # Descobrir o ofensor (quem puxou a nota pra baixo)
                # Aquele cujo algoritmo teve o score individual muito mais
                # baixo que seus pares
                details = explain.get("details", {})
                offender = None
                lowest_score = 1.0

                for alg_name, data in details.items():
                    # Ignite entity or complex short-circuits to avoid False offender
                    if not isinstance(data, dict) or "score" not in data:
                        continue
                    if data["score"] < lowest_score:
                        lowest_score = data["score"]
                        offender = alg_name

                msg = (
                    f"[yellow]Query:[/yellow] {q}\n"
                    f"[yellow]Target:[/yellow] {t}\n"
                    f"[white]Score Final:[/white] {score:.3f} (Puxado para baixo)\n"
                )
                if offender:
                    msg += (
                        f"[bold red]► Ofensor Detectado:[/bold red] "
                        f"O algoritmo '{offender}' marcou apenas {lowest_score:.2f}."
                    )

                console.print(
                    Panel(msg, title="Falso Negativo Detalhado", expand=False)
                )
        else:
            logger.info("=== Análise de Discrepância (Falsos Negativos) ===")
            for err in self.worst_errors[:5]:
                q, t = err["query"], err["target"]
                score = err["predicted_score"]
                logger.info("Query: %s | Target: %s | Score: %.3f", q, t, score)

                details = err["explain"].get("details", {})
                offender = None
                lowest_score = 1.0
                for alg_name, data in details.items():
                    if isinstance(data, dict) and "score" in data:
                        if data["score"] < lowest_score:
                            lowest_score = data["score"]
                            offender = alg_name

                if offender:
                    logger.info(
                        "  -> OFENSOR: '%s' marcou apenas %.2f",
                        offender,
                        lowest_score,
                    )

    def summary(self) -> None:
        """Imprime todo o painel consolidado."""
        self.show_time_profiling()
        self.show_worst_errors()
