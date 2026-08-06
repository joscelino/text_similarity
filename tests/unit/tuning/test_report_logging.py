r"""Testes de SEC-STD-004: uso correto de ``logging`` em ``tuning/report.py``.

Garante que ``CalibrationReport`` (a) emite mensagens via
``logging.getLogger(__name__)`` no lugar de ``print()`` e (b) já não
contém escapes literais ``\\n`` em suas strings de saída (foram
corrigidos para quebras de linha reais ``\n``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import pytest

from text_similarity.tuning import report as report_module
from text_similarity.tuning.report import CalibrationReport

LOGGER_NAME = "text_similarity.tuning.report"


def _make_report(worst_errors: List[Dict[str, Any]] | None = None) -> CalibrationReport:
    """Fabrica um :class:`CalibrationReport` mínimo para os testes."""
    best_weights = {"cosine": 1.0}
    best_metrics = {
        "f1_score": 0.987,
        "precision": 0.99,
        "recall": 0.98,
        "total_time_ms": 12.3,
    }
    all_results = [
        {
            "weights": {"cosine": 1.0},
            "metrics": {
                "f1_score": 0.987,
                "precision": 0.99,
                "recall": 0.98,
                "total_time_ms": 12.3,
            },
        },
        {
            "weights": {"edit": 1.0},
            "metrics": {
                "f1_score": 0.5,
                "precision": 0.4,
                "recall": 0.6,
                "total_time_ms": 22.0,
            },
        },
    ]
    return CalibrationReport(
        best_weights=best_weights,
        best_metrics=best_metrics,
        all_results=all_results,
        worst_errors=worst_errors or [],
    )


# ---------------------------------------------------------------------
# Logger existe e está configurado
# ---------------------------------------------------------------------


def test_module_defines_logger() -> None:
    """O módulo deve expor um ``logger = logging.getLogger(__name__)``."""
    assert hasattr(report_module, "logger")
    assert isinstance(report_module.logger, logging.Logger)
    assert report_module.logger.name == LOGGER_NAME


# ---------------------------------------------------------------------
# Ausência de print() builtin — verificação estática por grep no fonte
# ---------------------------------------------------------------------


def test_no_builtin_print_calls_in_source() -> None:
    """Nenhuma chamada ``print(`` de builtin deve permanecer no fonte.

    Chamadas ``console.print(...)`` do Rich são permitidas e ignoradas.
    """
    src_path = Path(report_module.__file__)
    src = src_path.read_text(encoding="utf-8")

    for lineno, line in enumerate(src.splitlines(), start=1):
        stripped = line.lstrip()
        # Ignorar chamadas do Rich (console.print) e docstrings.
        if "print(" not in stripped:
            continue
        # Se for uma menção em docstring/comentário sobre "print()", ignora.
        if stripped.startswith(("#", '"', "'")):
            continue
        # Chamada real do Rich Console (permitida).
        if "console.print(" in stripped or ".print(" in stripped:
            continue
        pytest.fail(f"print() builtin encontrado em {src_path}:{lineno}: {line!r}")


# ---------------------------------------------------------------------
# Nenhum escape literal '\n' remanescente (bug corrigido)
# ---------------------------------------------------------------------


def test_no_literal_backslash_n_in_source() -> None:
    r"""Não pode haver o escape literal ``\\\\n`` (dois caracteres) nas strings."""
    src_path = Path(report_module.__file__)
    src = src_path.read_text(encoding="utf-8")
    # O padrão ``\\n`` (representando os dois caracteres barra invertida + n)
    # não pode aparecer em nenhuma string, exceto em docstring que documente
    # o próprio motivo.
    for lineno, line in enumerate(src.splitlines(), start=1):
        if r"\\n" in line and "SEC-STD-004" not in line and "corrigidos" not in line:
            pytest.fail(
                f"Escape literal '\\\\n' encontrado em {src_path}:{lineno}: {line!r}"
            )


# ---------------------------------------------------------------------
# Testes funcionais: caplog captura as mensagens de logging
# ---------------------------------------------------------------------


def test_fallback_table_logs_via_logger(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_print_fallback_table`` deve emitir INFO logs — não usar ``print()``."""
    # Força o caminho de fallback mesmo que rich esteja instalado.
    monkeypatch.setattr(report_module, "HAS_RICH", False)

    rep = _make_report()

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        rep.show_time_profiling()

    messages = [rec.getMessage() for rec in caplog.records if rec.name == LOGGER_NAME]
    assert any("Dashboard de Calibração (Fallback)" in m for m in messages)
    assert any("Melhor Configuração" in m for m in messages)
    assert any("F1-Score" in m for m in messages)
    assert any("Histórico de Custo-Benefício" in m for m in messages)


def test_show_worst_errors_empty_logs_success(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Lista vazia de erros deve logar a mensagem 'Nenhum erro...'."""
    rep = _make_report(worst_errors=[])

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        rep.show_worst_errors()

    messages = [rec.getMessage() for rec in caplog.records if rec.name == LOGGER_NAME]
    assert any("Nenhum erro falso negativo" in m for m in messages)


def test_show_worst_errors_fallback_logs_details(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No fallback (sem Rich), detalhes de falsos negativos vão pelo logger."""
    monkeypatch.setattr(report_module, "HAS_RICH", False)

    worst = [
        {
            "query": "cadeira gamer",
            "target": "cadeira de escritório",
            "predicted_score": 0.42,
            "explain": {
                "details": {
                    "cosine": {"score": 0.9},
                    "edit": {"score": 0.1},  # ofensor
                    "phonetic": {"score": 0.7},
                }
            },
        }
    ]
    rep = _make_report(worst_errors=worst)

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        rep.show_worst_errors()

    messages = [rec.getMessage() for rec in caplog.records if rec.name == LOGGER_NAME]
    joined = " | ".join(messages)
    assert "cadeira gamer" in joined
    assert "cadeira de escritório" in joined
    assert "OFENSOR" in joined
    assert "edit" in joined
