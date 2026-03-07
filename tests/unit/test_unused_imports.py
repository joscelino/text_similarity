"""Teste para garantir que não existam importações não usadas no projeto."""

import subprocess
from pathlib import Path


def test_unused_imports():
    """Executa o ruff para verificar importações não usadas (F401)."""
    # Determina o caminho para o executável do ruff
    # Tenta usar o ruff do ambiente virtual se disponível
    venv_ruff = Path(".venv/Scripts/ruff.exe")
    if not venv_ruff.exists():
        venv_ruff = Path(".venv/bin/ruff")

    ruff_cmd = str(venv_ruff) if venv_ruff.exists() else "ruff"

    # Executa o ruff check apenas para a regra F401 (Unused Import)
    result = subprocess.run(
        [ruff_cmd, "check", ".", "--select", "F401"],
        capture_output=True,
        text=True,
        encoding="utf-8"
    )

    # Se o exit code for diferente de 0, significa que foram encontradas violações
    if result.returncode != 0:
        pytest_msg = f"Importações não usadas encontradas:\n{result.stdout}"
        assert False, pytest_msg
