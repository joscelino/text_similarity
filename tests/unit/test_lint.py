import subprocess
import sys


def test_ruff_linting() -> None:
    """Verifica se o código está nos padrões do Ruff (Lint)."""
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "src", "tests"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"Falha no Ruff Lint:\n{result.stdout}\n{result.stderr}"
    )


def test_ruff_formatting() -> None:
    """Verifica a formatação correta do código pelo Ruff."""
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "format", "--check", "src", "tests"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"Falha no Ruff Format:\n{result.stdout}\n{result.stderr}"
    )
