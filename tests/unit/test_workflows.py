"""Testes de regressão para os workflows de CI/CD em .github/workflows."""

import re
from pathlib import Path

WORKFLOWS_DIR = Path(__file__).resolve().parents[2] / ".github" / "workflows"

USES_PATTERN = re.compile(r"uses:\s*([\w.-]+/[\w./-]+)@([^\s#]+)")

# O setup-uv nao publica tags "major" (ex.: v9); so tags completas (ex.: v9.0.0).
# Regressao: o CI falhava com "Unable to resolve action `astral-sh/setup-uv@v9`".
FULL_SEMVER_TAG = re.compile(r"^v\d+\.\d+\.\d+$")


def _collect_actions() -> list[tuple[str, str, str]]:
    """Retorna tuplas (arquivo, action, ref) de todos os workflows."""
    actions = []
    workflow_files = sorted(WORKFLOWS_DIR.glob("*.y*ml"))
    assert workflow_files, f"Nenhum workflow encontrado em {WORKFLOWS_DIR}"
    for workflow_file in workflow_files:
        content = workflow_file.read_text(encoding="utf-8")
        for action, ref in USES_PATTERN.findall(content):
            actions.append((workflow_file.name, action, ref))
    return actions


def test_all_actions_have_a_ref() -> None:
    """Toda action referenciada com 'uses:' deve especificar uma ref (tag/SHA)."""
    actions = _collect_actions()
    assert actions, "Nenhuma action 'uses:' encontrada nos workflows"
    for filename, action, ref in actions:
        assert ref.strip(), f"{filename}: action '{action}' sem ref definida"


def test_setup_uv_is_pinned_to_full_semver() -> None:
    """astral-sh/setup-uv deve usar tag completa (vX.Y.Z), nao alias major (vX)."""
    actions = _collect_actions()
    setup_uv_refs = [
        (filename, ref)
        for filename, action, ref in actions
        if action == "astral-sh/setup-uv"
    ]
    assert setup_uv_refs, "astral-sh/setup-uv nao encontrado nos workflows"
    for filename, ref in setup_uv_refs:
        assert FULL_SEMVER_TAG.match(ref), (
            f"{filename}: 'astral-sh/setup-uv@{ref}' invalida. "
            "O setup-uv nao publica tags major (ex.: v9); "
            "use a tag completa (ex.: v9.0.0)."
        )
