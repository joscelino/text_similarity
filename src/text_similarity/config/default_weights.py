"""Perfis de pesos padrão para o :class:`HybridSimilarity`.

Este módulo centraliza os quatro perfis de pesos usados pelo
:class:`text_similarity.api.Comparator` para configurar o
:class:`text_similarity.core.hybrid.HybridSimilarity`. Cada perfil é um
``dict[str, float]`` mapeando o nome do algoritmo (``"cosine"``,
``"edit"``, ``"phonetic"``, ``"entity"``, ``"semantic"``) para o seu peso
relativo na soma ponderada.

Todos os perfis obedecem à invariante ``sum(weights.values()) == 1.0``
validada em nível de módulo via ``assert`` para falhar cedo em caso de
edição incorreta (SEC-DUP-002).

Perfis expostos:
    - :data:`BASIC_WEIGHTS`: modo básico, sem entidades e sem semântica.
    - :data:`BASIC_WEIGHTS_WITH_SEMANTIC`: modo básico + embeddings densos.
    - :data:`SMART_WEIGHTS`: modo inteligente com entidades e fonética.
    - :data:`SMART_WEIGHTS_WITH_SEMANTIC`: modo inteligente + embeddings.
"""

from __future__ import annotations

import math
from types import MappingProxyType
from typing import Mapping

# Perfil BASIC: cosseno + distância de edição, sem fonética.
BASIC_WEIGHTS: Mapping[str, float] = MappingProxyType(
    {
        "cosine": 0.5,
        "edit": 0.5,
        "phonetic": 0.0,
    }
)

# Perfil BASIC com semântica: redistribui espaço para embeddings.
BASIC_WEIGHTS_WITH_SEMANTIC: Mapping[str, float] = MappingProxyType(
    {
        "cosine": 0.3,
        "edit": 0.3,
        "phonetic": 0.0,
        "semantic": 0.4,
    }
)

# Perfil SMART: peso maior para fonética + entidades exatas.
SMART_WEIGHTS: Mapping[str, float] = MappingProxyType(
    {
        "cosine": 0.45,
        "edit": 0.25,
        "phonetic": 0.20,
        "entity": 0.10,
    }
)

# Perfil SMART com semântica: recalibrado quando embeddings são ativados.
SMART_WEIGHTS_WITH_SEMANTIC: Mapping[str, float] = MappingProxyType(
    {
        "cosine": 0.30,
        "edit": 0.15,
        "phonetic": 0.15,
        "entity": 0.10,
        "semantic": 0.30,
    }
)


# Invariante SEC-DUP-002: cada perfil deve somar 1.0 (tolerância padrão do
# math.isclose). Validação em nível de módulo garante falha imediata no
# import se algum peso for editado de forma inconsistente.
assert math.isclose(sum(BASIC_WEIGHTS.values()), 1.0), (
    f"BASIC_WEIGHTS deve somar 1.0, soma atual={sum(BASIC_WEIGHTS.values())}"
)
assert math.isclose(sum(BASIC_WEIGHTS_WITH_SEMANTIC.values()), 1.0), (
    "BASIC_WEIGHTS_WITH_SEMANTIC deve somar 1.0, "
    f"soma atual={sum(BASIC_WEIGHTS_WITH_SEMANTIC.values())}"
)
assert math.isclose(sum(SMART_WEIGHTS.values()), 1.0), (
    f"SMART_WEIGHTS deve somar 1.0, soma atual={sum(SMART_WEIGHTS.values())}"
)
assert math.isclose(sum(SMART_WEIGHTS_WITH_SEMANTIC.values()), 1.0), (
    "SMART_WEIGHTS_WITH_SEMANTIC deve somar 1.0, "
    f"soma atual={sum(SMART_WEIGHTS_WITH_SEMANTIC.values())}"
)


__all__ = [
    "BASIC_WEIGHTS",
    "BASIC_WEIGHTS_WITH_SEMANTIC",
    "SMART_WEIGHTS",
    "SMART_WEIGHTS_WITH_SEMANTIC",
]
