"""Configuração centralizada do Comparator.

Este pacote reúne todas as constantes de configuração e dataclasses de
configuração usadas pelo :class:`text_similarity.api.Comparator`,
eliminando a duplicação de literais de pesos e a duplicação de defaults
entre ``Comparator.__init__``, ``Comparator.basic`` e ``Comparator.smart``.

Public API:
    - :data:`BASIC_WEIGHTS`
    - :data:`BASIC_WEIGHTS_WITH_SEMANTIC`
    - :data:`SMART_WEIGHTS`
    - :data:`SMART_WEIGHTS_WITH_SEMANTIC`
    - :class:`ComparatorConfig`
"""

from __future__ import annotations

from text_similarity.config.comparator_config import ComparatorConfig
from text_similarity.config.default_weights import (
    BASIC_WEIGHTS,
    BASIC_WEIGHTS_WITH_SEMANTIC,
    SMART_WEIGHTS,
    SMART_WEIGHTS_WITH_SEMANTIC,
)

__all__ = [
    "BASIC_WEIGHTS",
    "BASIC_WEIGHTS_WITH_SEMANTIC",
    "SMART_WEIGHTS",
    "SMART_WEIGHTS_WITH_SEMANTIC",
    "ComparatorConfig",
]
