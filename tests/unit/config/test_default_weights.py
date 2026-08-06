"""Testes dos perfis de pesos padrão."""

from __future__ import annotations

import math

import pytest

from text_similarity.config.default_weights import (
    BASIC_WEIGHTS,
    BASIC_WEIGHTS_WITH_SEMANTIC,
    SMART_WEIGHTS,
    SMART_WEIGHTS_WITH_SEMANTIC,
)


@pytest.mark.parametrize(
    "profile_name,profile",
    [
        ("BASIC_WEIGHTS", BASIC_WEIGHTS),
        ("BASIC_WEIGHTS_WITH_SEMANTIC", BASIC_WEIGHTS_WITH_SEMANTIC),
        ("SMART_WEIGHTS", SMART_WEIGHTS),
        ("SMART_WEIGHTS_WITH_SEMANTIC", SMART_WEIGHTS_WITH_SEMANTIC),
    ],
)
def test_profile_sums_to_one(profile_name: str, profile: dict[str, float]) -> None:
    """Cada perfil de pesos deve somar 1.0 (tolerância padrão math.isclose)."""
    total = sum(profile.values())
    assert math.isclose(total, 1.0), f"{profile_name} soma {total}, esperado 1.0"


def test_basic_weights_has_no_semantic() -> None:
    """Perfil BASIC não deve reservar peso para o algoritmo semântico."""
    assert "semantic" not in BASIC_WEIGHTS


def test_basic_weights_with_semantic_has_semantic() -> None:
    """Perfil BASIC + semântica deve declarar o peso 'semantic' > 0."""
    assert "semantic" in BASIC_WEIGHTS_WITH_SEMANTIC
    assert BASIC_WEIGHTS_WITH_SEMANTIC["semantic"] > 0.0


def test_smart_weights_has_entity_and_phonetic() -> None:
    """Perfil SMART deve incluir 'entity' e 'phonetic' com pesos > 0."""
    assert SMART_WEIGHTS["entity"] > 0.0
    assert SMART_WEIGHTS["phonetic"] > 0.0


def test_smart_weights_with_semantic_has_semantic_and_entity() -> None:
    """Perfil SMART + semântica preserva 'entity' e adiciona 'semantic'."""
    assert SMART_WEIGHTS_WITH_SEMANTIC["semantic"] > 0.0
    assert SMART_WEIGHTS_WITH_SEMANTIC["entity"] > 0.0


def test_profiles_are_immutable_by_convention() -> None:
    """As constantes são MappingProxyType (view imutável)."""
    from types import MappingProxyType

    for profile in (
        BASIC_WEIGHTS,
        BASIC_WEIGHTS_WITH_SEMANTIC,
        SMART_WEIGHTS,
        SMART_WEIGHTS_WITH_SEMANTIC,
    ):
        assert isinstance(profile, MappingProxyType)
