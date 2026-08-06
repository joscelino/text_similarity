"""Testes do dataclass ``ComparatorConfig`` e integração com o Comparator."""

from __future__ import annotations

import math

import pytest

from text_similarity.api import Comparator
from text_similarity.config import (
    BASIC_WEIGHTS,
    BASIC_WEIGHTS_WITH_SEMANTIC,
    SMART_WEIGHTS,
    SMART_WEIGHTS_WITH_SEMANTIC,
    ComparatorConfig,
)


class TestComparatorConfigDefaults:
    """Valores default do dataclass."""

    def test_defaults_are_basic_mode(self) -> None:
        cfg = ComparatorConfig()
        assert cfg.mode == "basic"
        assert cfg.use_embeddings is False
        assert cfg.use_cache is True
        assert cfg.fusion_strategy == "linear"
        assert cfg.indexing_strategy == "tfidf"
        assert cfg.weights is None
        assert cfg.strict is True

    def test_can_construct_smart_config(self) -> None:
        cfg = ComparatorConfig(mode="smart", entities=["date"], use_embeddings=True)
        assert cfg.mode == "smart"
        assert cfg.entities == ["date"]
        assert cfg.use_embeddings is True


class TestComparatorConfigValidation:
    """Validação de campos em ``__post_init__``."""

    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="mode"):
            ComparatorConfig(mode="invalid")

    def test_weights_sum_not_one_raises(self) -> None:
        with pytest.raises(ValueError, match="somar 1.0"):
            ComparatorConfig(weights={"cosine": 0.3, "edit": 0.3})

    def test_weights_sum_one_ok(self) -> None:
        cfg = ComparatorConfig(weights={"cosine": 0.7, "edit": 0.3})
        assert cfg.weights == {"cosine": 0.7, "edit": 0.3}

    def test_weights_sum_close_to_one_ok(self) -> None:
        """Tolerância via math.isclose (default)."""
        weights = {"a": 0.1 + 0.2, "b": 0.7}  # 0.3 + 0.7 mas 0.1+0.2 = 0.30000...4
        cfg = ComparatorConfig(weights=weights)
        assert math.isclose(sum(cfg.weights.values()), 1.0)


def _assert_weights_close(actual: dict[str, float], expected: dict[str, float]) -> None:
    """Compara dois dicts de pesos com tolerância de float.

    HybridSimilarity pode normalizar internamente (dividindo por sum), o
    que introduz ruído de ponto flutuante — precisamos comparar com
    ``math.isclose`` em vez de igualdade exata.
    """
    assert set(actual.keys()) == set(expected.keys()), (
        f"chaves diferem: atual={set(actual.keys())} esperado={set(expected.keys())}"
    )
    for key, value in expected.items():
        assert math.isclose(actual[key], value, abs_tol=1e-9), (
            f"peso '{key}': atual={actual[key]}, esperado={value}"
        )


class TestComparatorPresets:
    """Presets basic/smart devem construir ComparatorConfig equivalente."""

    def test_basic_preset_uses_basic_weights(self) -> None:
        comp = Comparator.basic()
        assert comp.mode == "basic"
        assert comp.use_embeddings is False
        _assert_weights_close(comp.algorithm.weights, dict(BASIC_WEIGHTS))

    def test_smart_preset_uses_smart_weights(self) -> None:
        comp = Comparator.smart()
        assert comp.mode == "smart"
        assert comp.use_embeddings is False
        _assert_weights_close(comp.algorithm.weights, dict(SMART_WEIGHTS))

    def test_smart_with_embeddings_uses_semantic_weights(self) -> None:
        comp = Comparator.smart(use_embeddings=True)
        assert comp.use_embeddings is True
        _assert_weights_close(comp.algorithm.weights, dict(SMART_WEIGHTS_WITH_SEMANTIC))

    def test_basic_via_init_with_embeddings(self) -> None:
        comp = Comparator(mode="basic", use_embeddings=True)
        _assert_weights_close(comp.algorithm.weights, dict(BASIC_WEIGHTS_WITH_SEMANTIC))


class TestComparatorConfigInjection:
    """Comparator aceita ``config=`` OU parâmetros individuais."""

    def test_config_object_is_stored(self) -> None:
        cfg = ComparatorConfig(mode="smart", use_embeddings=True)
        comp = Comparator(config=cfg)
        assert comp.config is cfg
        assert comp.mode == "smart"

    def test_individual_params_still_work(self) -> None:
        """Chamada retro-compatível: Comparator(mode='basic')."""
        comp = Comparator(mode="basic")
        assert comp.mode == "basic"

    def test_config_overrides_individual_params(self) -> None:
        """Se ``config=`` for passado, os demais args são ignorados."""
        cfg = ComparatorConfig(mode="smart")
        comp = Comparator(mode="basic", config=cfg)
        assert comp.mode == "smart"


class TestWeightsOverride:
    """Override do perfil default via ``weights=``."""

    def test_config_weights_override_default_profile(self) -> None:
        custom = {"cosine": 0.6, "edit": 0.4}
        cfg = ComparatorConfig(mode="basic", weights=custom)
        comp = Comparator(config=cfg)
        assert comp.algorithm.weights == custom

    def test_weights_via_init_override_default(self) -> None:
        custom = {"cosine": 0.8, "edit": 0.2}
        comp = Comparator(mode="basic", weights=custom)
        assert comp.algorithm.weights == custom

    def test_weights_override_ignores_use_embeddings(self) -> None:
        """Weights explícito tem precedência sobre a escolha por use_embeddings."""
        custom = {"cosine": 0.5, "edit": 0.5}
        comp = Comparator(mode="smart", use_embeddings=True, weights=custom)
        assert comp.algorithm.weights == custom


class TestBackwardCompatibility:
    """Equivalência semântica entre a API anterior e a nova."""

    def test_smart_use_embeddings_true_equivalence(self) -> None:
        """Comparator.smart(use_embeddings=True) mantém pesos anteriores."""
        comp = Comparator.smart(use_embeddings=True)
        _assert_weights_close(
            comp.algorithm.weights,
            {
                "cosine": 0.30,
                "edit": 0.15,
                "phonetic": 0.15,
                "entity": 0.10,
                "semantic": 0.30,
            },
        )

    def test_smart_default_equivalence(self) -> None:
        comp = Comparator.smart()
        _assert_weights_close(
            comp.algorithm.weights,
            {
                "cosine": 0.45,
                "edit": 0.25,
                "phonetic": 0.20,
                "entity": 0.10,
            },
        )

    def test_basic_default_equivalence(self) -> None:
        comp = Comparator.basic()
        _assert_weights_close(
            comp.algorithm.weights,
            {
                "cosine": 0.5,
                "edit": 0.5,
                "phonetic": 0.0,
            },
        )
