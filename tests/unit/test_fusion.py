"""Testes unitários para o módulo RRFusion (Reciprocal Rank Fusion)."""

from __future__ import annotations

import pytest

from text_similarity.core.fusion import RRFusion


class TestRRFusionBasic:
    """Testes fundamentais da lógica RRF."""

    def test_single_ranking_preserves_order(self) -> None:
        """Um único ranking deve manter a ordem original."""
        rrf = RRFusion(k=60)
        rankings = [
            [
                {"candidate": "A", "score": 0.9},
                {"candidate": "B", "score": 0.7},
                {"candidate": "C", "score": 0.3},
            ]
        ]
        result = rrf.fuse(rankings, ["cosine"])

        assert result[0]["candidate"] == "A"
        assert result[1]["candidate"] == "B"
        assert result[2]["candidate"] == "C"

    def test_top_ranked_everywhere_gets_score_one(self) -> None:
        """Candidato #1 em todos os rankings deve receber score normalizado 1.0."""
        rrf = RRFusion(k=60)
        rankings = [
            [
                {"candidate": "A", "score": 0.9},
                {"candidate": "B", "score": 0.5},
            ],
            [
                {"candidate": "A", "score": 0.8},
                {"candidate": "B", "score": 0.3},
            ],
        ]
        result = rrf.fuse(rankings, ["cosine", "edit"])

        assert result[0]["candidate"] == "A"
        assert result[0]["score"] == pytest.approx(1.0)

    def test_consistent_top_beats_inconsistent(self) -> None:
        """Candidato no topo vence candidato #1 em apenas um ranking."""
        rrf = RRFusion(k=60)
        rankings = [
            [
                {"candidate": "A", "score": 0.95},
                {"candidate": "B", "score": 0.50},
                {"candidate": "C", "score": 0.30},
            ],
            [
                {"candidate": "B", "score": 0.90},
                {"candidate": "A", "score": 0.85},
                {"candidate": "C", "score": 0.10},
            ],
            [
                {"candidate": "A", "score": 0.80},
                {"candidate": "B", "score": 0.60},
                {"candidate": "C", "score": 0.40},
            ],
        ]
        result = rrf.fuse(rankings, ["cosine", "edit", "phonetic"])

        # A: ranks 1,2,1 -> B: ranks 2,1,2 -> A deve vencer (2x rank 1 vs 1x rank 1)
        assert result[0]["candidate"] == "A"
        assert result[1]["candidate"] == "B"

    def test_rrf_formula_correctness(self) -> None:
        """Verifica o cálculo RRF manualmente com k=60."""
        rrf = RRFusion(k=60)
        rankings = [
            [
                {"candidate": "X", "score": 0.9},
                {"candidate": "Y", "score": 0.5},
            ],
            [
                {"candidate": "Y", "score": 0.8},
                {"candidate": "X", "score": 0.3},
            ],
        ]
        result = rrf.fuse(rankings, ["algo1", "algo2"])

        # X: rank 1 em algo1 (1/61) + rank 2 em algo2 (1/62)
        expected_x = (1 / 61 + 1 / 62)
        # Y: rank 2 em algo1 (1/62) + rank 1 em algo2 (1/61)
        expected_y = (1 / 62 + 1 / 61)
        # Ambos devem ter o mesmo score (simétrico)
        max_rrf = 2 / 61  # 2 algoritmos, rank 1 em ambos

        x_result = next(r for r in result if r["candidate"] == "X")
        y_result = next(r for r in result if r["candidate"] == "Y")

        assert x_result["score"] == pytest.approx(expected_x / max_rrf)
        assert y_result["score"] == pytest.approx(expected_y / max_rrf)
        assert x_result["score"] == pytest.approx(y_result["score"])


class TestRRFusionDetails:
    """Testes do formato de saída e detalhes."""

    def test_output_contains_fusion_key(self) -> None:
        """Cada resultado deve conter 'fusion': 'rrf'."""
        rrf = RRFusion(k=60)
        rankings = [[{"candidate": "A", "score": 0.9}]]
        result = rrf.fuse(rankings, ["cosine"])

        assert result[0]["fusion"] == "rrf"

    def test_details_contain_rank_and_contribution(self) -> None:
        """Detalhes devem incluir rank, raw_score e rrf_contribution por algoritmo."""
        rrf = RRFusion(k=60)
        rankings = [
            [{"candidate": "A", "score": 0.85}],
            [{"candidate": "A", "score": 0.72}],
        ]
        result = rrf.fuse(rankings, ["cosine", "edit"])

        details = result[0]["details"]
        assert "cosine" in details
        assert "edit" in details
        assert details["cosine"]["rank"] == 1
        assert details["cosine"]["raw_score"] == pytest.approx(0.85)
        assert details["cosine"]["rrf_contribution"] == pytest.approx(1 / 61)
        assert details["edit"]["rank"] == 1
        assert details["edit"]["raw_score"] == pytest.approx(0.72)

    def test_scores_are_normalized_zero_to_one(self) -> None:
        """Todos os scores RRF devem estar no intervalo [0, 1]."""
        rrf = RRFusion(k=60)
        rankings = [
            [
                {"candidate": "A", "score": 0.9},
                {"candidate": "B", "score": 0.5},
                {"candidate": "C", "score": 0.1},
            ],
            [
                {"candidate": "C", "score": 0.8},
                {"candidate": "A", "score": 0.4},
                {"candidate": "B", "score": 0.2},
            ],
        ]
        result = rrf.fuse(rankings, ["cosine", "edit"])

        for item in result:
            assert 0.0 <= item["score"] <= 1.0


class TestRRFusionMissingCandidates:
    """Testes de candidatos ausentes em alguns rankings."""

    def test_missing_candidate_gets_penalty_rank(self) -> None:
        """Candidato ausente de um ranking recebe rank penalizado."""
        rrf = RRFusion(k=60)
        rankings = [
            [
                {"candidate": "A", "score": 0.9},
                {"candidate": "B", "score": 0.5},
            ],
            [
                {"candidate": "A", "score": 0.8},
                # B ausente do segundo ranking
            ],
        ]
        result = rrf.fuse(rankings, ["cosine", "edit"])

        b_result = next(r for r in result if r["candidate"] == "B")
        # B: rank 2 em cosine + penalty rank 2 (len=1, rank=2) em edit
        assert b_result["details"]["edit"]["rank"] == 2  # penalty
        assert b_result["details"]["edit"]["raw_score"] == 0.0

    def test_present_everywhere_beats_partial(self) -> None:
        """Candidato em todos os rankings supera parcialmente presente."""
        rrf = RRFusion(k=60)
        rankings = [
            [
                {"candidate": "A", "score": 0.6},
                {"candidate": "B", "score": 0.5},
            ],
            [
                {"candidate": "A", "score": 0.5},
                # B ausente
            ],
        ]
        result = rrf.fuse(rankings, ["cosine", "edit"])

        assert result[0]["candidate"] == "A"


class TestRRFusionParameterK:
    """Testes do efeito do parâmetro k."""

    def test_small_k_amplifies_rank_differences(self) -> None:
        """Com k pequeno, a diferença entre rank 1 e rank 2 é maior."""
        rrf_small = RRFusion(k=1)
        rrf_large = RRFusion(k=100)

        rankings = [
            [
                {"candidate": "A", "score": 0.9},
                {"candidate": "B", "score": 0.5},
            ],
        ]

        result_small = rrf_small.fuse(rankings, ["cosine"])
        result_large = rrf_large.fuse(rankings, ["cosine"])

        # Com k=1: rank1 = 1/2, rank2 = 1/3 -> ratio = 1.5
        # Com k=100: rank1 = 1/101, rank2 = 1/102 -> ratio ~= 1.01
        ratio_small = result_small[0]["score"] / result_small[1]["score"]
        ratio_large = result_large[0]["score"] / result_large[1]["score"]

        assert ratio_small > ratio_large

    def test_default_k_is_60(self) -> None:
        """K padrão deve ser 60 conforme a literatura."""
        rrf = RRFusion()
        assert rrf.k == 60


class TestRRFusionEdgeCases:
    """Testes de casos limítrofes."""

    def test_empty_rankings_returns_empty(self) -> None:
        """Lista vazia de rankings retorna lista vazia."""
        rrf = RRFusion(k=60)
        assert rrf.fuse([], []) == []

    def test_single_candidate_single_ranking(self) -> None:
        """Um candidato em um ranking retorna score 1.0."""
        rrf = RRFusion(k=60)
        rankings = [[{"candidate": "A", "score": 0.5}]]
        result = rrf.fuse(rankings, ["cosine"])

        assert len(result) == 1
        assert result[0]["candidate"] == "A"
        assert result[0]["score"] == pytest.approx(1.0)

    def test_all_rankings_empty(self) -> None:
        """Rankings vazios retornam lista vazia."""
        rrf = RRFusion(k=60)
        result = rrf.fuse([[], []], ["cosine", "edit"])
        assert result == []

    def test_result_is_sorted_descending(self) -> None:
        """Resultado deve estar ordenado por score descendente."""
        rrf = RRFusion(k=60)
        rankings = [
            [
                {"candidate": "C", "score": 0.3},
                {"candidate": "A", "score": 0.1},
            ],
            [
                {"candidate": "A", "score": 0.9},
                {"candidate": "C", "score": 0.2},
            ],
        ]
        result = rrf.fuse(rankings, ["cosine", "edit"])

        scores = [r["score"] for r in result]
        assert scores == sorted(scores, reverse=True)
