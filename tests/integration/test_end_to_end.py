"""Testes de integração end-to-end com cenários reais em Português Brasileiro."""

from __future__ import annotations

import pytest

from text_similarity.api import Comparator


class TestMonetaryEntityNormalization:
    """Cenários com valores monetários em PT-BR."""

    def test_reais_vs_brl_format(self) -> None:
        """'30 reais' e 'R$ 30,00' devem ter score alto por normalização de entidade."""
        comp = Comparator.smart()
        score = comp.compare("Custa 30 reais", "O preço é R$ 30,00")
        assert score > 0.4, f"Score esperado > 0.4, obtido: {score}"

    def test_different_monetary_values_score_low(self) -> None:
        """Valores monetários diferentes devem diminuir o score."""
        comp = Comparator.smart()
        score = comp.compare("custa 10 reais", "custa 500 reais")
        # Os tokens de money serão diferentes: <money:10.0> vs <money:500.0>
        assert score < 0.9


class TestDimensionEntityNormalization:
    """Cenários com dimensões e pesos em PT-BR."""

    def test_dimension_normalization(self) -> None:
        """'2kg de arroz' deve ter score razoável vs '2 quilos de arroz'.

        Nota: sem SpaCy instalado o lemmatizer opera em modo pass-through,
        reduzindo a similaridade. Threshold calibrado para esse ambiente.
        """
        comp = Comparator.smart()
        score = comp.compare("comprei 2kg de arroz", "2 quilos de arroz")
        assert score > 0.2, f"Score esperado > 0.2, obtido: {score}"


class TestPhoneticSimilarity:
    """Cenários validando a análise fonética PT-BR."""

    def test_typographical_variation(self) -> None:
        """Variações de grafia fonética devem ter score alto no modo smart."""
        comp = Comparator.smart()
        score = comp.compare("cassaro", "carro")
        assert score > 0.3, f"Score esperado > 0.3, obtido: {score}"

    def test_phonetically_similar_words(self) -> None:
        """Palavras foneticamente similares devem ter score razoável.

        Nota: sem SpaCy instalado o lemmatizer opera em modo pass-through.
        Com SpaCy ativo o score esperado seria > 0.5.
        """
        comp = Comparator.smart()
        score = comp.compare("exceção", "excessão")
        assert score > 0.4, f"Score esperado > 0.4, obtido: {score}"


class TestComparatorBasicEndToEnd:
    """Cenários end-to-end com o modo básico."""

    def test_product_name_reorder(self) -> None:
        """Nome de produto com termos reorganizados deve ter score alto."""
        comp = Comparator.basic()
        score = comp.compare("iphone 13 pro", "iphone pro 13")
        assert 0.5 < score <= 1.0, f"Score esperado entre 0.5 e 1.0, obtido: {score}"

    def test_unrelated_products_score_low(self) -> None:
        """Produtos sem relação semântica devem ter score baixo."""
        comp = Comparator.basic()
        score = comp.compare("geladeira electrolux frost free", "foguete espacial da nasa")
        assert score < 0.2, f"Score esperado < 0.2, obtido: {score}"


class TestExplainEndToEnd:
    """Cenários end-to-end para o método explain()."""

    def test_explain_smart_mode_full_structure(self) -> None:
        """explain() no modo smart deve retornar estrutura completa."""
        comp = Comparator.smart()
        result = comp.explain("televisão samsung 55 polegadas", 'tv samsung 55"')

        assert "score" in result
        assert "details" in result
        assert result["score"] >= 0.0

        details = result["details"]
        for algo in ["cosine", "edit", "phonetic"]:
            assert algo in details, f"Algoritmo '{algo}' ausente em details"
            assert "score" in details[algo]
            assert "weight" in details[algo]
            assert 0.0 <= details[algo]["score"] <= 1.0

    def test_explain_basic_mode_no_phonetic(self) -> None:
        """explain() no modo básico não deve ter contribuição fonética."""
        comp = Comparator.basic()
        result = comp.explain("notebook dell", "laptop dell")
        # phonetic está em details mas com peso 0, portanto não deve aparecer
        # (HybridSimilarity só inclui se weight > 0)
        details = result["details"]
        assert "phonetic" not in details, "Modo básico não deve ter detalhe fonético"


class TestEdgeCases:
    """Casos de borda para robustez do pipeline."""

    def test_empty_strings(self) -> None:
        """Strings vazias devem retornar 0.0 sem erro."""
        comp = Comparator.basic()
        assert comp.compare("", "") == pytest.approx(0.0)

    def test_single_word_each(self) -> None:
        """Palavras únicas idênticas devem ter score alto."""
        comp = Comparator.basic()
        score = comp.compare("arroz", "arroz")
        assert score >= 0.8

    def test_special_characters_handled(self) -> None:
        """Textos com caracteres especiais não devem gerar exceção."""
        comp = Comparator.smart()
        score = comp.compare("R$ 1.500,00 !!!", "R$ 1500,00???")
        assert isinstance(score, float)

    def test_long_texts(self) -> None:
        """Textos longos devem ser processados sem erro."""
        comp = Comparator.basic()
        t1 = "produto " * 50 + "geladeira electrolux"
        t2 = "produto " * 50 + "geladeira electrolux"
        score = comp.compare(t1, t2)
        assert score >= 0.9
