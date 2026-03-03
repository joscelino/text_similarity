import pytest

from text_similarity.core.cosine import CosineSimilarity
from text_similarity.core.hybrid import HybridSimilarity
from text_similarity.core.phonetic import PhoneticSimilarity
from text_similarity.core.rapidfuzz_cmp import EditDistanceSimilarity


def test_edit_distance_similarity() -> None:
    algo = EditDistanceSimilarity(method="ratio")
    # "rato" vs "gato" (1 modificação)
    score = algo.compare("rato", "gato")
    assert 0.0 < score < 1.0
    assert algo.compare("cachorro", "cachorro") == 1.0
    assert algo.compare("cachorro", "gato") < 0.4


def test_edit_distance_partial() -> None:
    algo = EditDistanceSimilarity(method="partial_ratio")
    # partial ratio é muito forte quando a string inteira está dentro da outra
    score = algo.compare("gato", "o gato do vizinho")
    assert score == 1.0


def test_edit_distance_token_sort_ratio() -> None:
    """token_sort_ratio deve tratar textos com mesmas palavras em ordem diferente."""
    algo = EditDistanceSimilarity(method="token_sort_ratio")
    # Mesmas palavras, ordem diferente — token_sort_ratio normaliza para 1.0
    score = algo.compare("iphone 13 pro max", "pro max iphone 13")
    assert score == 1.0
    # Textos completamente diferentes devem ter score baixo
    assert algo.compare("geladeira electrolux", "foguete lunar") < 0.5


def test_cosine_similarity() -> None:
    algo = CosineSimilarity(ngram_range=(1, 1))
    score = algo.compare("rato grande amarelo", "rato grand eamarelo")
    # "rato" é o único termo em comum
    assert score > 0.1
    assert score < 1.0

    # Textos exatos tem score 1.0 (approximado por causa de ponto flutuante)
    assert algo.compare("um dois tres", "um dois tres") == pytest.approx(1.0)


def test_phonetic_similarity() -> None:
    algo = PhoneticSimilarity()

    # "sessão", "sessao", "seção", "cessão"
    assert algo.compare("exceção", "esessao") > 0.7

    # casa e kaza (1 char muda o ratio na edição)
    assert algo.compare("casa", "kaza") > 0.7

    # paçoca e pasoka
    assert algo.compare("paçoca", "pasoka") > 0.8


def test_hybrid_similarity() -> None:
    algo = HybridSimilarity(weights={"cosine": 0.5, "edit": 0.5, "phonetic": 0.0})

    score = algo.compare("iphone 13 pro", "iphone pro 13")

    # cosseno (bag of words) dirá que é igual (se usar ngram 1) ou parecido.
    # edit distance ratio vai penalizar a troca de ordem.
    # resultado final deve ser a mescla
    assert 0.5 < score < 1.0

    explanation = algo.explain("casa", "kaza")
    assert "score" in explanation
    assert "cosine" in explanation["details"]
    assert "edit" in explanation["details"]


def test_cosine_custom_ngram_range() -> None:
    """CosineSimilarity deve funcionar com ngram_range customizado."""
    algo_unigram = CosineSimilarity(ngram_range=(1, 1))
    algo_bigram = CosineSimilarity(ngram_range=(2, 2))

    t1, t2 = "telefone celular samsung", "celular samsung galaxy"

    score_uni = algo_unigram.compare(t1, t2)
    score_bi = algo_bigram.compare(t1, t2)

    # Ambos devem retornar float válido no range [0.0, 1.0]
    assert 0.0 <= score_uni <= 1.0
    assert 0.0 <= score_bi <= 1.0
    # Unigrama tende a dar score maior (mais termos em comum individualmente)
    assert score_uni >= score_bi


def test_hybrid_explain_short_circuit_consistent_with_compare() -> None:
    """explain() deve retornar o mesmo score que compare() no short-circuit."""
    algo = HybridSimilarity()

    # Tags de entidade para forçar o short-circuit (entity_score == 1.0)
    t1 = "<productmodel:GN500>"
    t2 = "<productmodel:GN500> outros termos no texto"

    score_compare = algo.compare(t1, t2)
    result_explain = algo.explain(t1, t2)

    assert score_compare == pytest.approx(result_explain["score"])
    # Quando há short-circuit, o score deve ser 0.95
    assert result_explain["score"] == pytest.approx(0.95)
    assert result_explain["details"]["entity"]["short_circuit"] is True
