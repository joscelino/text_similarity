"""Testes unitários para a ferramenta de Dashboard de Calibração."""


from text_similarity.api import Comparator
from text_similarity.tuning.calibrator import WeightCalibrator


def test_calibrator_immutability():
    """Garante que a função não avaria o objeto Comparator original do usuário."""
    comp = Comparator.smart()
    original_weights = comp.algorithm.weights.copy()

    # Cria grid search
    configs = [
        {"cosine": 0.1, "edit": 0.9},
        {"phonetic": 1.0},
    ]

    gs = [
        {"query": "pneu", "target": "peneu", "match": True},
        {"query": "carro", "target": "avião", "match": False},
    ]

    tuner = WeightCalibrator(comp, configs)
    _ = tuner.evaluate(gs)

    # O comparator_original do usuário não pode ter sido editado pelo loop de cfg
    assert comp.algorithm.weights == original_weights


def test_calibrator_empty_string_handling():
    """Garante que cenários defeituosos não interrompem um teste em lote."""
    comp = Comparator.basic()
    
    gs = [
        {"query": "teste limpo", "target": "teste limpo", "match": True},
        {"query": "", "target": "algum lixo", "match": False},
        {"query": "  ", "target": "", "match": False},
    ]
    
    configs = [{"cosine": 1.0}]
    tuner = WeightCalibrator(comp, configs)
    
    # Não deve "cachar" exceptions nem travar
    report = tuner.evaluate(gs)
    
    # Ele acertou perfeitamente: o primeiro é match (1.0) e os vazios dão score 0.0 (!match).
    assert report.best_metrics["f1_score"] > 0.99


def test_calibrator_correct_f1_score():
    """Validação matemática simplificada para precisão e recall na melhor config."""
    comp = Comparator.smart()
    
    # Base: 2 matches reais, 1 falso verdadeiro
    # Levenshtein pega erros pequenos ("casa" vs "caza"). Cosseno é burro para isso ("casa" vs "caza" -> 0.0)
    gs = [
        {"query": "casa", "target": "caza", "match": True}, # Esperamos que cfg1 pontue alto (match) e cfg2 zero
        {"query": "celular", "target": "cel", "match": False}, # Todos perdem ou pontuam baixo
        {"query": "teste", "target": "teste", "match": True}, # Todos ganham
    ]

    configs = [
        {"edit": 1.0},    # Vai encontrar casa vs caza
        {"cosine": 1.0},  # Vai falhar em casa vs caza (retorna 0 TFIDF match)
    ]

    tuner = WeightCalibrator(comp, configs, match_threshold=0.65)
    report = tuner.evaluate(gs)

    # O melhor peso garantidamente tem de ser o EDIT neste grid para a amostra acima.
    assert report.best_weights == {"edit": 1.0}
    assert report.best_metrics["f1_score"] == 1.0  # Perfect Score com EDIT

    # Testando o Discrepancy (O pior cenário, cfg de Cosseno, deve ter gerado 1 FN "casa" vs "caza")
    # Para o Cosseno: TP=1 ("teste"), FN=1 ("casa"/"caza"), FP=0
    cosine_res = [r for r in report.all_results if r["weights"] == {"cosine": 1.0}][0]
    assert cosine_res["metrics"]["f1_score"] < 1.0
