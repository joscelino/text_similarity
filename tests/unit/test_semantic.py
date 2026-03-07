"""Testes para o cálculo de similaridade semântica isolado."""

import pytest

from text_similarity.core.semantic import SemanticSimilarity
from text_similarity.exceptions import StageProcessingError

# Pula todos os testes de semântica se a biblioteca extra não estiver instalada.
pytestmark = pytest.mark.skipif(
    not pytest.importorskip("sentence_transformers", exc_type=ImportError),
    reason="Dependência 'sentence-transformers' não instalada.",
)


def test_semantic_basic_similarity():
    """Testa se a rede detecta que veículos e carros são similares."""
    # Usando um modelo super leve e rápido apenas para teste contínuo
    # 'all-MiniLM-L6-v2' tem apenas 80MB.
    sem = SemanticSimilarity(model_name="all-MiniLM-L6-v2", device="cpu")

    # Identidade exata
    assert sem.compare("carro", "carro") > 0.99

    # Sinônimos fortes (deve pontuar alto, > 0.6 em modelos decentes)
    score_synonym = sem.compare("fast car", "quick vehicle")
    assert score_synonym > 0.5

    # Completamente desconexos
    score_diff = sem.compare("fast car", "delicious yellow banana")
    assert score_diff < 0.3


def test_semantic_empty_strings():
    """Testa recuo seguro (fallback) para strings limpas/vazias."""
    sem = SemanticSimilarity(model_name="all-MiniLM-L6-v2", device="cpu")
    assert sem.compare("", "algo") == 0.0
    assert sem.compare("algo", "") == 0.0
    assert sem.compare("", "") == 0.0


def test_lazy_loading_isolation(monkeypatch):
    """Garante que a importação pesada só ocorre na chamada do compare()."""
    # Importa módulo isolado para limpar globals
    import text_similarity.core.semantic as sem_module

    # Reseta o estado global se algum outro teste sujou
    sem_module._GLOBAL_MODEL = None
    sem_module._CURRENT_MODEL_NAME = None

    # Inicializar a classe NÃO deve causar carregamento ou crashes
    algo = sem_module.SemanticSimilarity(model_name="invalid-model-name-1234")
    assert algo._model_ref is None

    # Tentar comparar algo real fará o gatilho, aí sim subindo a rede e falhando
    with pytest.raises(StageProcessingError):
        algo.compare("a", "b")
