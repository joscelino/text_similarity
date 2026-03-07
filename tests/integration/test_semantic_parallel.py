"""Testes de estresse integrando Semantic Embeddings com ProcessPoolExecutor."""

import pytest

from text_similarity.api import Comparator


# Segurança condicional para quem clona repositório limpo
pytestmark = pytest.mark.skipif(
    not pytest.importorskip("sentence_transformers", exc_type=ImportError),
    reason="Dependência 'sentence-transformers' não instalada.",
)


def test_parallel_with_semantics_stress():
    """Garante que rodar similaridade semântica em lote paralela funciona e não dá Memory/Pickle Error.

    Na restrição arquitetural solicitada, a lógica léxica atua como filtro.
    Só depois (em _score_candidates) os embeddings são calculados.
    Neste teste, as strings base (1) batem exatamente com o Léxico e (2)
    acionam a carga Global Lazy Init nas threads do ProcessPool.
    """
    comp = Comparator(mode="smart", use_embeddings=True)

    # 1. Ajustando os queries: A semântica diz que isso aproxima de veículo flex
    queries = ["automóvel movido a duas gasolinas", "veículo bicombustível automotivo"]

    # 2. Criando uma base suja razoável. Para não gastar 30s da CI local,
    # enviaremos 1.000 amostras randômicas falsas, e as corretas misturadas.
    candidates = [f"Item irrelevante e distante {i}" for i in range(1000)]

    # Injetando as verdadeiras na peneira lexical
    candidates.append("automóvel movido a gasolina")
    candidates.append("carro flex")

    # 3. Rodando processo paralelo via `compare_many_to_many` que criará o TF-IDF.
    # Ao passar das filtragens, o Pool delegará os "Top N (default 50)" finais
    # pro SemanticSimilarity processar. Isso comprova que a arquitetura isola o PyTorch.
    all_results = comp.compare_many_to_many(
        queries=queries,
        candidates=candidates,
        top_n=5,
        min_cosine=0.01,  # Permite que algo desça para o short-list semântico
        strategy="parallel",
        n_workers=2,
    )

    assert len(all_results) == 2, "Deveria ter retornado resultados para 2 queries"

    # Verificando se calculou o "semantic" key perfeitamente
    res1, res2 = all_results
    assert len(res1) <= 5
    assert len(res2) <= 5

    # Valida que as chaves semânticas foram populadas nos details
    # e que o código rodou perfeitamente nos Workers em paralelo.
    for results in all_results:
        for r in results:
            if r["candidate"] in ["automóvel movido a gasolina", "carro flex"]:
                details = r.get("details", {})

                # Se entidade der match total (short-circuit), ele talvez nao tenha `semantic`.
                # Analisamos apenas os que passaram pelo funil matemático completo
                if not "short_circuit" in details.get("entity", {}):
                    assert "semantic" in details, (
                        "O peso Semantic_Similarity operou localmente."
                    )
                    assert details["semantic"]["score"] >= 0.0
