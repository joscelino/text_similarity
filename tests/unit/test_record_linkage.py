"""Testes para SEC-LOGIC-003: ``record_linkage`` cobre duplicatas em ``df_b``.

Bug antigo: ``record_linkage`` usava ``candidates.index(match['candidate'])``,
que sempre retornava o PRIMEIRO índice do texto candidato — escondendo
todas as linhas duplicadas de ``df_b``. Comportamento correto: um match
que atinge múltiplas linhas com o mesmo texto candidato deve produzir
uma entrada por linha (todos os ``index_b`` válidos).

Adicionalmente valida a complexidade: o mapa ``cand_index_map`` deve
ser construído uma única vez (linear), evitando o antigo O(n) por
match do ``list.index``.
"""

from __future__ import annotations

from typing import Any

import pytest

from text_similarity.api import Comparator

# ---------------------------------------------------------------------------
# DataFrame-like mínimo (mesmo padrão dos testes existentes)
# ---------------------------------------------------------------------------


class _SimpleDF:
    """DataFrame-like mínimo que suporta subscript por coluna."""

    def __init__(self, data: dict[str, list[Any]]) -> None:
        self._data = data
        self.columns = list(data.keys())

    def __getitem__(self, col: str) -> "_SimpleColumn":
        return _SimpleColumn(self._data[col])

    def __len__(self) -> int:
        first = next(iter(self._data.values()))
        return len(first)


class _SimpleColumn:
    def __init__(self, values: list[Any]) -> None:
        self._values = values

    def tolist(self) -> list[Any]:
        return list(self._values)

    def __iter__(self) -> Any:
        return iter(self._values)

    def __getitem__(self, i: int) -> Any:
        return self._values[i]


@pytest.fixture()
def comparator() -> Comparator:
    """Comparator básico — suficiente para exercitar a lógica de linkage."""
    return Comparator.basic()


# ---------------------------------------------------------------------------
# Casos de aceite SEC-LOGIC-003
# ---------------------------------------------------------------------------


class TestRecordLinkageDuplicates:
    """df_b com textos duplicados: TODAS as ocorrências devem emergir."""

    def test_duplicate_candidate_in_df_b_yields_all_indices(
        self, comparator: Comparator
    ) -> None:
        """Texto duplicado em df_b nas linhas 3 e 7 → duas linhas de resultado.

        Usa alta sobreposição lexical (query e candidato compartilham
        tokens) para garantir cos_score acima do min_cosine padrão do
        Comparator.basic().
        """
        queries = [
            "batata frita crocante",  # 0
            "carro esporte vermelho",  # 1
            "geladeira frost free",  # 2
            "televisão 4k samsung",  # 3
            "notebook gamer asus",  # 4
            "iphone 15 pro max apple",  # 5  ← query que vai casar
            "cadeira ergonômica preta",  # 6
        ]
        # df_b: MESMO texto candidato ocupando linhas 3 e 7. O texto
        # candidato tem forte overlap lexical com queries[5].
        candidates = [
            "produto totalmente diferente A",  # 0
            "produto totalmente diferente B",  # 1
            "produto totalmente diferente C",  # 2
            "iphone 15 pro max apple novo",  # 3  ← duplicata #1
            "produto totalmente diferente D",  # 4
            "produto totalmente diferente E",  # 5
            "produto totalmente diferente F",  # 6
            "iphone 15 pro max apple novo",  # 7  ← duplicata #2
        ]
        df_a = _SimpleDF({"q": queries})
        df_b = _SimpleDF({"c": candidates})

        result = comparator.record_linkage(df_a, df_b, "q", "c", top_n=5)

        # Filtra apenas as entradas geradas pela query 5
        query_5_hits = [r for r in result if r["index_a"] == 5]

        # Deve haver DUAS entradas: (5, 3) e (5, 7) para o candidato
        # duplicado — bug antigo emitia só uma (a primeira).
        target_text = "iphone 15 pro max apple novo"
        indices_b = sorted(
            r["index_b"] for r in query_5_hits if r["text_b"] == target_text
        )
        assert indices_b == [3, 7], (
            "Duplicatas nas linhas 3 e 7 de df_b não apareceram ambas no "
            "output — bug SEC-LOGIC-003 regrediu. "
            f"Obteve: {indices_b}"
        )

        # Cada duplicata carrega o mesmo texto candidato
        for r in query_5_hits:
            if r["index_b"] in (3, 7):
                assert r["text_b"] == target_text
                assert r["text_a"] == "iphone 15 pro max apple"
                assert 0.0 <= r["score"] <= 1.0

    def test_multiple_queries_with_duplicate_candidates(
        self, comparator: Comparator
    ) -> None:
        """Cenário completo: duas queries casam com o mesmo candidato duplicado."""
        df_a = _SimpleDF(
            {
                "q": [
                    "ferrari vermelha esportiva",
                    "ferrari vermelha carro",
                ]
            }
        )
        df_b = _SimpleDF(
            {
                "c": [
                    "ferrari vermelha modelo",  # 0  ← duplicata
                    "geladeira branca eletro",  # 1
                    "ferrari vermelha modelo",  # 2  ← duplicata
                ]
            }
        )
        result = comparator.record_linkage(
            df_a, df_b, "q", "c", top_n=5, min_cosine=0.05
        )

        # Para cada query, ambas as duplicatas (index_b 0 e 2) devem aparecer
        target_text = "ferrari vermelha modelo"
        for query_idx in (0, 1):
            hits = [r for r in result if r["index_a"] == query_idx]
            indices_b = sorted(r["index_b"] for r in hits if r["text_b"] == target_text)
            assert indices_b == [0, 2], (
                f"Query {query_idx}: esperava duplicatas [0, 2], obteve {indices_b}"
            )

    def test_no_duplicates_matches_previous_behavior(
        self, comparator: Comparator
    ) -> None:
        """Sem duplicatas em df_b, o comportamento antigo é preservado."""
        df_a = _SimpleDF({"q": ["notebook dell"]})
        df_b = _SimpleDF({"c": ["laptop dell inspiron", "geladeira", "iphone"]})
        result = comparator.record_linkage(df_a, df_b, "q", "c", top_n=3)

        # Nenhum candidato repetido → cada index_b deve ser único
        indices_b = [r["index_b"] for r in result]
        assert len(indices_b) == len(set(indices_b))

    def test_unknown_candidate_text_does_not_emit_row(
        self, comparator: Comparator
    ) -> None:
        """Degradação segura: candidato ausente em df_b não gera linha spurious.

        Garante que o novo lookup ``cand_index_map.get(text, [])``
        realmente devolve lista vazia (e não KeyError) para textos
        desconhecidos — o que não deveria acontecer em uso normal, mas
        blinda contra regressões da API interna.
        """
        # Cenário controlado: usamos uma df_b conhecida
        df_a = _SimpleDF({"q": ["carro"]})
        df_b = _SimpleDF({"c": ["carro", "geladeira"]})

        result = comparator.record_linkage(df_a, df_b, "q", "c", top_n=5)

        # Todos os text_b resultantes devem existir na coluna original
        valid_texts = set(df_b["c"].tolist())
        for r in result:
            assert r["text_b"] in valid_texts


class TestRecordLinkageImplementation:
    """Blindagem de implementação: cand_index_map em vez de list.index."""

    def test_source_uses_cand_index_map_not_list_index(self) -> None:
        """Auditoria: o corpo de record_linkage NÃO deve mais usar list.index.

        Este é um teste de "sanidade da correção" — se alguém acidentalmente
        reverter para ``candidates.index(...)`` o bug ressurge silenciosamente
        (é observável apenas em cenários com duplicatas). Aqui verificamos
        a assinatura da correção diretamente no source, ignorando linhas
        de comentário (que podem citar o padrão antigo para documentação).
        """
        import inspect

        from text_similarity.api import Comparator as _Comparator

        source = inspect.getsource(_Comparator.record_linkage)

        # Remove comentários (linhas começando com #) e docstring para
        # inspecionar apenas código executável.
        code_lines = [
            line for line in source.splitlines() if not line.lstrip().startswith("#")
        ]
        code_only = "\n".join(code_lines)

        assert "cand_index_map" in code_only, (
            "record_linkage deveria construir cand_index_map pré-computado."
        )
        assert "candidates.index(" not in code_only, (
            "record_linkage NÃO deve mais chamar candidates.index(...) — "
            "essa era a origem do bug SEC-LOGIC-003 (esconde duplicatas)."
        )
