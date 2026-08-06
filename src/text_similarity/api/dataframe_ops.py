"""Mixin com operações DataFrame-like do :class:`Comparator`.

Isola do módulo principal (SEC-STD-001) três métodos:

- :meth:`DataFrameOpsMixin._extract_column` — helper genérico compatível
  com pandas, polars, cuDF, modin e qualquer objeto subscritável.
- :meth:`DataFrameOpsMixin.compare_dataframe` — compara uma query contra
  uma coluna de texto de um DataFrame-like.
- :meth:`DataFrameOpsMixin.record_linkage` — cruza dois DataFrames-like
  encontrando pares mais similares.

SEC-STD-005: superfícies públicas agora recebem :class:`pandas.DataFrame`
tipado (via ``pandas-stubs``) em vez de ``Any``. Para preservar
compatibilidade com polars/cuDF/modin sem introduzir dependências
opcionais, mantemos um :class:`DataFrameLike` — Protocol estrutural que
o mypy aceita quando o usuário passa outros DataFrames "duck-typed".
Internamente ainda usamos ``cast()`` pontual para satisfazer o tipo mais
estrito do pandas quando necessário.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Protocol, Union, cast

if TYPE_CHECKING:
    import pandas as pd


class DataFrameLike(Protocol):
    """Contrato estrutural mínimo aceito pelos métodos de DataFrame.

    Qualquer objeto que satisfaça o subscript ``df[coluna]`` cujo
    retorno seja iterável (opcionalmente com ``.tolist()`` ou
    ``.to_list()``) é aceito — cobre ``pandas``, ``polars``, ``cudf``,
    ``modin``, ``pyarrow.Table`` etc. sem exigir dependência opcional.

    Note:
        Esta abstração NÃO é ``Any``: mypy validará que o objeto
        implementa ``__getitem__(str)``. É o mínimo pedido pela SPEC
        SEC-STD-005 para superfícies públicas.
    """

    def __getitem__(self, key: str) -> Any:  # pragma: no cover - protocol
        """Retorna a coluna nomeada."""
        ...


#: Tipo público aceito por :meth:`compare_dataframe` e
#: :meth:`record_linkage`. Preferimos ``pd.DataFrame`` (com stubs),
#: mas mantemos ``DataFrameLike`` como *fallback* para polars/cuDF etc.
DataFrameInput = Union["pd.DataFrame", DataFrameLike]


class DataFrameOpsMixin:
    """Mixin com operações sobre DataFrames-like."""

    @staticmethod
    def _extract_column(df: DataFrameInput, col: str) -> List[str]:
        """Extrai coluna de qualquer DataFrame-like como lista de strings.

        Suporta pandas, polars, cuDF, modin e qualquer objeto
        subscritável que retorne uma coluna iterável.

        Args:
            df: DataFrame-like com suporte a subscript por nome de coluna
                (:class:`pandas.DataFrame` ou :class:`DataFrameLike`).
            col: Nome da coluna a extrair.

        Returns:
            Lista de strings com os valores da coluna.
        """
        column = df[col]
        if hasattr(column, "tolist"):  # pandas, cuDF, modin, numpy
            return cast(List[str], column.tolist())
        if hasattr(column, "to_list"):  # polars, pyarrow
            return cast(List[str], column.to_list())
        return list(column)  # fallback genérico

    def compare_dataframe(
        self,
        df: DataFrameInput,
        text_column: str,
        query: str,
        top_n: int = 50,
        min_cosine: float = 0.1,
        preprocess: bool = True,
    ) -> List[Dict[str, Any]]:
        """Compara uma query contra uma coluna de texto de um DataFrame-like.

        Compatível com pandas, polars, cuDF, modin ou qualquer objeto
        que suporte subscript por nome de coluna. Retorna uma lista de
        dicionários — converta para o DataFrame da sua escolha conforme
        necessário.

        Args:
            df: :class:`pandas.DataFrame` (ou :class:`DataFrameLike`)
                com os candidatos.
            text_column: Nome da coluna de texto para comparar.
            query: Texto de busca.
            top_n: Número máximo de resultados.
            min_cosine: Limiar mínimo de cosseno.
            preprocess: Se False, bypassa o pré-processamento.

        Returns:
            Lista de dicts com as chaves do DataFrame original + ``score``,
            ordenada do maior para o menor score.
        """
        candidates = self._extract_column(df, text_column)
        # ``compare_batch`` é provido pelo BatchMixin no Comparator.
        results = self.compare_batch(  # type: ignore[attr-defined]
            query,
            candidates,
            top_n=top_n,
            min_cosine=min_cosine,
            preprocess=preprocess,
        )

        # Materialize all rows once to avoid repeated column extractions.
        # Usamos ``cast(Any, df)`` apenas internamente: a superfície pública
        # continua tipada como ``DataFrameInput`` (SEM ``Any``).
        df_any: Any = cast(Any, df)
        col_names: List[str]
        if hasattr(df_any, "columns"):
            col_names = list(df_any.columns)
        elif hasattr(df_any, "schema"):
            col_names = list(df_any.schema.names())
        elif hasattr(df_any, "__len__") and len(df_any) > 0:
            col_names = list(df_any[0].keys())
        else:
            col_names = [text_column]
        rows_by_text: Dict[str, List[Dict[str, Any]]] = {}
        for i, text in enumerate(candidates):
            row: Dict[str, Any] = {}
            for c in col_names:
                col_vals = self._extract_column(df, c)
                row[c] = col_vals[i]
            rows_by_text.setdefault(text, []).append(row)

        output: List[Dict[str, Any]] = []
        seen_texts: set[str] = set()
        for r in results:
            text = r["candidate"]
            if text not in seen_texts and text in rows_by_text:
                seen_texts.add(text)
                record = dict(rows_by_text[text][0])
                record["score"] = r["score"]
                output.append(record)

        output.sort(key=lambda x: x["score"], reverse=True)
        return output

    def record_linkage(
        self,
        df_a: DataFrameInput,
        df_b: DataFrameInput,
        col_a: str,
        col_b: str,
        top_n: int = 5,
        min_cosine: float = 0.1,
        preprocess: bool = True,
    ) -> List[Dict[str, Any]]:
        """Cruza dois DataFrames-like encontrando pares mais similares.

        Compatível com pandas, polars, cuDF, modin ou qualquer objeto
        que suporte subscript por nome de coluna.

        Para cada linha do ``df_a``, encontra os ``top_n`` candidatos
        mais similares no ``df_b``, retornando uma lista de dicionários
        com os pares e scores.

        Args:
            df_a: :class:`pandas.DataFrame` (ou :class:`DataFrameLike`)
                com as queries (tabela A).
            df_b: :class:`pandas.DataFrame` (ou :class:`DataFrameLike`)
                com os candidatos (tabela B).
            col_a: Coluna de texto em df_a.
            col_b: Coluna de texto em df_b.
            top_n: Número máximo de matches por query.
            min_cosine: Limiar mínimo de cosseno.
            preprocess: Se False, bypassa o pré-processamento.

        Returns:
            Lista de dicts com chaves: ``index_a``, ``text_a``,
            ``index_b``, ``text_b``, ``score``, ``details``,
            ordenada do maior para o menor score.
        """
        queries = self._extract_column(df_a, col_a)
        candidates = self._extract_column(df_b, col_b)
        # ``compare_many_to_many`` é provido pelo BatchMixin.
        all_results = self.compare_many_to_many(  # type: ignore[attr-defined]
            queries,
            candidates,
            top_n=top_n,
            min_cosine=min_cosine,
            preprocess=preprocess,
        )

        # SEC-LOGIC-003: pré-computa mapeamento texto→[índices] em df_b
        # UMA vez, em vez de usar ``candidates.index(...)`` (O(n) e que
        # só retornava a PRIMEIRA ocorrência, escondendo duplicatas).
        cand_index_map: Dict[str, List[int]] = {}
        for idx, text in enumerate(candidates):
            cand_index_map.setdefault(text, []).append(idx)

        records: List[Dict[str, Any]] = []
        for query_idx, matches in enumerate(all_results):
            text_a = queries[query_idx]
            # Deduplica matches por texto: se ``compare_many_to_many``
            # retornou o mesmo texto candidato mais de uma vez, queremos
            # expandir para os índices reais de df_b apenas UMA vez por
            # texto.
            seen_texts: set[str] = set()
            for match in matches:
                cand_text = match["candidate"]
                if cand_text in seen_texts:
                    continue
                seen_texts.add(cand_text)
                for cand_idx in cand_index_map.get(cand_text, []):
                    records.append(
                        {
                            "index_a": query_idx,
                            "text_a": text_a,
                            "index_b": cand_idx,
                            "text_b": cand_text,
                            "score": match["score"],
                            "details": match["details"],
                        }
                    )

        records.sort(key=lambda x: x["score"], reverse=True)
        return records


__all__ = ["DataFrameOpsMixin", "DataFrameLike", "DataFrameInput"]
