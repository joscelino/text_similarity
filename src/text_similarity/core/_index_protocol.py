"""Protocolos estruturais para índices de similaridade (SEC-STD-005 / SEC-DUP-003).

Define ``IndexProtocol``, o contrato mínimo compartilhado por
:class:`~text_similarity.core.bm25.BM25Index` e
:class:`~text_similarity.core.dense.DenseIndex`. Usado para:

1. Tipar as anotações de ``bm25_index``/``dense_index`` em
   :mod:`text_similarity.pipeline.parallel` (substituindo ``Any``).
2. Tipar o registry ``INDEX_BUILDERS`` e o helper
   ``_get_or_build_index`` em :mod:`text_similarity.api.index_manager`.

O alias ``_CosineIndex`` mantém compatibilidade com a nomenclatura
adotada pelo scan de segurança (SEC-STD-005) — ambos apontam para o
mesmo :class:`typing.Protocol`.
"""

from __future__ import annotations

import os
from typing import Any, List, Protocol, Union, runtime_checkable

from numpy.typing import NDArray


@runtime_checkable
class IndexProtocol(Protocol):
    """Contrato estrutural para índices de similaridade em batch.

    Qualquer classe que implemente ``fit`` e ``get_scores_normalized``
    (e opcionalmente ``save``) satisfaz este ``Protocol``. Não requer
    herança explícita — é *duck typing* verificado em tempo de
    checagem estática (``mypy``) e opcionalmente em runtime via
    :func:`isinstance` (graças a ``@runtime_checkable``).

    Métodos:
        fit(documents): Indexa o corpus de candidatos.
        get_scores_normalized(query): Retorna scores em ``[0, 1]``.
        save(path): Persiste o índice em disco (opcional/parcial).
    """

    def fit(self, documents: List[str]) -> "IndexProtocol":  # noqa: D401
        """Indexa um corpus de documentos pré-processados."""
        ...

    def get_scores_normalized(self, query: str) -> NDArray[Any]:  # noqa: D401
        """Similaridade da query contra o corpus, normalizada em ``[0, 1]``."""
        ...

    def save(self, path: Union[str, "os.PathLike[str]"]) -> None:  # noqa: D401
        """Persiste o índice em disco (formato específico da implementação)."""
        ...


# Alias usado nas anotações de pipeline/parallel.py — cf. SEC-STD-005.
_CosineIndex = IndexProtocol


__all__ = ["IndexProtocol", "_CosineIndex"]
