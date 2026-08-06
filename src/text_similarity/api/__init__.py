"""Fachada pública do pacote :mod:`text_similarity.api` (SEC-STD-001).

Este módulo mantém a superfície pública histórica — ``from
text_similarity.api import Comparator`` continua funcionando —
enquanto delega a implementação real para submódulos por domínio:

- :mod:`text_similarity.api.comparator` — orquestração do Comparator.
- :mod:`text_similarity.api.batch` — operações batch e reranking.
- :mod:`text_similarity.api.scoring` — scoring linear e RRF.
- :mod:`text_similarity.api.dataframe_ops` — operações DataFrame-like.
- :mod:`text_similarity.api.index_manager` — ciclo de vida dos índices
  (BM25 / Dense) com registry ``INDEX_BUILDERS`` e helper
  ``_get_or_build_index``.
"""

from __future__ import annotations

from .comparator import Comparator

__all__ = ["Comparator"]
