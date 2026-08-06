"""Mixin de gerenciamento de índices do :class:`Comparator`.

Centraliza (SEC-DUP-003 + SEC-STD-001):

- :data:`INDEX_BUILDERS` — registry ``str → Callable[..., IndexProtocol]``
  para instanciação lazy de backends (``"bm25"``, ``"dense"``).
- :meth:`IndexManagerMixin._get_or_build_index` — helper que faz o
  reuso ou reconstrução do índice ativo e SEMPRE atualiza
  ``self._active_index``.
- :meth:`IndexManagerMixin.save_index` / :meth:`load_index` — persistência.
- :meth:`IndexManagerMixin.preprocess_catalog` — cache do catálogo em disco.
- :meth:`IndexManagerMixin.unload_embeddings_model` — libera modelo semântico.

O registry evita ``if/elif`` clonado em ``compare_many_to_many``:
adicionar um terceiro backend requer apenas registrar novo builder.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

from text_similarity.core._index_protocol import IndexProtocol
from text_similarity.core.hybrid import HybridSimilarity
from text_similarity.core.semantic import SemanticSimilarity

if TYPE_CHECKING:
    from text_similarity.core.bm25 import BM25Index
    from text_similarity.core.dense import DenseIndex

#: Tipo aceito por :meth:`IndexManagerMixin.save_index` /
#: :meth:`IndexManagerMixin.load_index`. Substitui o antigo ``str | Any``
#: para atender ao critério SEC-STD-005 ("sem ``Any`` em superfícies
#: públicas"). Aceita ``str`` ou qualquer objeto que implemente
#: ``os.PathLike[str]`` (ex: ``pathlib.Path``).
IndexPath = Union[str, "os.PathLike[str]"]


# ---------------------------------------------------------------------
# Builders lazy — imports adiados evitam carregar sentence-transformers
# quando o usuário só usa BM25 (e vice-versa).
# ---------------------------------------------------------------------
def _build_bm25(
    *,
    bm25_k1: float = 1.2,
    bm25_b: float = 0.75,
    **_ignored: Any,
) -> "BM25Index":
    """Instancia um :class:`BM25Index` configurado.

    Args:
        bm25_k1: Saturação de term frequency do BM25 (padrão 1.2).
        bm25_b: Normalização por comprimento do BM25 (padrão 0.75).
        **_ignored: Parâmetros extras são ignorados para permitir que o
            helper chame todos os builders com o mesmo kwargs unificado.

    Returns:
        Nova instância de ``BM25Index`` (ainda sem ``fit``).
    """
    from text_similarity.core.bm25 import BM25Index

    return BM25Index(k1=bm25_k1, b=bm25_b)


def _build_dense(
    *,
    dense_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    dense_model_revision: str | None = None,
    dense_precision: str = "float32",
    **_ignored: Any,
) -> "DenseIndex":
    """Instancia um :class:`DenseIndex` configurado.

    Args:
        dense_model_name: Nome do modelo sentence-transformers.
        dense_model_revision: Revisão (SHA) do modelo. Propagado como
            ``revision=<sha>`` para ``SentenceTransformer``.
        dense_precision: ``"float32"``, ``"int8"`` ou ``"binary"``.
        **_ignored: Parâmetros extras são ignorados (ver ``_build_bm25``).

    Returns:
        Nova instância de ``DenseIndex`` (ainda sem ``fit``).
    """
    from text_similarity.core.dense import DenseIndex

    return DenseIndex(
        model_name=dense_model_name,
        revision=dense_model_revision,
        precision=dense_precision,
    )


#: Registry oficial de builders. Mapeia ``indexing_strategy`` → callable
#: que retorna um objeto satisfazendo :class:`IndexProtocol`. Registrar
#: um novo backend basta para que ``_get_or_build_index`` funcione com ele.
INDEX_BUILDERS: Dict[str, Callable[..., IndexProtocol]] = {
    "bm25": _build_bm25,
    "dense": _build_dense,
}


class IndexManagerMixin:
    """Mixin que expõe o ciclo de vida dos índices (BM25 / Dense)."""

    # Atributos providos pelo Comparator (documentados para o type checker).
    _active_index: Optional[IndexProtocol]
    algorithm: Any
    indexing_strategy: str
    bm25_k1: float
    bm25_b: float
    dense_model_name: str
    dense_model_revision: str | None
    dense_precision: str

    # ------------------------------------------------------------------
    # Helper central (SEC-DUP-003)
    # ------------------------------------------------------------------
    def _get_or_build_index(
        self,
        strategy: str,
        p_candidates: List[str],
    ) -> IndexProtocol:
        """Retorna o índice ativo — reutilizando ou reconstruindo conforme necessário.

        Fluxo:
            1. Se ``self._active_index`` já é uma instância do tipo esperado
               para ``strategy``, reutiliza-a (não refaz ``fit``).
            2. Caso contrário, chama o builder registrado em
               :data:`INDEX_BUILDERS` (passando toda a configuração do
               comparador via kwargs) e ajusta o corpus.
            3. Em AMBOS os caminhos, atualiza ``self._active_index = index``
               antes de retornar (evita bug latente cf. SEC-LOGIC citada
               na SPEC de refactor).

        Args:
            strategy: ``"bm25"`` ou ``"dense"``. Deve ser uma chave de
                :data:`INDEX_BUILDERS`.
            p_candidates: Lista de textos JÁ pré-processados.

        Returns:
            Instância satisfazendo :class:`IndexProtocol`, pronta para
            :meth:`get_scores_normalized`.

        Raises:
            KeyError: Se ``strategy`` não estiver em :data:`INDEX_BUILDERS`.
        """
        if strategy not in INDEX_BUILDERS:
            raise KeyError(
                f"Estratégia de indexação '{strategy}' não registrada. "
                f"Chaves disponíveis: {sorted(INDEX_BUILDERS)}"
            )

        # Import local: mantém dense.py fora do módulo até realmente ser usado.
        from text_similarity.core.bm25 import BM25Index
        from text_similarity.core.dense import DenseIndex

        index: IndexProtocol
        if strategy == "bm25" and isinstance(self._active_index, BM25Index):
            index = self._active_index
        elif strategy == "dense" and isinstance(self._active_index, DenseIndex):
            index = self._active_index
        else:
            builder = INDEX_BUILDERS[strategy]
            index = builder(
                bm25_k1=self.bm25_k1,
                bm25_b=self.bm25_b,
                dense_model_name=self.dense_model_name,
                dense_model_revision=self.dense_model_revision,
                dense_precision=self.dense_precision,
            )
            index.fit(p_candidates)

        # Sempre atualizar — comportamento invariante do helper.
        self._active_index = index
        return index

    # ------------------------------------------------------------------
    # Persistência
    # ------------------------------------------------------------------
    def save_index(self, path: IndexPath) -> None:
        """Salva o índice ativo (BM25 ou Dense) em disco.

        Args:
            path: Caminho do arquivo de saída — ``str`` ou
                ``os.PathLike[str]`` (ex: ``pathlib.Path("idx.pkl")``).

        Raises:
            RuntimeError: Se nenhum índice estiver disponível para salvar.
        """
        if self._active_index is None:
            raise RuntimeError(
                "Nenhum índice disponível. Execute compare_batch ou "
                "compare_many_to_many primeiro para construir o índice."
            )
        self._active_index.save(path)

    def load_index(self, path: IndexPath) -> None:
        """Carrega um índice do disco substituindo o índice ativo.

        Args:
            path: Caminho do arquivo gerado por ``save_index()``. Aceita
                ``str`` ou ``os.PathLike[str]``.

        Raises:
            ValueError: Se o arquivo for inválido ou corrompido.
            RuntimeError: Se ``indexing_strategy`` não suporta persistência.
        """
        if self.indexing_strategy == "bm25":
            from text_similarity.core.bm25 import BM25Index

            self._active_index = BM25Index.load(path)
        elif self.indexing_strategy == "dense":
            from text_similarity.core.dense import DenseIndex

            self._active_index = DenseIndex.load(path)
        else:
            raise RuntimeError(
                "save_index/load_index suportados apenas para "
                "indexing_strategy='bm25' ou 'dense'."
            )

    # ------------------------------------------------------------------
    # Utilidades de recursos
    # ------------------------------------------------------------------
    def unload_embeddings_model(self) -> None:
        """Libera o modelo semântico (sentence-transformers) da memória global.

        Útil para liberar RAM/VRAM após uma sessão de inferência intensa,
        ou antes de trocar para um modelo diferente. Após a chamada, o
        modelo será recarregado automaticamente na próxima comparação
        semântica. Sem efeito se ``use_embeddings=False``.
        """
        if isinstance(self.algorithm, HybridSimilarity):
            semantic = self.algorithm.algorithms.get("semantic")
            if isinstance(semantic, SemanticSimilarity):
                semantic.unload()

    def preprocess_catalog(
        self,
        candidates: List[str],
        cache_path: str = "catalog_cache.pkl",
    ) -> List[str]:
        """Pré-processa candidatos e salva em disco para reuso.

        Na primeira execução, processa todos os candidatos e salva o
        resultado em ``cache_path``. Em execuções subsequentes com o
        mesmo catálogo, carrega direto do disco (~80% economia de tempo).

        A invalidação é automática via hash SHA-256 do conteúdo.

        Args:
            candidates: Lista de textos candidatos.
            cache_path: Caminho do arquivo de cache em disco.

        Returns:
            Lista de textos pré-processados.
        """
        # Atributos ``cache`` e ``_process_batch`` vêm do Comparator.
        cache = getattr(self, "cache", None)
        if cache is not None:
            loaded = cache.load_catalog(candidates, cache_path)
            if loaded is not None:
                return loaded  # type: ignore[no-any-return]

        processed = self._process_batch(candidates, preprocess=True)  # type: ignore[attr-defined]

        if cache is not None:
            cache.save_catalog(candidates, processed, cache_path)

        return processed  # type: ignore[no-any-return]


__all__ = ["IndexManagerMixin", "INDEX_BUILDERS", "IndexPath"]
