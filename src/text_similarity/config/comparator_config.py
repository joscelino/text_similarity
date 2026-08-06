"""Dataclass ``ComparatorConfig`` — configuração unificada do Comparator.

Elimina a duplicação de assinatura entre ``Comparator.__init__``,
``Comparator.basic`` e ``Comparator.smart`` (SEC-DUP-001) e promove o
parâmetro ``use_embeddings`` a campo tipado explícito (SEC-STD-002).

Exemplo:
    >>> from text_similarity.config import ComparatorConfig
    >>> cfg = ComparatorConfig(mode="smart", use_embeddings=True)
    >>> cfg.mode
    'smart'
    >>> cfg.use_embeddings
    True
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class ComparatorConfig:
    """Configuração unificada do :class:`text_similarity.api.Comparator`.

    Este dataclass reúne todos os parâmetros configuráveis do Comparator
    em um único objeto imutável-por-convenção, permitindo:

    - Eliminar a duplicação de assinatura entre ``__init__``, ``basic`` e
      ``smart`` (SEC-DUP-001).
    - Promover ``use_embeddings`` a campo tipado explícito (SEC-STD-002).
    - Validar a soma de pesos = 1.0 em ``__post_init__`` quando o usuário
      fornece um dicionário customizado via ``weights``.

    Attributes:
        mode: Modo de operação — ``"basic"`` ou ``"smart"``.
        entities: Lista de entidades a extrair no modo ``"smart"``.
            Ignorado no modo ``"basic"``.
        use_cache: Habilita o cache in-memory de textos pré-processados.
        use_embeddings: Ativa Similaridade Semântica baseada em embeddings
            densos. Quando ``True``, o Comparator seleciona um perfil de
            pesos que reserva parte da soma para ``"semantic"``.
        fusion_strategy: Estratégia de fusão para operações batch —
            ``"linear"`` (soma ponderada) ou ``"rrf"``
            (Reciprocal Rank Fusion).
        rrf_k: Constante de suavização do RRF (padrão 60). Ignorado
            quando ``fusion_strategy == "linear"``.
        rrf_weights: Pesos por algoritmo para o RRF. Se ``None``, todos
            os algoritmos contribuem igualmente. Ignorado quando
            ``fusion_strategy == "linear"``.
        indexing_strategy: Estratégia de indexação para batch —
            ``"tfidf"``, ``"bm25"`` ou ``"dense"``.
        bm25_k1: Saturação de term frequency do BM25 (padrão 1.2).
        bm25_b: Normalização por comprimento do BM25 (padrão 0.75).
        dense_model_name: Nome do modelo sentence-transformers.
            **Atenção de segurança:** não aceite valores fornecidos por
            usuários não confiáveis. Se necessário, aplique uma whitelist
            na aplicação hospedeira. Para fixar uma revisão específica do
            HuggingFace, use ``dense_model_revision``.
        dense_model_revision: Revisão (SHA) do modelo sentence-transformers
            a ser carregada. Quando informado, é propagado como
            ``revision=<sha>`` para ``SentenceTransformer``, permitindo pin
            de supply chain. Padrão ``None`` (usa a revisão padrão do
            HuggingFace). Recomenda-se fixar um SHA confiável em produção.
        dense_precision: Precisão dos embeddings do DenseIndex —
            ``"float32"``, ``"int8"`` ou ``"binary"``.
        parallel_threshold: Número mínimo de textos para ativar
            pré-processamento paralelo em ``_process_batch``.
        strict: Propagado para
            :class:`~text_similarity.core.semantic.SemanticSimilarity`.
            ``True`` (padrão) é recomendado para produção.
        weights: Override opcional do perfil de pesos padrão. Quando
            fornecido, deve ser um ``dict[str, float]`` cuja soma seja
            aproximadamente 1.0 (validado em ``__post_init__``).

    Raises:
        ValueError: Se ``weights`` for fornecido e a soma dos valores
            não for aproximadamente 1.0.
        ValueError: Se ``mode`` não for ``"basic"`` nem ``"smart"``.
    """

    mode: str = "basic"
    entities: Optional[list[str]] = None
    use_cache: bool = True
    use_embeddings: bool = False
    fusion_strategy: Literal["linear", "rrf"] = "linear"
    rrf_k: int = 60
    rrf_weights: Optional[dict[str, float]] = None
    indexing_strategy: Literal["tfidf", "bm25", "dense"] = "tfidf"
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    dense_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
    dense_model_revision: Optional[str] = None
    dense_precision: str = "float32"
    parallel_threshold: int = 1000
    strict: bool = True
    weights: Optional[dict[str, float]] = field(default=None)

    def __post_init__(self) -> None:
        """Valida os campos após a inicialização do dataclass.

        Regras:
            - ``mode`` deve ser ``"basic"`` ou ``"smart"``.
            - Se ``weights`` for fornecido, sua soma deve ser
              aproximadamente 1.0 (via ``math.isclose``).

        Raises:
            ValueError: Se qualquer regra acima falhar.
        """
        if self.mode not in ("basic", "smart"):
            raise ValueError(
                f"mode deve ser 'basic' ou 'smart', recebido: {self.mode!r}"
            )

        if self.weights is not None:
            total = sum(self.weights.values())
            if not math.isclose(total, 1.0):
                raise ValueError(
                    "weights deve somar 1.0 (tolerância math.isclose), "
                    f"soma recebida={total}"
                )


__all__ = ["ComparatorConfig"]
