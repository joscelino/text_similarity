"""Módulo com algoritmo de similaridade focado em interceptação de entidades.

Calcula a interseção entre entidades previamente demarcadas (ex: <productmodel:...>)
nos textos e, caso ocorra interseção total (a busca contida no outro), garante
um score máximo (short-circuit).
"""

from __future__ import annotations

import re

from text_similarity.core.base import SimilarityAlgorithm


class EntityIntersectionSimilarity(SimilarityAlgorithm):
    """Aplica similaridade baseada apenas em entidades normalizadas (<entidade:valor>).

    Este algoritmo tenta encontrar interseções de entidades entre dois textos.
    Especialmente útil se usado de forma short-circuit: se um texto é essencialmente
    uma busca específica (<productmodel:X>) e ela está contida no outro texto,
    retorna 1.0 (ou altíssimo), ignorando o volume restante do texto longo.
    """

    def __init__(self, target_entities: list[str] | None = None) -> None:
        """Inicializa identificando de quais entidades procurar.

        Args:
            target_entities: Lista de prefixos (ex: ["productmodel", "money"]).
                Se None, considera qualquer tag no padrão <X:Y>.
        """
        self.target_entities = target_entities

    def _extract_tags(self, text: str) -> set[str]:
        """Busca todas as tags <entidade:valor> no texto."""
        # Regex captura o padrão <tipo:valor> normalizado pelo pipeline
        pattern = r"<([a-zA-Z0-9_\-]+):([^>]+)>"
        tags = set()
        for match in re.finditer(pattern, text):
            ent_type, ent_val = match.groups()

            # Filtra caso target_entities foi definido
            if self.target_entities is None or ent_type in self.target_entities:
                tags.add(f"<{ent_type}:{ent_val}>")

        return tags

    def compare(self, text1: str, text2: str) -> float:
        """Retorna similaridade focado exclusivamente nas entidades em comum.

        Se o conjunto de entidades de T1 estiver totalmente contido no
        conjunto de T2 (ou vice-versa) E não for vazio, retorna 1.0.
        Caso contrário, 0.0. A ideia primária é de ser um ativador binário
        para interceptação de "short-circuit".
        """
        tags1 = self._extract_tags(text1)
        tags2 = self._extract_tags(text2)

        if not tags1 or not tags2:
            return 0.0

        # Verifica contenção: se algum valor (sem os delimitadores <tipo: ... >)
        # do menor conjunto está inteiramente contido em algum valor do
        # maior conjunto. Isso ajuda quando as strings de origem colaram
        # os modelos, ex: <productmodel:PPW38002> in <productmodel:PPW38002PROFEMUR>

        # Determina qual conjunto tem menos tags (a "busca" provavelmente)
        if len(tags1) <= len(tags2):
            search_tags, target_tags = list(tags1), list(tags2)
        else:
            search_tags, target_tags = list(tags2), list(tags1)

        for s_tag in search_tags:
            # Extrai apenas o valor real mapeado
            # ex: <productmodel:GN500> -> GN500
            s_val = s_tag.split(":", 1)[1][:-1]

            # Checa se esse valor existe como substring dentro de algum target_tag
            found = False
            for t_tag in target_tags:
                t_val = t_tag.split(":", 1)[1][:-1]
                if s_val in t_val:
                    found = True
                    break

            # Como a busca é de contenção PERFECTA (todas as tags buscadas devem
            # ser encontradas para validar o short-circuit), se uma falhar, aborta.
            if not found:
                return 0.0

        return 1.0
