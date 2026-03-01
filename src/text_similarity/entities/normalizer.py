from __future__ import annotations

from .inspector import EntityInspector


class EntityNormalizer:
    """
    Normaliza o texto convertendo as entidades encontradas em tokens neutros
    para uso no backend de TF-IDF/Cosine.
    Ex: "Custa 30 reais" -> "Custa <money:30.0>".
    """

    def __init__(self, entities: list[str] = None):
        self.inspector = EntityInspector(entities)

    def normalize(self, text: str) -> str:
        """
        Substitui valores no texto por suas tags correspondentes de baixo pra cima.
        """
        matches = self.inspector.inspect(text)

        # Resolvemos da última para a primeira entidade no texto para
        # evitar que os offsets mudem e quebrem as substituições anteriores.
        # Precisamos remover sobreposições também.
        normalized = text

        # Filtra ocorrências com sobreposição
        valid_matches = []
        for match in sorted(matches, key=lambda m: m.start):
            # Se sobrepõe com a última validada, descarta
            if valid_matches and match.start < valid_matches[-1].end:
                continue
            valid_matches.append(match)

        valid_matches.sort(key=lambda m: m.start, reverse=True)

        for m in valid_matches:
            tag = f"<{m.entity_type}:{m.value}>"
            normalized = normalized[: m.start] + tag + normalized[m.end :]

        return normalized
