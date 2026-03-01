from __future__ import annotations

class Comparator:
    """
    Classe principal para comparação de similaridade de textos em português.
    """

    def __init__(self, mode: str = "basic", **kwargs: object) -> None:
        self.mode = mode
        # TODO: inicializar estágios conforme o projeto

    @classmethod
    def basic(cls) -> "Comparator":
        return cls(mode="basic")

    @classmethod
    def smart(
        cls,
        entities: list[str] | None = None,
        backend: str = "regex_dateparser",
        semantic: bool = False,
    ) -> "Comparator":
        return cls(mode="smart", entities=entities, backend=backend, semantic=semantic)

    def compare(self, text1: str, text2: str) -> float:
        # TODO: implementar pipeline
        return 0.0

    def explain(self, text1: str, text2: str) -> dict[str, object]:
        # TODO: retornar scores detalhados
        return {"score": 0.0, "stages": {}}
