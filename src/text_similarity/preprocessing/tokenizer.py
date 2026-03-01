from __future__ import annotations

import re


class Tokenizer:
    """
    Responsável por quebrar o texto limpo em tokens.
    Uma implementação simples baseada em regex que lida com o formato PT-BR
    e respeita as tags de entidade (<money:10>, etc).
    """

    def tokenize(self, text: str) -> list[str]:
        """
        Tokeniza o texto mas mantém as tags de entidade unidas.
        """
        # Encontra ou tags exatas <tipo:valor> ou palavras alfa-numéricas
        # que já foram limpas pelo text_cleaner.
        # Captura:
        # 1. <...> (Tags do Normalizer)
        # 2. Palavras compostas com hífen (S22-ultra)
        # 3. Palavras / números
        pattern = r"(<[^>]+>|[\w\-]+)"

        matches = re.finditer(pattern, text)
        return [m.group(1) for m in matches]
