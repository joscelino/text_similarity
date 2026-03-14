"""Módulo para subdivisão de strings em tokens e limites de palavras."""

from __future__ import annotations

import re


class Tokenizer:
    """Responsável por quebrar o texto limpo em tokens.

    Uma implementação simples baseada em regex que lida com o formato PT-BR
    e respeita as tags de entidade (<money:10>, etc).
    """

    _RE_TOKEN = re.compile(r"(<[^>]+>|[\w\-]+)")

    def tokenize(self, text: str) -> list[str]:
        """Tokeniza o texto mas mantém as tags de entidade unidas."""
        return [m.group(1) for m in self._RE_TOKEN.finditer(text)]
