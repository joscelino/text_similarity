"""Módulo de gerência de cache via disco para otimizar tempo de pipeline."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

from joblib import Memory


class PipelineCache:
    """Gerenciador de cache para otimização de processamento no pipeline.

    Utiliza joblib.Memory para cache em disco (ideal para grandes catálogos) e hashes.
    Implementamos LRU/Memória para deduplicação rápida.
    """

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        """Inicializa a estrutura de cache persistente via Joblib.

        Args:
        cache_dir: Caminho para diretório de cache. Se None, usa var temporária.
        """
        if cache_dir is None:
            self.cache_dir = Path(tempfile.gettempdir()) / "text_similarity_cache"
        else:
            self.cache_dir = Path(cache_dir)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # O Memory do Joblib cuida do cache no disco e invalidação transparente
        self.memory = Memory(self.cache_dir, verbose=0)

        # O LRU em memória será gerenciado num dicionário limpo caso a
        # caso se necessário ou decoradores lru_cache na chamada da API.

    def hash_text(self, text: str) -> str:
        """Retorna uma chave SHA-256 única para o texto, já minúsculo."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def clear(self) -> None:
        """Limpa todo o cache em disco mantido pelo Joblib."""
        self.memory.clear(warn=False)
