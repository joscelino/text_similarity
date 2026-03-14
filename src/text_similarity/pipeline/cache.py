"""Módulo de gerência de cache via disco para otimizar tempo de pipeline."""

from __future__ import annotations

import hashlib
import pickle
import tempfile
from pathlib import Path
from typing import List, Optional

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

    def save_catalog(
        self, candidates: List[str], processed: List[str], cache_path: str
    ) -> None:
        """Salva candidatos processados em disco com hash de integridade.

        Args:
            candidates: Lista de textos originais dos candidatos.
            processed: Lista de textos já pré-processados.
            cache_path: Caminho do arquivo de cache em disco.
        """
        catalog_hash = hashlib.sha256("\n".join(candidates).encode("utf-8")).hexdigest()
        data = {
            "version": "1.0",
            "catalog_hash": catalog_hash,
            "processed": processed,
        }
        with open(Path(cache_path), "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_catalog(
        self, candidates: List[str], cache_path: str
    ) -> Optional[List[str]]:
        """Carrega candidatos do disco se hash bater.

        Args:
            candidates: Lista de textos originais para validar integridade.
            cache_path: Caminho do arquivo de cache em disco.

        Returns:
            Lista de textos processados se o cache for válido, None caso contrário.
        """
        path = Path(cache_path)
        if not path.exists():
            return None
        catalog_hash = hashlib.sha256("\n".join(candidates).encode("utf-8")).hexdigest()
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)  # noqa: S301
            if data.get("catalog_hash") == catalog_hash:
                return data["processed"]
        except (pickle.UnpicklingError, KeyError, EOFError):
            pass
        return None

    def clear(self) -> None:
        """Limpa todo o cache em disco mantido pelo Joblib."""
        self.memory.clear(warn=False)
