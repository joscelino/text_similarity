"""Módulo de gerência de cache via disco para otimizar tempo de pipeline."""

from __future__ import annotations

import hashlib
import json
import tempfile
import warnings
from pathlib import Path
from typing import List, Optional, cast

from joblib import Memory

from text_similarity.core._serialization import SecurityWarning


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

        O arquivo é gravado em JSON UTF-8 (sem pickle) para mitigar riscos
        de desserialização insegura.

        Args:
            candidates: Lista de textos originais dos candidatos.
            processed: Lista de textos já pré-processados.
            cache_path: Caminho do arquivo de cache em disco.
        """
        catalog_hash = hashlib.sha256("\n".join(candidates).encode("utf-8")).hexdigest()
        data = {
            "version": "2.0",
            "catalog_hash": catalog_hash,
            "processed": processed,
        }
        with open(Path(cache_path), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def _looks_like_pickle(data: bytes) -> bool:
        """Heurística: arquivos pickle/joblib não começam com '{'."""
        if not data:
            return False
        return not data.lstrip().startswith(b"{")

    def load_catalog(
        self, candidates: List[str], cache_path: str
    ) -> Optional[List[str]]:
        """Carrega candidatos do disco se hash bater.

        Arquivos em formato legado (pickle/joblib) são detectados, ignorados
        e um ``SecurityWarning`` é emitido; o caller deve reprocessar o
        catálogo.

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
            raw = path.read_bytes()
            if self._looks_like_pickle(raw):
                warnings.warn(
                    "Cache legado em pickle detectado e ignorado. "
                    "Apague o arquivo para reprocessar em formato JSON.",
                    SecurityWarning,
                    stacklevel=2,
                )
                return None
            data = json.loads(raw.decode("utf-8"))
            if data.get("catalog_hash") == catalog_hash:
                return cast(List[str], data["processed"])
        except (json.JSONDecodeError, KeyError, EOFError):
            raise ValueError(f"Arquivo de cache não é JSON válido: {path}")
        return None

    def clear(self) -> None:
        """Limpa todo o cache em disco mantido pelo Joblib."""
        self.memory.clear(warn=False)
