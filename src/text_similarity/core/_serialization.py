r"""Serialização segura de índices (BM25, Dense) com HMAC-SHA256.

Este módulo centraliza o formato ``tsbr-index-v2``: um envelope binário
autenticado que substitui a serialização baseada em ``pickle``/``joblib``,
mitigando o risco descrito em OWASP A08:2021 – *Software and Data
Integrity Failures* (execução arbitrária de código via desserialização
de dados não confiáveis).

Formato de arquivo (``tsbr-index-v2``)
--------------------------------------

Um arquivo tem exatamente duas partes separadas por um único ``\n``::

    <HEADER_JSON_LINE>\n<PAYLOAD_BYTES>

* ``HEADER_JSON_LINE`` – JSON UTF-8 minificado com as chaves fixas
  ``{"format": "tsbr-index-v2", "type": "<BM25Index|DenseIndex|...>",
  "version": "2.0", "hmac": "<hex ou string vazia>"}``.
* ``PAYLOAD_BYTES`` – bytes brutos do payload. Para BM25Index é um JSON
  UTF-8; para DenseIndex é ``[len_meta:uint32 BE][meta_json][npz_bytes]``.

O HMAC-SHA256 é calculado sobre ``PAYLOAD_BYTES`` inteiro. A verificação
acontece ANTES de qualquer parse do payload — o header é intencionalmente
pequeno e possui esquema fixo, sendo o único trecho parseado antes da
validação (para descobrir o valor de HMAC esperado).

Chave HMAC
----------

* Fornecida via parâmetro ``hmac_key`` (``bytes`` ou ``str``); ou
* Variável de ambiente ``TSBR_HMAC_KEY``.

Sem chave, o arquivo é gravado/carregado sem autenticação e um
``warnings.warn`` alerta o usuário. Nesse caso a integridade contra
corrupção acidental é garantida apenas pelo ``integrity_hash`` SHA-256
opcional (checksum não autenticado) presente nos índices — para
autenticação real contra adulteração, configure ``hmac_key``.
"""

from __future__ import annotations

import hashlib
import hmac as _hmac
import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

__all__ = [
    "INDEX_FORMAT_VERSION",
    "INDEX_FORMAT_MAGIC",
    "HMAC_ENV_VAR",
    "SecurityWarning",
    "dump_index",
    "load_index",
    "dump_authenticated_bytes",
    "load_authenticated_bytes",
    "resolve_hmac_key",
    "looks_like_legacy_pickle",
    "sentence_transformers_install_hint",
]

INDEX_FORMAT_VERSION: str = "2.0"
"""Versão do formato serializado autenticado usado por BM25Index/DenseIndex."""

INDEX_FORMAT_MAGIC: str = "tsbr-index-v2"
"""Identificador (magic string) do envelope autenticado."""

HMAC_ENV_VAR: str = "TSBR_HMAC_KEY"
"""Variável de ambiente inspecionada quando ``hmac_key`` não é passada."""


class SecurityWarning(UserWarning):
    """Emitida quando um índice legado (pickle/joblib) é carregado.

    Sinaliza que o usuário optou explicitamente por rodar código
    desserializado (``allow_legacy_pickle=True``) e assumiu o risco.
    """


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------


def resolve_hmac_key(
    hmac_key: Union[bytes, str, None],
) -> Optional[bytes]:
    """Resolve a chave HMAC efetiva a partir de parâmetro ou variável de env.

    Args:
        hmac_key: Chave em ``bytes`` ou ``str``. Se ``None``, tenta ler
            ``os.environ['TSBR_HMAC_KEY']``.

    Returns:
        A chave em ``bytes`` ou ``None`` se não houver chave alguma.
    """
    if hmac_key is None:
        env_val = os.environ.get(HMAC_ENV_VAR)
        if env_val:
            return env_val.encode("utf-8")
        return None
    if isinstance(hmac_key, str):
        return hmac_key.encode("utf-8")
    return bytes(hmac_key)


def looks_like_legacy_pickle(data: bytes) -> bool:
    """Heurística: identifica se ``data`` parece ser um arquivo pickle/joblib.

    O novo formato começa OBRIGATORIAMENTE com ``{`` (header JSON). Qualquer
    outro início é tratado como possivelmente legado.

    Args:
        data: Prefixo (ou totalidade) do arquivo em ``bytes``.

    Returns:
        ``True`` se os bytes não começam com o marcador do formato novo.
    """
    if not data:
        return False
    return not data.startswith(b"{")


def _build_header(type_name: str, mac_hex: str) -> bytes:
    header: Dict[str, str] = {
        "format": INDEX_FORMAT_MAGIC,
        "type": type_name,
        "version": INDEX_FORMAT_VERSION,
        "hmac": mac_hex,
    }
    return json.dumps(header, sort_keys=True, ensure_ascii=False).encode("utf-8")


def _compute_hmac(payload_bytes: bytes, key: Optional[bytes]) -> str:
    if key is None:
        return ""
    return _hmac.new(key, payload_bytes, hashlib.sha256).hexdigest()


def _validate_header_and_split(
    raw: bytes,
) -> Tuple[Dict[str, Any], bytes]:
    """Separa header (JSON pequeno) do payload bruto sem parsear o payload."""
    nl = raw.find(b"\n")
    if nl < 0:
        raise ValueError(
            "Arquivo de índice inválido: separador de header ausente. "
            "O arquivo pode estar corrompido ou não seguir o formato "
            f"'{INDEX_FORMAT_MAGIC}'."
        )
    header_bytes = raw[:nl]
    payload_bytes = raw[nl + 1 :]

    try:
        header_obj = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Header inválido no arquivo de índice: {exc}") from exc

    if not isinstance(header_obj, dict):
        raise ValueError("Header do índice deve ser um objeto JSON.")
    if header_obj.get("format") != INDEX_FORMAT_MAGIC:
        raise ValueError(
            f"Formato de índice desconhecido: {header_obj.get('format')!r}. "
            f"Esperado {INDEX_FORMAT_MAGIC!r}."
        )
    if header_obj.get("version") != INDEX_FORMAT_VERSION:
        raise ValueError(
            f"Versão de índice incompatível: esperada "
            f"{INDEX_FORMAT_VERSION!r}, encontrada "
            f"{header_obj.get('version')!r}."
        )
    return header_obj, payload_bytes


# ---------------------------------------------------------------------------
# API pública – bytes cruas (útil para DenseIndex/NPZ)
# ---------------------------------------------------------------------------


def dump_authenticated_bytes(
    payload_bytes: bytes,
    path: Union[str, Path],
    *,
    type_name: str,
    hmac_key: Union[bytes, str, None] = None,
) -> None:
    """Grava ``payload_bytes`` em ``path`` com envelope HMAC.

    Args:
        payload_bytes: Bytes brutos do payload (JSON, NPZ, etc.).
        path: Caminho de saída.
        type_name: Identificador do tipo do índice (ex: ``"BM25Index"``).
        hmac_key: Chave HMAC. Se ``None``, tenta ``TSBR_HMAC_KEY``;
            se ainda assim vazia, grava sem HMAC e emite ``warnings.warn``.
    """
    key = resolve_hmac_key(hmac_key)
    if key is None:
        warnings.warn(
            "Nenhuma chave HMAC fornecida (nem via parâmetro nem via "
            f"variável de ambiente {HMAC_ENV_VAR!r}). O arquivo será "
            "gravado sem autenticação e ficará vulnerável a adulteração "
            "silenciosa. Para autenticar, configure hmac_key.",
            stacklevel=3,
        )
    mac_hex = _compute_hmac(payload_bytes, key)
    header_bytes = _build_header(type_name, mac_hex)
    path = Path(path)
    with open(path, "wb") as fh:
        fh.write(header_bytes)
        fh.write(b"\n")
        fh.write(payload_bytes)


def load_authenticated_bytes(
    path: Union[str, Path],
    *,
    hmac_key: Union[bytes, str, None] = None,
    allow_legacy_pickle: bool = False,
    expected_type: Optional[str] = None,
) -> Tuple[Dict[str, Any], bytes]:
    """Lê arquivo autenticado e retorna ``(header, payload_bytes)``.

    A validação HMAC acontece ANTES de qualquer parse do payload. Se o
    arquivo não estiver no formato novo, comporta-se conforme a política:

    * ``allow_legacy_pickle=False`` (padrão): levanta ``ValueError``.
    * ``allow_legacy_pickle=True``: emite ``SecurityWarning`` e devolve
      ``({"format": "legacy-pickle"}, raw_bytes)`` — o caller decide
      como desempacotar (ex.: ``joblib.load`` sob risco explícito).

    Args:
        path: Caminho do arquivo.
        hmac_key: Chave HMAC ou ``None``.
        allow_legacy_pickle: Opt-in para tolerar arquivos antigos
            (pickle/joblib). Padrão ``False``.
        expected_type: Se fornecido, valida ``header['type']``.

    Returns:
        Tupla ``(header_dict, payload_bytes)``. Para arquivos legados,
        ``header_dict`` conterá ``{"format": "legacy-pickle"}`` e
        ``payload_bytes`` conterá os bytes brutos do arquivo original.

    Raises:
        ValueError: Se o arquivo for legado (pickle) sem
            ``allow_legacy_pickle=True``, ou se o HMAC falhar, ou se o
            header for inválido.
    """
    path = Path(path)
    with open(path, "rb") as fh:
        raw = fh.read()

    if looks_like_legacy_pickle(raw):
        if not allow_legacy_pickle:
            raise ValueError(
                "Arquivo de índice em formato legado (pickle/joblib) "
                "detectado. Carregar esse formato executa código Python "
                "arbitrário embutido no arquivo e é considerado inseguro "
                "(OWASP A08:2021). Se você confia na origem do arquivo, "
                "opte-in explicitamente passando allow_legacy_pickle=True, "
                "ou migre para o formato novo com "
                "`python -m text_similarity.tools.migrate_index`."
            )
        warnings.warn(
            "Carregando índice em formato legado (pickle/joblib). Isso "
            "executa código arbitrário embutido no arquivo. Migre para o "
            f"formato {INDEX_FORMAT_MAGIC!r} o quanto antes.",
            SecurityWarning,
            stacklevel=3,
        )
        return {"format": "legacy-pickle"}, raw

    header, payload_bytes = _validate_header_and_split(raw)

    if expected_type is not None and header.get("type") != expected_type:
        raise ValueError(
            f"Tipo de índice inválido: esperado {expected_type!r}, "
            f"encontrado {header.get('type')!r}."
        )

    stored_hmac = header.get("hmac", "") or ""
    key = resolve_hmac_key(hmac_key)

    if key is not None:
        if not stored_hmac:
            raise ValueError(
                "Chave HMAC fornecida, mas o arquivo foi gravado sem HMAC. "
                "Regrave o índice com hmac_key ou remova a chave para "
                "carregar sem autenticação (não recomendado)."
            )
        computed = _compute_hmac(payload_bytes, key)
        if not _hmac.compare_digest(computed, stored_hmac):
            raise ValueError(
                "Falha de verificação HMAC: o arquivo foi adulterado ou "
                "a chave HMAC está incorreta. O payload NÃO será parseado."
            )
    else:
        if stored_hmac:
            warnings.warn(
                "O arquivo possui HMAC, mas nenhuma chave foi fornecida "
                "para validar. Prosseguindo sem autenticação — configure "
                f"hmac_key ou {HMAC_ENV_VAR} para verificar integridade.",
                stacklevel=3,
            )
        else:
            warnings.warn(
                "Arquivo carregado sem verificação HMAC (nem o arquivo "
                "nem o caller forneceram chave). A integridade contra "
                "adulteração NÃO é garantida.",
                stacklevel=3,
            )

    return header, payload_bytes


# ---------------------------------------------------------------------------
# API pública – dicts JSON (útil para BM25Index e afins)
# ---------------------------------------------------------------------------


def dump_index(
    payload: Dict[str, Any],
    path: Union[str, Path],
    *,
    hmac_key: Union[bytes, str, None] = None,
) -> None:
    """Serializa ``payload`` (dict JSON-safe) para ``path`` com HMAC.

    O ``payload`` deve conter a chave ``"type"`` (nome do índice), que é
    replicada no header do arquivo para permitir validação rápida antes
    do parse.

    Args:
        payload: Dicionário JSON-serializável. Deve conter ``"type"``.
        path: Caminho de saída.
        hmac_key: Chave HMAC-SHA256 (``bytes`` ou ``str``). Se ``None``,
            tenta ``TSBR_HMAC_KEY``; se ainda vazia, grava sem HMAC e
            emite ``warnings.warn``.

    Raises:
        TypeError: Se ``payload`` não for dict.
        ValueError: Se ``payload["type"]`` estiver ausente.
    """
    if not isinstance(payload, dict):
        raise TypeError(f"payload deve ser dict, recebido {type(payload).__name__}")
    type_name = payload.get("type")
    if not isinstance(type_name, str) or not type_name:
        raise ValueError("payload['type'] é obrigatório e deve ser string.")

    payload_bytes = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    dump_authenticated_bytes(
        payload_bytes,
        path,
        type_name=type_name,
        hmac_key=hmac_key,
    )


def load_index(
    path: Union[str, Path],
    *,
    hmac_key: Union[bytes, str, None] = None,
    allow_legacy_pickle: bool = False,
) -> Dict[str, Any]:
    """Carrega e valida um índice serializado com ``dump_index``.

    A validação HMAC ocorre ANTES de qualquer parse do JSON de payload.

    Args:
        path: Caminho do arquivo.
        hmac_key: Chave HMAC ou ``None``.
        allow_legacy_pickle: Se ``True``, permite carregar arquivos
            legados (pickle/joblib) via ``joblib.load``, emitindo
            ``SecurityWarning``. Padrão ``False`` (arquivo legado é
            rejeitado com ``ValueError``).

    Returns:
        O dicionário original passado para ``dump_index``. Para
        arquivos legados, retorna o dict desempacotado por joblib.

    Raises:
        ValueError: Header inválido, HMAC inválido, ou arquivo legado
            sem ``allow_legacy_pickle=True``.
    """
    header, payload_bytes = load_authenticated_bytes(
        path,
        hmac_key=hmac_key,
        allow_legacy_pickle=allow_legacy_pickle,
    )
    if header.get("format") == "legacy-pickle":
        # Só aqui, sob opt-in explícito, invocamos joblib.
        import joblib

        return joblib.load(path)  # type: ignore[no-any-return]

    try:
        obj = json.loads(payload_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Payload do índice não é JSON válido: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError("Payload do índice deve ser um objeto JSON.")
    return obj


def sentence_transformers_install_hint(caller: str) -> str:
    """Retorna mensagem de instalação do extra ``[semantic]``.

    Centraliza o hint de instalação de ``sentence-transformers`` para
    evitar duplicação de strings literais entre ``DenseIndex`` e
    ``SemanticSimilarity``.

    Args:
        caller: Nome do consumidor (ex: ``"DenseIndex"``,
            ``"SemanticSimilarity"``).

    Returns:
        Mensagem formatada indicando como instalar o extra.
    """
    return (
        f"{caller} requer sentence-transformers. "
        "Instale com: pip install text-similarity-br[semantic]  "
        "ou: uv add text-similarity-br[semantic]"
    )
