r"""CLI para migrar índices legados (pickle/joblib) para ``tsbr-index-v2``.

Uso::

    python -m text_similarity.tools.migrate_index legacy.pkl new.tsbr-index \
        --index-type bm25 \
        --hmac-env TSBR_HMAC_KEY \
        --i-accept-pickle-risk

Motivação
---------

O formato antigo dos índices era baseado em ``joblib.dump``/``pickle``,
o que expõe os usuários a execução de código arbitrário se um arquivo
de origem não confiável for carregado (OWASP A08:2021).

Esta ferramenta permite que usuários com índices já salvos em produção
migrem para o novo formato **sem executar pickle em produção**: a
desempacotagem legada acontece uma única vez, num ambiente isolado e
com consentimento explícito via ``--i-accept-pickle-risk``.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import List, Optional, Sequence

from ..core._serialization import (
    HMAC_ENV_VAR,
    INDEX_FORMAT_MAGIC,
    INDEX_FORMAT_VERSION,
    load_index,
    looks_like_legacy_pickle,
)
from ..core.bm25 import BM25Index
from ..core.dense import DenseIndex


class MigrationError(RuntimeError):
    """Erro fatal durante a migração de índice."""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m text_similarity.tools.migrate_index",
        description=(
            "Migra um índice legado (pickle/joblib) para o formato "
            f"seguro {INDEX_FORMAT_MAGIC} (versão {INDEX_FORMAT_VERSION})."
        ),
    )
    parser.add_argument(
        "legacy_path",
        type=Path,
        help="Caminho do arquivo legado (pickle/joblib).",
    )
    parser.add_argument(
        "new_path",
        type=Path,
        help="Caminho de saída do índice no formato novo.",
    )
    parser.add_argument(
        "--index-type",
        choices=("bm25", "dense"),
        required=True,
        help="Tipo do índice a ser migrado.",
    )
    key_group = parser.add_mutually_exclusive_group()
    key_group.add_argument(
        "--hmac-key",
        default=None,
        help=(
            "Chave HMAC-SHA256 (string). Se omitida, tenta --hmac-env; se "
            "ambos ausentes, o arquivo será gravado SEM autenticação."
        ),
    )
    key_group.add_argument(
        "--hmac-env",
        default=None,
        help=(
            "Nome de variável de ambiente contendo a chave HMAC. "
            f"Padrão implícito: {HMAC_ENV_VAR}."
        ),
    )
    parser.add_argument(
        "--i-accept-pickle-risk",
        action="store_true",
        help=(
            "Confirmação explícita de que o usuário entende o risco de "
            "desempacotar um pickle (execução de código arbitrário) e "
            "confia na origem do arquivo. OBRIGATÓRIA."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Sobrescreve new_path se já existir.",
    )
    return parser


def _resolve_hmac_key_from_args(
    hmac_key: Optional[str],
    hmac_env: Optional[str],
) -> Optional[bytes]:
    if hmac_key is not None:
        return hmac_key.encode("utf-8")
    env_name = hmac_env or HMAC_ENV_VAR
    env_val = os.environ.get(env_name)
    if env_val:
        return env_val.encode("utf-8")
    return None


def _load_legacy_bm25(legacy_path: Path) -> BM25Index:
    """Carrega índice BM25 legacy via ``allow_legacy_pickle=True``."""
    payload = load_index(legacy_path, allow_legacy_pickle=True)
    data = payload["data"]
    idx = BM25Index(k1=data["k1"], b=data["b"])
    idx._corpus_size = int(data["corpus_size"])
    idx._avgdl = float(data["avgdl"])
    idx._doc_freqs = dict(data["doc_freqs"])
    idx._doc_lens = list(data["doc_lens"])
    idx._term_freqs = [dict(tf) for tf in data["term_freqs"]]
    return idx


def _load_legacy_dense(legacy_path: Path) -> DenseIndex:
    """Carrega índice Dense legacy via ``allow_legacy_pickle=True``."""
    return DenseIndex.load(legacy_path, allow_legacy_pickle=True)


def _verify_new_file_format(new_path: Path) -> None:
    """Confirma que ``new_path`` está no formato ``tsbr-index-v2``."""
    with open(new_path, "rb") as fh:
        head = fh.read(4096)
    if looks_like_legacy_pickle(head):
        raise MigrationError(
            f"Arquivo migrado {new_path!r} NÃO está no formato "
            f"{INDEX_FORMAT_MAGIC}. Migração abortada."
        )
    # Primeira linha deve conter o magic string do header.
    header_line = head.split(b"\n", 1)[0]
    if INDEX_FORMAT_MAGIC.encode("utf-8") not in header_line:
        raise MigrationError(
            f"Header do arquivo migrado não contém {INDEX_FORMAT_MAGIC!r}."
        )


def migrate(
    legacy_path: Path,
    new_path: Path,
    *,
    index_type: str,
    hmac_key: Optional[bytes] = None,
    accepted_pickle_risk: bool = False,
    force: bool = False,
) -> None:
    """Executa a migração ``legacy_path`` → ``new_path`` (formato v2).

    Args:
        legacy_path: Arquivo em pickle/joblib.
        new_path: Destino do arquivo no formato novo.
        index_type: ``"bm25"`` ou ``"dense"``.
        hmac_key: Chave HMAC (bytes) para o novo arquivo. Se ``None``,
            grava sem autenticação (com warnings.warn).
        accepted_pickle_risk: Confirmação explícita de aceite do risco
            de desempacotar pickle. Obrigatória.
        force: Sobrescreve ``new_path`` se existir.

    Raises:
        MigrationError: Caso alguma etapa falhe.
    """
    if not accepted_pickle_risk:
        raise MigrationError(
            "Migração abortada: passe --i-accept-pickle-risk para "
            "confirmar que você entende que desempacotar o arquivo "
            "legado pode executar código arbitrário embutido nele. "
            "Rode apenas em ambiente isolado e com arquivo de origem "
            "confiável."
        )

    if not legacy_path.exists():
        raise MigrationError(f"Arquivo legado não encontrado: {legacy_path!r}")

    if new_path.exists() and not force:
        raise MigrationError(
            f"Arquivo de destino já existe: {new_path!r}. "
            "Use --force para sobrescrever."
        )

    new_path.parent.mkdir(parents=True, exist_ok=True)

    if index_type == "bm25":
        with warnings.catch_warnings():
            # SecurityWarning já é reforçado abaixo; suprimimos duplicado
            warnings.simplefilter("default")
            idx_bm25 = _load_legacy_bm25(legacy_path)
        idx_bm25.save(new_path, hmac_key=hmac_key)
    elif index_type == "dense":
        with warnings.catch_warnings():
            warnings.simplefilter("default")
            idx_dense = _load_legacy_dense(legacy_path)
        idx_dense.save(new_path, hmac_key=hmac_key)
    else:  # pragma: no cover — argparse já valida choices
        raise MigrationError(f"Tipo de índice desconhecido: {index_type!r}")

    _verify_new_file_format(new_path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entrada CLI. Retorna código de saída (0 = sucesso)."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        hmac_key = _resolve_hmac_key_from_args(args.hmac_key, args.hmac_env)
        migrate(
            args.legacy_path,
            args.new_path,
            index_type=args.index_type,
            hmac_key=hmac_key,
            accepted_pickle_risk=args.i_accept_pickle_risk,
            force=args.force,
        )
    except MigrationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 — CLI top-level
        print(f"error: falha inesperada durante migração: {exc}", file=sys.stderr)
        return 1

    print(
        f"OK: índice migrado para {args.new_path!r} no formato "
        f"{INDEX_FORMAT_MAGIC} v{INDEX_FORMAT_VERSION}."
    )
    if hmac_key is None:
        print(
            "AVISO: arquivo gravado SEM autenticação HMAC. "
            f"Configure --hmac-key ou ${HMAC_ENV_VAR} para autenticar.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))


# Reexport para docs / import externo
__all__: List[str] = ["main", "migrate", "MigrationError"]
