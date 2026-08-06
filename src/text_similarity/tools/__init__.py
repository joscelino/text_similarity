"""Utilitários operacionais (CLI) da biblioteca text-similarity-br.

Comandos disponíveis:

* :mod:`text_similarity.tools.migrate_index` — migra um índice legado
  (pickle/joblib do BM25Index ou DenseIndex) para o formato seguro
  ``tsbr-index-v2`` (JSON/NPZ + HMAC-SHA256).
"""
