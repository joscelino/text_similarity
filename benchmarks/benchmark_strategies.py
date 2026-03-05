"""Benchmark comparativo: vectorized vs parallel.

Este script compara as estratégias 'vectorized' e 'parallel' da API
para demonstrar quando o multiprocessing traz ganhos reais.

Uso:
    uv run python benchmarks/benchmark_strategies.py
"""

from __future__ import annotations

import random
import time
from typing import List

from text_similarity.api import Comparator


def gerar_texto_sintetico(n_palavras: int = 8) -> str:
    """Gera uma frase sintética com palavras aleatórias."""
    palavras_base = [
        "arroz", "feijão", "carne", "frango", "peixe", "legume",
        "verdura", "fruta", "leite", "café", "açúcar", "sal",
        "óleo", "manteiga", "pão", "queijo", "presunto", "ovo",
        "macarrão", "molho", "farinha", "tempero", "alho", "cebola",
        "tomate", "batata", "cenoura", "abobrinha", "berinjela",
        "mesa", "cadeira", "sofá", "cama", "armário", "estante",
        "televisão", "computador", "celular", "fone", "carregador",
        "notebook", "tablet", "monitor", "teclado", "mouse",
        "samsung", "apple", "dell", "lenovo", "motorola", "xiaomi",
        "iphone", "galaxy", "redmi", "pixel", "ultra", "pro",
        "novo", "usado", "seminovo", "original", "importado",
        "comprar", "vender", "trocar", "alugar", "oferta",
    ]
    return " ".join(random.choices(palavras_base, k=n_palavras))


def gerar_queries(n: int) -> List[str]:
    """Gera N queries sintéticas."""
    return [gerar_texto_sintetico(random.randint(4, 8)) for _ in range(n)]


def gerar_candidatos(n: int) -> List[str]:
    """Gera N candidatos sintéticos."""
    return [gerar_texto_sintetico(random.randint(5, 15)) for _ in range(n)]


def benchmark(
    comp: Comparator, queries: List[str], candidates: List[str],
    strategy: str, top_n: int, min_cosine: float, n_workers: int | None = None,
) -> float:
    """Mede tempo de execução de compare_many_to_many."""
    inicio = time.perf_counter()
    comp.compare_many_to_many(
        queries, candidates, top_n=top_n, min_cosine=min_cosine,
        strategy=strategy, n_workers=n_workers,
    )
    return time.perf_counter() - inicio


def main() -> None:
    """Executa o benchmark comparativo."""
    random.seed(42)
    comp = Comparator.basic()

    configs = [
        {"n_queries": 10, "n_candidates": 1_000},
        {"n_queries": 10, "n_candidates": 10_000},
        {"n_queries": 50, "n_candidates": 10_000},
        {"n_queries": 100, "n_candidates": 10_000},
    ]

    header = (
        f"{'Queries':>8} | {'Candidatos':>12} | "
        f"{'Vectorized (s)':>15} | {'Parallel (s)':>14} | {'Speedup':>8}"
    )
    sep = "-" * len(header)

    print("\n" + "=" * len(header))
    print("  BENCHMARK: strategy='vectorized' vs strategy='parallel'")
    print("=" * len(header))
    print(header)
    print(sep)

    for cfg in configs:
        n_q = cfg["n_queries"]
        n_c = cfg["n_candidates"]

        queries = gerar_queries(n_q)
        candidates = gerar_candidatos(n_c)

        comp.clear_cache()
        t_vec = benchmark(
            comp, queries, candidates,
            strategy="vectorized", top_n=50, min_cosine=0.1,
        )

        comp.clear_cache()
        t_par = benchmark(
            comp, queries, candidates,
            strategy="parallel", top_n=50, min_cosine=0.1,
        )

        speedup = t_vec / t_par if t_par > 0 else float("inf")

        print(
            f"{n_q:>8} | {n_c:>12,} | "
            f"{t_vec:>15.3f} | {t_par:>14.3f} | {speedup:>7.1f}x"
        )

    print(sep)
    print("Concluído!\n")


if __name__ == "__main__":
    main()
