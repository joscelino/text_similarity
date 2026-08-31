[Português](https://github.com/joscelino/text_similarity/blob/main/README.pt-br.md)

# Text Similarity PT-BR

[![CI Pipeline](https://github.com/joscelino/text_similarity/actions/workflows/pipeline.yaml/badge.svg)](https://github.com/joscelino/text_similarity/actions/workflows/pipeline.yaml)
[![Docs](https://readthedocs.org/projects/text-similarity/badge/?version=latest)](https://text-similarity.readthedocs.io/pt-br/latest/)
[![PyPI](https://img.shields.io/pypi/v/text-similarity-br)](https://pypi.org/project/text-similarity-br/)
[![Python](https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11%20|%203.12-blue)](https://www.python.org)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> An optimized Python library specialized in text similarity comparison for Brazilian Portuguese (PT-BR). Ideal for NLP systems, chatbots, sentiment analysis, and data matching where the language's peculiarities, money formatting, regional phonetics, and measurements influence the true intent and similarity of texts.

---

## 📚 Table of Contents

- [✨ Key Features](#-key-features)
- [Requirements](#requirements)
- [🚀 Installation](#-installation)
- [📖 How to Use](#-how-to-use)
  - [Basic Mode](#basic-mode-fast-and-simple)
  - [Smart Mode](#smart-mode-entities-and-phonetics)
  - [Semantic Mode](#semantic-mode-word-embeddings)
  - [Batch Processing](#batch-processing)
  - [Multi-Query Comparison](#multi-query-comparison-compare_many_to_many)
  - [Rank Fusion via RRF](#rank-fusion-via-rrf-fusion_strategyrrf)
  - [Parallel Execution](#parallel-execution-strategyparallel)
  - [Async Integration](#async-integration-fastapi-aiohttp)
  - [Re-Ranking Vector Database Results](#re-ranking-vector-database-results)
  - [Understanding Why They Matched (Explain)](#understanding-why-they-matched-explain)
  - [Using Only for Text Processing](#using-only-for-text-processing)
  - [Preprocessing Bypass](#preprocessing-bypass-preprocessfalse)
- [📊 DataFrame Integration](#-dataframe-integration)
- [⚡ High-Volume Performance](#-high-volume-performance)
- [🎯 Interpreting Scores](#-interpreting-scores)
- [📈 Weight Calibration (Grid Search)](#-weight-calibration-grid-search)
- [⚙️ Cache Configuration](#️-cache-configuration)
- [🔒 Security](#-security)
  - [Upgrade Guide](#upgrade-guide)
- [🔌 Extensibility](#-extensibility--registering-custom-entities)
- [🤝 Contributing](#-contributing)

---

## ✨ Key Features

- **Specialized Cleaning (TextCleaner):** Expansion of modern contractions ("vc" -> "você", "fds" -> "fim de semana") and accent handling focused on Brazilian Portuguese.
- **Entity Detection (EntityNormalizer):** Intelligent extraction and preservation of quantities before "brute-force cleaning" would destroy them. (Example: converts `R$ 30,00` to the unique tag `<money:30.0>`).
  - Money (`R$ 30,00`, `30 reais`)
  - Dates (`12/03/2023`, `ontem`)
  - Dimensions/Weights (`2kg`, `10 m`)
  - Product Models (`S22 Ultra`, `iPhone 13 Pro`)
- **Advanced Preprocessing:** Tokenization, removal of Portuguese _stopwords_, and Lemmatization (with native SpaCy `pt_core_news_sm` support).
- **Hybrid Comparisons:** Combined algorithms that go beyond words (Bag-of-Words).
  - *Cosine (TF-IDF)*: For lexical variation.
  - *BM25 (Okapi BM25)*: Alternative to TF-IDF, superior for short texts (products, models). Selectable via `indexing_strategy="bm25"`.
  - *Dense Index (sentence-transformers)*: Initial filter by dense semantic similarity, capturing synonyms without lexical overlap. Selectable via `indexing_strategy="dense"`.
  - *Edit Distance (Levenshtein)*: Fast, using `rapidfuzz` to detect typos.
  - *Phonetics (Adapted PT-BR Metaphone)*: Treats "cassaa" and "caça" as identical weights.
  - *Entity Intersection*: "Short-Circuit" logic that guarantees a match (very high score) if the essential search entity (e.g., `GN500`) is validated intact in longer texts.
- **Optimized Pipeline (Joblib Cache):** Native disk cache support. Large texts already processed by Regex/SpaCy stages do not consume processing again.
- **Optimized High-Volume Performance:** Pre-compiled regex, parallel preprocessing via `ProcessPoolExecutor`, spaCy batching with `nlp.pipe()`, persistent catalog cache on disk, and LRU cache for dateparser.

---

## Requirements

- **Python:** >= 3.9
- **Main dependencies:** `scikit-learn`, `rapidfuzz`, `sentence-transformers` (included since v0.4.0)
- **Optional dependency:** `spacy` + model `pt_core_news_sm` (for lemmatization)

---

## 🚀 Installation

```bash
# With uv (recommended)
uv add text-similarity-br

# With pip
pip install text-similarity-br
```

Starting from version 0.4.0, the package already includes `sentence-transformers` as a dependency, enabling **Semantic Similarity** without additional installation.

With optional SpaCy lemmatization support:

```bash
# With uv
uv add "text-similarity-br[nlp]"
uv run python -m spacy download pt_core_news_sm

# With pip
pip install "text-similarity-br[nlp]"
python -m spacy download pt_core_news_sm
```

---

## 📖 How to Use

The public API is designed around the `Comparator` facade, ensuring ease of use without hiding customizable power.

### Basic Mode (Fast and Simple)

Operates only on Bag-of-Words and spelling corrections (Levenshtein/Cosine). Ideal for high data volume and short texts.

```python
from text_similarity.api import Comparator

comp = Comparator.basic()

# Input texts are in Brazilian Portuguese (PT-BR).
score = comp.compare("iphone 13 pro", "iphone pro 13")
print(f"Similarity: {score:.2f}")  # Output ~0.8 to 1.0 depending on weight
```

### "Smart" Mode (Entities and Phonetics)

Natively activates extractors for Currency, Date, Dimensions, Product Models, and applies phonetic calculations. Accepts the parameters `fusion_strategy` (`"linear"` or `"rrf"`) and `rrf_k` to control ranking fusion in batch operations.

```python
from text_similarity.api import Comparator

comp = Comparator.smart()

# Input texts are in Brazilian Portuguese (PT-BR).
new_score = comp.compare("Foi me cobrado 30 reais", "O preço é R$ 30,00")

print(f"Smart Similarity: {new_score:.2f}")
# High result due to exact financial entity identification

# --- Perfect Model Intersection (Short-circuit) ---
# Input texts are in Brazilian Portuguese (PT-BR).
score_modelo = comp.compare("GN500", "Temos as peças GN 500, GN 1000 e SK 200")
print(f"Embedded Model Score: {score_modelo:.2f}")
# Result: ~0.95. When the searched model "GN500" is found isolated inside the
# long target text, the intersection algorithm directly ensures a high score,
# ignoring the rest of the long string that would cause dilution.
```

#### Filtering Specific Entities

By default, `smart` mode activates **all** extractors (`money`, `date`, `dimension`, `number`, `product_model`). You can restrict only to the entities relevant to your domain by passing the `entities` parameter:

```python
from text_similarity.api import Comparator

# Only product models — ideal for technical parts catalogs
comp = Comparator.smart(entities=["product_model"])

# Only monetary values — ideal for financial systems
comp_fin = Comparator.smart(entities=["money", "number"])

# Dates and dimensions — ideal for reports and technical datasheets
comp_lab = Comparator.smart(entities=["date", "dimension"])
```

> **Tip:** Filtering entities improves precision by avoiding false positives. An active `date` extractor in a product catalog could incorrectly map SKUs containing year digits.

#### Weight and Algorithm Options

When using `smart` mode, you can balance the following algorithms through the `weights` parameter (in the constructor) or `rrf_weights` (in average/batch functions):

| Option | Technical Name | What it evaluates | Best use |
| :--- | :--- | :--- | :--- |
| **`cosine`** | Cosine (TF-IDF) | Frequency and rarity of words. | Detect identical keywords. |
| **`bm25`** | Okapi BM25 | Relevance with term frequency saturation. | Short texts (products, SKUs). Enabled via `indexing_strategy="bm25"`. |
| **`edit`** | Levenshtein | Character proximity (spelling). | Capture typos. |
| **`phonetic`** | Phonetics (PT-BR) | Pronunciation of words in Portuguese. | Capture letter swaps with equal sound (e.g., S/Z/X). |
| **`semantic`** | Semantic | Meaning and context (Embeddings). | Find synonyms (e.g., "carro" vs "veículo"). |
| **`entity`** | Entities | Specific identifiers. | Ensure codes and models match. |

### Semantic Mode (Word Embeddings)

To capture the real semantic intent between synonyms that share no letters (e.g., `"veículo"` vs `"carro"`), you can activate the **Sentence-Transformers** engine.

```python
from text_similarity.api import Comparator

# Enables dense models under the hood
comp = Comparator.smart(use_embeddings=True)

# Input texts are in Brazilian Portuguese (PT-BR).
score = comp.compare("automóvel bicombustível", "carro flex")
print(f"Semantic Similarity: {score:.2f}")  # High score, unlike pure TF-IDF.
```

*Note: The first call in each isolated process may take a few extra milliseconds to load the PyTorch model into RAM. In batch methods (`compare_batch` / `strategy="parallel"`), Semantic Similarity acts as a final super-optimized evaluation only on the `top_n` returned by TF-IDF.*

### Batch Processing

For use cases where you need to compare a *query* against hundreds or thousands of candidates, use the `compare_batch` method. It is highly optimized by applying sparse matrices via Scikit-Learn and mathematical short-circuits, delivering consolidated results up to **~48x faster** depending on volume.

```python
from text_similarity.api import Comparator

comp = Comparator.smart()

# Input texts are in Brazilian Portuguese (PT-BR).
query = "Notebook Dell Inspiron 15"
candidates = [
    "Dell Inspiron 15 polegadas i5",
    "Notebook Lenovo Thinkpad",
    "Mouse sem fio logitech",
    # ... 10,000 other items
]

# Quickly filter by minimum TF-IDF (0.1) and extract the top 5
results = comp.compare_batch(query, candidates, top_n=5, min_cosine=0.1)

for r in results:
    print(f"Score: {r['score']:.2f} | Match: {r['candidate']}")
```

### Multi-Query Comparison (`compare_many_to_many`)

When you need to compare **multiple queries** against the same candidate catalog, use `compare_many_to_many`. It pre-computes the candidate TF-IDF matrix **only once**, eliminating redundant recalculations and delivering significant speedups in high-volume scenarios.

```python
from text_similarity.api import Comparator

comp = Comparator.smart()

queries = [
    "Notebook Dell Inspiron 15",
    "Mouse sem fio logitech",
    "Monitor Samsung 27 polegadas",
]
candidates = [
    "Dell Inspiron 15 polegadas i5",
    "Notebook Lenovo Thinkpad",
    "Mouse logitech wireless",
    "Monitor Samsung 27'' 4K",
    # ... thousands of items
]

# Returns a list of results for EACH query
all_results = comp.compare_many_to_many(
    queries, candidates, top_n=5, min_cosine=0.1
)

for query, results in zip(queries, all_results):
    print(f"\n🔍 Query: {query}")
    for r in results:
        print(f"  Score: {r['score']:.2f} | {r['candidate']}")
```

> **When to use which?**
> - `compare_batch()` → 1 query × N candidates (e.g., a user's text search).
> - `compare_many_to_many()` → M queries × N candidates (e.g., batch deduplication, database matching).

### Rank Fusion via RRF (`fusion_strategy="rrf"`)

By default, `Comparator` combines algorithm scores by **weighted sum** (strategy `"linear"`). For scenarios where raw algorithm scores have very different scales (e.g., mixing lexical with semantic), you can use **Reciprocal Rank Fusion (RRF)**, which is based on the **position** of candidates in each ranking rather than raw scores:

```python
from text_similarity.api import Comparator

# RRF: combines rankings by position, eliminating scale problems
comp = Comparator.smart(fusion_strategy="rrf")

results = comp.compare_batch(
    "Notebook Dell Inspiron",
    candidates,
    top_n=10,
    min_cosine=0.1,
)

# Each result includes RRF details: rank and contribution of each algorithm
for r in results:
    print(f"Score: {r['score']:.2f} | {r['candidate']}")
    print(f"  Details: {r['details']}")
```

The `rrf_k` parameter (default 60) controls smoothing: larger values attenuate the difference between positions in the ranking.

```python
# RRF with more aggressive smoothing
comp = Comparator.smart(fusion_strategy="rrf", rrf_k=100)
```

#### Per-Algorithm Weights (`rrf_weights`)

By default, all algorithms contribute equally in RRF. Use `rrf_weights` to give more importance to specific algorithms — for example, prioritizing semantic similarity over lexical search:

```python
from text_similarity.api import Comparator

# Prioritizes semantic (70%) over lexical (30%) in the final ranking
comp = Comparator.smart(
    use_embeddings=True,
    fusion_strategy="rrf",
    rrf_weights={"cosine": 0.3, "semantic": 0.7},
)

# Prioritizes phonetics for domains with frequent typos
comp_phon = Comparator.smart(
    fusion_strategy="rrf",
    rrf_weights={"cosine": 0.3, "edit": 0.2, "phonetic": 0.5},
)
```

The applied formula is: `score = Σ weight_i * 1/(k + rank_i)`. Algorithms not listed in `rrf_weights` receive a default weight of `1.0`.

> **When to use `"rrf"` vs `"linear"`:**
> - `fusion_strategy="linear"` (default) → When algorithms operate on similar scales and weights have been calibrated for your domain.
> - `fusion_strategy="rrf"` → When mixing algorithms with distinct scales (e.g., TF-IDF + Semantic), or when candidates consistently well-positioned across multiple rankings should be prioritized regardless of absolute score.
> - `rrf_weights` → When, in addition to using RRF, you want a particular algorithm to have more influence on the final ranking position.

Also available via direct import for advanced use — useful when you already have your own rankings (e.g., from Elasticsearch, a vector database, or custom algorithms) and want to fuse them:

```python
from text_similarity import RRFusion

# Each sublist is the ranking of ONE algorithm, ordered by descending score.
# Structure is: [{"candidate": str, "score": float}, ...]
rankings_by_algorithm = [
    # Ranking from the "cosine" algorithm
    [
        {"candidate": "Dell Inspiron 15 i5", "score": 0.92},
        {"candidate": "Notebook Lenovo", "score": 0.45},
        {"candidate": "Mouse Logitech", "score": 0.10},
    ],
    # Ranking from the "semantic" algorithm
    [
        {"candidate": "Dell Inspiron 15 i5", "score": 0.85},
        {"candidate": "Mouse Logitech", "score": 0.30},
        {"candidate": "Notebook Lenovo", "score": 0.20},
    ],
]

# Algorithm names, in the SAME ORDER as the sublists above
algorithm_names = ["cosine", "semantic"]

# Equal weights (default)
rrf = RRFusion(k=60)

# Or with per-algorithm weights
rrf = RRFusion(k=60, weights={"cosine": 0.4, "semantic": 0.6})

fused_ranking = rrf.fuse(rankings_by_algorithm, algorithm_names)

for item in fused_ranking:
    print(f"RRF Score: {item['score']:.3f} | {item['candidate']}")
    # Each item includes details: rank, raw_score, rrf_contribution, weight
```

> **Note:** In standard usage via `Comparator.smart(fusion_strategy="rrf")`, these rankings are built automatically by `Comparator`. The direct `RRFusion` import is for scenarios where you want to fuse rankings from external sources.

### Parallel Execution (`strategy="parallel"`)

For **high-volume** scenarios (50+ queries × 10k+ candidates), activate the parallel strategy that distributes queries across multiple processes via `ProcessPoolExecutor`:

```python
from text_similarity.api import Comparator

comp = Comparator.smart()

# Distributes across 4 processes (default: os.cpu_count())
results = comp.compare_many_to_many(
    queries, candidates, top_n=5, min_cosine=0.1,
    strategy="parallel", n_workers=4,
)

# Also works with compare_batch
result = comp.compare_batch(
    "single query", candidates, top_n=10,
    strategy="parallel", n_workers=4,
)
```

> **⚠️ When NOT to use `parallel`:** For few queries (< 20) or few candidates (< 5k), the overhead of process creation can outweigh the gain. Use `strategy="vectorized"` (default) in those cases.

### Async Integration (FastAPI, aiohttp)

For **async web servers**, use the `_async` methods that offload CPU-bound work to a `ProcessPoolExecutor`, keeping the event loop free:

```python
from fastapi import FastAPI
from text_similarity.api import Comparator

app = FastAPI()
comp = Comparator.smart()

@app.post("/search")
async def search(query: str, candidates: list[str]):
    results = await comp.compare_batch_async(
        query, candidates, top_n=10, n_workers=4
    )
    return {"results": results}

@app.post("/bulk-search")
async def bulk_search(queries: list[str], candidates: list[str]):
    results = await comp.compare_many_to_many_async(
        queries, candidates, top_n=5, n_workers=4
    )
    return {"results": results}
```

> **Available async methods:** `compare_batch_async()` and `compare_many_to_many_async()`. Both use `strategy="parallel"` internally.

### Re-Ranking Vector Database Results

When you already have results from a vector database (Pinecone, Qdrant, Milvus, PGVector, Elasticsearch) and want to **re-rank** them using PT-BR linguistic validation (edit, phonetics, entities), use `rerank_vector_results`. It works as a **Brazilian Portuguese linguistic Cross-Encoder**, applying `HybridSimilarity` algorithms over the results already filtered by the database.

```python
from text_similarity.api import Comparator

comp = Comparator.smart(entities=["product_model"])

# Results coming from your vector database (Qdrant, Pinecone, etc.)
vector_results = [
    {"id": "doc1", "text": "Peças industriais variadas", "score": 0.90},
    {"id": "doc2", "text": "Ferramentas GN série completa", "score": 0.80},
    {"id": "doc3", "text": "Motor elétrico trifásico", "score": 0.70},
    {"id": "doc4", "text": "Peças GN500 originais", "score": 0.45},
]

# Re-ranks using linguistic validation
reranked = comp.rerank_vector_results(
    "GN500",
    vector_results,
    preprocess_query=True,        # pipeline on the user query
    preprocess_candidates=True,   # pipeline on texts (if raw)
)

for r in reranked:
    print(f"Score: {r['score']:.2f} (vector: {r['vector_score']:.2f}) | {r['candidate']}")
# "Peças GN500 originais" rises from position #4 to #1 via entity short-circuit
```

The result includes:
- `id` — document identifier (preserved from input, if present)
- `candidate` — original text
- `score` — final HybridSimilarity score
- `vector_score` — original vector database score
- `details` — per-algorithm details (cosine, edit, phonetic, entity)

> **Input format:** Each candidate must have at least `"text"` (str) and `"score"` (float). The `"id"` field is optional.

> **Preprocessing:** Use `preprocess_candidates=False` (default) when database texts are already normalized. Use `True` when texts are raw and need cleaning/entity extraction.

> **RRF compatible:** Works with `fusion_strategy="rrf"` to combine rankings by position:
> ```python
> comp = Comparator.smart(entities=["product_model"], fusion_strategy="rrf")
> reranked = comp.rerank_vector_results("GN500", vector_results)
> ```

### Understanding Why They Matched (Explain)

Sometimes you need to debug user intent or show evidence that the algorithm crossover detected similarity. Use `.explain()`:

```python
from text_similarity.api import Comparator

comp = Comparator.smart()

# Input texts are in Brazilian Portuguese (PT-BR).
details = comp.explain("televisão samsung 55 polegadas", "tv samsung 55\"")

print(details["score"])
# 0.85
print(details["details"])
# {'cosine': 0.82, 'edit': 0.80, 'phonetic': 0.95} -> Very high phonetic match and detected 55 dimension.
```

> **Behavior with empty strings:** `explain("", "any text")` returns `{"score": 0.0, "details": {}}` without raising an exception.

> **Short-circuit in `explain()`:** When an entity is detected with total intersection (e.g., search for `<productmodel:GN500>` found in the target text), `explain()` returns `{"score": 0.95, "details": {"entity": {..., "short_circuit": True}}}`, just like `compare()`.

> **`compare_batch()` with empty list:** `comp.compare_batch("any", [])` returns `[]` immediately, without processing.

### Using Only for Text Processing

If your goal is not to perform comparisons, but only to take advantage of the robust Portuguese processing engine (to clean databases, train models, remove accents, expand contractions, and lemmatize), you can instantiate `Pipeline` stages autonomously and officially:

```python
from text_similarity.pipeline.pipeline import PreprocessingPipeline
from text_similarity.pipeline.backends import CleanTextStage, TokenizerStage, StopwordsStage

# Build your custom pipeline with only what you need:
pipeline = PreprocessingPipeline([
    CleanTextStage(),  # Contraction expansion ("vc" -> "você"), no accents, lowercase
    TokenizerStage(),  # Safe tokenization
    StopwordsStage()   # Removal of useless PT-BR connectives
])

# Input text is in Brazilian Portuguese (PT-BR).
raw_text = "Limpando meeu texto, crz... vc viu a promo???"
treated_text, stats = pipeline.process(raw_text)

print(treated_text)
# Expected output (treated bag of words): "limpar texto crz ver promo"
```

### Preprocessing Bypass (`preprocess=False`)

When your texts **have already been cleaned externally** (e.g., coming from an ETL pipeline, normalized database, or another NLP system), you can disable preprocessing to avoid redundant transformations and gain performance:

```python
from text_similarity.api import Comparator

comp = Comparator.smart()

# Texts already normalized by your external pipeline
clean1 = "samsung galaxy s22 ultra 256gb"
clean2 = "samsung galaxy s22 ultra 256gb preto"

# Bypasses cleaning, tokenization, stopwords, and lemmatization
score = comp.compare(clean1, clean2, preprocess=False)
print(f"Score: {score:.2f}")

# Also works with explain
details = comp.explain(clean1, clean2, preprocess=False)
```

Works in **all comparison methods**:

```python
# Batch — 1 query × N already-cleaned candidates
results = comp.compare_batch(
    "galaxy s22", cleaned_candidates,
    top_n=10, min_cosine=0.1, preprocess=False,
)

# Multi-query — M queries × N already-cleaned candidates
all_results = comp.compare_many_to_many(
    cleaned_queries, cleaned_candidates,
    top_n=5, preprocess=False,
)

# Async
results = await comp.compare_batch_async(
    "galaxy s22", cleaned_candidates,
    top_n=10, preprocess=False,
)
```

> **When to use `preprocess=False`:**
> - Data coming from ETL pipelines that already normalize text.
> - Re-ranking results already processed by another system (e.g., Elasticsearch, vector database).
> - Benchmarks where you want to isolate the cost of similarity algorithms without pipeline overhead.
>
> **Attention:** With `preprocess=False`, the in-memory cache **is not used** (there is no hash nor storage), and no pipeline stage is executed — including entity extraction. Make sure your texts are in the format expected by the algorithms.

---

## 📊 DataFrame Integration

The library automatically recognizes the DataFrame type used — **pandas, polars, cuDF, modin**, or any object subscriptable by column name. No additional dependency is installed: methods return `List[dict]`, and you convert to the DataFrame of your choice.

### DataFrame Search (`compare_dataframe`)

Compares a query against a text column and returns a `List[dict]` with the most similar rows, including all original keys and a `score` key:

```python
# pandas
import pandas as pd
from text_similarity.api import Comparator

comp = Comparator.smart(entities=["product_model"])

catalog = pd.DataFrame({
    "sku": ["A001", "A002", "A003", "A004"],
    "descricao": [
        "Notebook Dell Inspiron 15 i5",
        "Mouse Logitech MX Master 3",
        "Monitor Samsung 27 4K",
        "Teclado Mecânico Redragon",
    ],
    "preco": [3200, 450, 1800, 380],
})

results = comp.compare_dataframe(
    df=catalog,
    text_column="descricao",
    query="notebook dell inspiron",
    top_n=3,
    min_cosine=0.1,
)

# results is List[dict] — convert as you wish
df_result = pd.DataFrame(results)
print(df_result[["sku", "descricao", "preco", "score"]])
```

```python
# polars (no changes to the call needed)
import polars as pl

catalog_pl = pl.DataFrame({
    "sku": ["A001", "A002", "A003", "A004"],
    "descricao": [
        "Notebook Dell Inspiron 15 i5",
        "Mouse Logitech MX Master 3",
        "Monitor Samsung 27 4K",
        "Teclado Mecânico Redragon",
    ],
    "preco": [3200, 450, 1800, 380],
})

results = comp.compare_dataframe(catalog_pl, "descricao", "notebook dell inspiron")
df_result = pl.DataFrame(results)
```

### Record Linkage (`record_linkage`)

Compares two tables finding the most similar pairs — ideal for deduplication between suppliers or catalog matching. Returns `List[dict]`:

```python
import pandas as pd
from text_similarity.api import Comparator

comp = Comparator.smart()

table_a = pd.DataFrame({
    "id_a": [1, 2, 3],
    "produto_a": ["iPhone 13 Pro 256GB", "Samsung Galaxy S22", "Notebook Dell i5"],
})

table_b = pd.DataFrame({
    "id_b": [10, 20, 30, 40],
    "produto_b": [
        "Apple iPhone 13 Pro",
        "Galaxy S22 Ultra",
        "Dell Inspiron 15 i5 8GB",
        "Mouse sem fio Logitech",
    ],
})

pairs = comp.record_linkage(
    df_a=table_a,
    df_b=table_b,
    col_a="produto_a",
    col_b="produto_b",
    top_n=2,
    min_cosine=0.1,
)

# pairs is List[dict] — convert as you wish
df_pairs = pd.DataFrame(pairs)
print(df_pairs[["text_a", "text_b", "score"]])
```

Each dict contains: `index_a`, `text_a`, `index_b`, `text_b`, `score`, `details`.

> **When to use `compare_dataframe` vs `record_linkage`:**
> - `compare_dataframe` → 1 query × N DataFrame rows (a user's search).
> - `record_linkage` → M rows × N rows (deduplication between two bases, supplier matching).

---

## ⚡ High-Volume Performance

The library is optimized for high-volume scenarios (100+ queries × 100k+ candidates) with multiple techniques that significantly reduce processing time.

### Persistent Catalog Cache (`preprocess_catalog`)

When the same candidate catalog is reused across runs (e.g., daily matching rounds against a product base), use `preprocess_catalog()` to save preprocessed texts to disk. On the first run, it processes and saves. On subsequent runs, it loads directly — saving ~80% of total time.

```python
from text_similarity.api import Comparator

comp = Comparator.smart()

# First run: process + save to disk
candidates = ["Dell Inspiron 15", "Mouse Logitech MX", ...]  # 150k items
p_candidates = comp.preprocess_catalog(candidates, cache_path="my_catalog.pkl")

# Subsequent runs: load from disk instantly
p_candidates = comp.preprocess_catalog(candidates, cache_path="my_catalog.pkl")

# Use with compare_many_to_many + preprocess=False on already-processed candidates
results = comp.compare_many_to_many(
    queries, p_candidates, top_n=10, preprocess=False,
)
```

Invalidation is automatic via SHA-256 hash: if the catalog changes (items added, removed, or altered), the cache is reprocessed automatically.

### Automatic Parallel Preprocessing

For batches with more than 1,000 texts, `_process_batch()` automatically distributes work across multiple processes via `ProcessPoolExecutor`, without configuration. Compatible with Windows (`spawn`).

### Internal Optimizations

The following optimizations are applied automatically and require no user code changes:

| Optimization | Impact | Description |
|---|---|---|
| Pre-compiled regex | ~15-25% | All 12 regex patterns are compiled once at class level |
| Parallel preprocessing | ~40-60% | Large batches (>1k texts) are distributed across multiple processes |
| spaCy batch (`nlp.pipe()`) | ~20-40% | Lemmatization via spaCy uses batch processing instead of individual calls |
| Persistent cache | ~80% (re-run) | Processed catalogs are saved to disk and reused across runs |
| dateparser LRU cache | ~5-10% | Already resolved dates are cached in memory (up to 1024 entries) |
| Optimized phonetics | ~5-10% | Phonetic substitutions via compiled regex + map instead of sequential `.replace()` |

### BM25 Indexing (`indexing_strategy="bm25"`)

By default, the filtering pipeline uses TF-IDF + cosine. For scenarios with **short texts** (products, models, SKUs with 3-15 tokens), BM25 (Okapi BM25) offers superior ranking thanks to term frequency saturation and document length normalization.

```python
from text_similarity.api import Comparator

# BM25 as indexing strategy
comp = Comparator.smart(
    entities=["product_model"],
    indexing_strategy="bm25",
)

# Identical usage — the entire API works transparently with BM25
results = comp.compare_batch("samsung galaxy s22", candidates, top_n=10)

# Multi-query: BM25 index built once, reused by all queries
all_results = comp.compare_many_to_many(queries, candidates, top_n=5)

# Async (FastAPI, aiohttp, Starlette) — inherits indexing_strategy automatically
results = await comp.compare_batch_async("samsung galaxy s22", candidates, top_n=10)
all_results = await comp.compare_many_to_many_async(queries, candidates, top_n=5)
```

The parameters `bm25_k1` (frequency saturation) and `bm25_b` (length normalization) can be adjusted for your domain. For short products (3-8 tokens), `bm25_k1=1.5` and `bm25_b=0.3` reduce length penalty:

```python
# Optimized for short product catalogs
comp = Comparator.smart(
    indexing_strategy="bm25",
    bm25_k1=1.5,
    bm25_b=0.3,
)
```

#### BM25 + `entities=["product_model"]`: synergy for technical models

The combination `indexing_strategy="bm25"` + `entities=["product_model"]` is especially effective for catalogs with models in the format `XX.NNN.NN` (e.g., `QS.250.08`, `QG.418.17`).

The pipeline operates in **two distinct stages**:

| Stage | TF-IDF | BM25 |
|---|---|---|
| **Pre-filtering** | Vectorizes tokens (`qs`, `250`, `08`) — `250` has low IDF (appears in many candidates) | Same, but `<productmodel:QS25008>` (generated by the extractor) has very high IDF — exact candidate rises in ranking |
| **Final scoring** | `HybridSimilarity` (cosine + edit + phonetic + entity) — identical in both cases | Identical — entity short-circuit triggers `0.95` if the exact model is in `top_n` |

**With isolated TF-IDF:** the token `250` appears in `250.080.612`, `250.080.588`, etc. — the exact candidate may not enter `top_n` if there are many competitors with the same numeric prefix.

**With BM25 + entities:** the tag `<productmodel:QS25008>` is unique in the corpus → maximum IDF → exact candidate always enters `top_n` → short-circuit guarantees `score = 0.95`.

```python
from text_similarity.api import Comparator

candidates = [
    "QS.250.08",
    "250.080.612",
    "250.080.588",
    "250.080.342",
]

comp = Comparator.smart(
    entities=["product_model"],
    indexing_strategy="bm25",
)

results = comp.compare_batch("QS.250.08", candidates, top_n=4, min_cosine=0.0)
# results[0] → {"candidate": "QS.250.08", "score": 0.95}   ← short-circuit active
# results[1] → {"candidate": "250.080.612", "score": ~0.3}  ← no shared entity
```

> **When to use this combination:** technical parts catalogs, SKUs with dot or hyphen separators, industrial product references. For free-text searches (e.g., "quero um celular barato"), the combination adds no benefit — use `Comparator.smart()` without a specific `indexing_strategy`.

### Dense Indexing (`indexing_strategy="dense"`)

For scenarios where the query and candidates are **semantically equivalent but do not share words** (e.g., `"veículo flex"` vs `"carro bicombustível"`), the dense index uses `sentence-transformers` embeddings as an initial filter, capturing semantic similarity before `HybridSimilarity` even kicks in.

```python
from text_similarity.api import Comparator

# Dense index — solves the synonym recall gap
comp = Comparator.smart(
    indexing_strategy="dense",
)

# Candidate will be found even without lexical overlap
results = comp.compare_batch("veículo flex", candidates, top_n=10)
```

By default it uses the `paraphrase-multilingual-MiniLM-L12-v2` model (multilingual, includes PT-BR). To use another model:

```python
comp = Comparator.smart(
    indexing_strategy="dense",
    dense_model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
)
```

> **⚠️ Important limitation:** `DenseIndex` runs on CPU and takes ~5-10 minutes to index 150k documents. **Use only for small/medium catalogs (up to ~10k items).** For large volumes with semantic recall, use `rerank_vector_results` combined with an external vector database (Qdrant, Pinecone, etc.).

> **When to use `"dense"`:** Catalogs up to ~10k items with high lexical variation — synonyms, informal language, customer support.

> **Compatible with all features:** Dense works with `strategy="parallel"`, `fusion_strategy="rrf"`, `preprocess=False`, and async methods. The switch is transparent — just change `indexing_strategy`.

#### Impact Estimate: TF-IDF vs BM25 vs Dense

| Metric | TF-IDF | BM25 | Dense |
|---|---|---|---|
| Ranking quality (short texts) | Baseline | **+10-20% precision@10** | Variable by domain |
| Semantic recall (synonyms) | Low | Low | **High** |
| Indexing time (150k candidates) | ~2s | ~1-3s (comparable) | Not recommended* |
| Time per query | ~5ms (sparse matmul) | ~15-30ms (loop) | ~5-20ms |
| Memory | ~50MB (sparse matrix) | ~80-100MB (dicts) | ~200-500MB |

*\*On CPU, `DenseIndex` takes ~5-10 minutes to index 150k candidates. It is suitable only for catalogs up to ~10k items.*

**Recommendation:** use BM25 for product/SKU catalogs, TF-IDF for long-text bases or extreme query volume, and Dense only for catalogs up to ~10k items with high lexical variation between query and candidates.

### Optimization: Avoiding Semantic Recalculation with `indexing_strategy="dense"`

When `indexing_strategy="dense"` and `use_embeddings=True` are used simultaneously **with the same model**, the library automatically detects that the query and candidate embeddings have already been computed in the filtering phase by `DenseIndex` and **reuses the score** in the hybrid phase — eliminating redundant reencoding:

```python
# DenseIndex and SemanticSimilarity use the same model by default.
# No additional configuration is needed — the optimization is automatic.
comp = Comparator.smart(
    indexing_strategy="dense",
    use_embeddings=True,
)

results = comp.compare_batch("veículo flex", candidates, top_n=10)
# sentence-transformers encode() runs only ONCE per candidate,
# not twice (filtering + hybrid scoring).
```

> **When it does NOT occur:** If `dense_model_name` differs from the default `SemanticSimilarity` model, the models are distinct and reuse is not applied — each algorithm uses its own encoder.

### Unloading the Model from Memory (`unload_embeddings_model`)

After an intensive inference session, you can release the semantic model from RAM/VRAM:

```python
comp = Comparator.smart(use_embeddings=True)

# ... processing ...

# Releases the model from global memory
comp.unload_embeddings_model()

# The model will be automatically reloaded on the next semantic comparison
```

---

## 🎯 Interpreting Scores

The returned score ranges from `0.0` (completely different) to `1.0` (identical).

| Range | Interpretation |
|---|---|
| `>= 0.85` | Very strong match — likely duplicate or minimal description variation |
| `0.60 – 0.84` | Probable match — same item with different description (e.g., code with/without space) |
| `0.35 – 0.59` | Uncertain match — requires manual review |
| `< 0.35` | No relevant semantic relationship |

> **Tip:** For domains with product codes (materials, SKUs, technical parts), a threshold of `>= 0.60` is a good starting point. Calibrate with known pairs from your domain to adjust precision × recall.

---

## 📈 Weight Calibration (Grid Search)

To achieve the best precision in specific domains, you can calibrate the `HybridSimilarity` algorithm weights using `WeightCalibrator`. It allows testing multiple weight combinations against a "Gold Standard" dataset (manually annotated) and generates a detailed performance report comparing precision and time cost (latency).

```python
from text_similarity.api import Comparator
from text_similarity.tuning.calibrator import WeightCalibrator

comp = Comparator.smart()

# Gold Standard test dataset
gold_standard = [
    {"query": "casa", "target": "caza", "match": True},
    {"query": "celular", "target": "fone", "match": False},
]

# Weight configurations you want to compare
configs = [
    {"cosine": 0.5, "edit": 0.5},
    {"edit": 1.0},
    {"phonetic": 0.8, "cosine": 0.2},
]

calibrator = WeightCalibrator(comp, configs)
report = calibrator.evaluate(gold_standard)

# Displays the results dashboard (requires extra 'tuning')
report.summary()
```

To enable rich visualization (rich terminal dashboard):

```bash
# With uv
uv add "text-similarity-br[tuning]"

# With pip
pip install "text-similarity-br[tuning]"
```

---

## ⚙️ Cache Configuration

The library maintains an in-memory cache (SHA-256) to avoid reprocessing the same text multiple times through the pipeline. By default, the cache is **enabled**.

```python
from text_similarity.api import Comparator

# Cache enabled by default
comp = Comparator.smart(use_cache=True)

# Disable cache (useful in memory-limited environments or tests)
comp_no_cache = Comparator.smart(use_cache=False)
```

### Persistent Disk Cache

For high-volume scenarios with reusable catalogs, use `preprocess_catalog()` to save to disk and eliminate reprocessing between runs. See the [Persistent Catalog Cache](#persistent-catalog-cache-preprocess_catalog) section for details.

### Manually Clearing the Cache

Use `clear_cache()` when you need to force reprocessing — for example, after changing active entities or when freeing memory after a large batch:

```python
comp = Comparator.smart()

# Processes and stores in cache
comp.compare("product A", "product B")

# Frees all in-memory cache and clears disk cache (Joblib)
comp.clear_cache()
```

---

## 🔒 Security

This section summarizes the library's security choices. For the complete list of changes, see the release notes on GitHub.

### Index Authentication with `hmac_key`

`BM25Index` and `DenseIndex` use the `tsbr-index-v2` format, which includes a JSON header with metadata and an optional HMAC-SHA256. When saving, provide a key to guarantee the integrity and authenticity of the file on disk:

```python
from text_similarity.core.bm25 import BM25Index
from text_similarity.core.dense import DenseIndex

# Save with HMAC
index.save("catalog.tsbr-index", hmac_key=b"my-secret-32byte-key")

# Load validating HMAC
index = BM25Index.load("catalog.tsbr-index", hmac_key=b"my-secret-32byte-key")
```

> **Attention:** when `hmac_key` is not provided, the keyless SHA-256 `integrity_hash` still protects against accidental corruption, but **does not** guarantee authenticity. In production, always configure a secret key and store it in a secrets manager (e.g., environment variable, Vault, AWS Secrets Manager).

### Supply Chain: Pinning Model Revision

For production environments requiring reproducibility and control over HuggingFace models, the library allows pinning a specific revision of the embeddings model via `dense_model_revision`.

When provided, the SHA is propagated as `revision=<sha>` to `SentenceTransformer`, ensuring the same weights are loaded across all replicas and runs. The internal model cache key also includes `device` and `revision`, preventing incorrect reuse between distinct configurations.

```python
from text_similarity import Comparator

# Supply chain pin: loads exactly the informed revision
comp = Comparator.smart(
    use_embeddings=True,
    indexing_strategy="dense",
    dense_model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    dense_model_revision="a4b7c3d9e8f0123456789abcdef0123456789abcd",
)
```

> **Security:** the `dense_model_name` parameter **should not accept values directly provided by untrusted users**. When exposed in APIs or interfaces, apply a whitelist of allowed models in the host application.

> **Preference for SafeTensors:** when publishing or versioning your own models, prefer the `safetensors` format over arbitrary pickles, reducing the surface of insecure deserialization attacks.

### `SemanticSimilarity` Strict Mode

Starting from this version, `SemanticSimilarity` operates in strict mode by default (`strict=True`). Backend failures (missing model, OOM, CUDA error, etc.) are converted to `SemanticSimilarityError` instead of silently returning `0.0` and poisoning the ranking.

```python
from text_similarity import Comparator

# Recommended for production: failures are explicit
comp = Comparator.smart(use_embeddings=True, strict=True)

# Tolerant fallback: failures return 0.0 and stacktrace goes to log
comp_tolerant = Comparator.smart(use_embeddings=True, strict=False)
```

Use `strict=True` whenever ranking quality is critical; use `strict=False` only in exploration scenarios where a partial return is acceptable.

### `Comparator` Thread-Safety

`Comparator` is thread-safe for concurrent use. The in-memory cache is protected by `threading.Lock()`, allowing the same instance to be shared between threads of a `ThreadPoolExecutor` or concurrent requests of a web server:

```python
from concurrent.futures import ThreadPoolExecutor
from text_similarity import Comparator

comp = Comparator.smart()

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [
        executor.submit(comp.compare, "iphone 13", "iphone 13 pro")
        for _ in range(100)
    ]
    results = [f.result() for f in futures]
```

### Upgrade Guide

Indexes saved in versions ≤ 0.8.x use pickle/joblib and are **not automatically loaded** from this version. Migrate to the secure `tsbr-index-v2` format with the CLI utility:

```bash
python -m text_similarity.tools.migrate_index \
    legacy.pkl \
    new.tsbr-index \
    --i-accept-pickle-risk
```

The command reads the legacy file, converts it to the new format, and optionally applies HMAC. To apply authentication, pass `--hmac-key` (or use the `BM25Index.save` / `DenseIndex.save` API after migration).

See the release notes on GitHub for the complete list of BREAKING CHANGES.

---

## 🔌 Extensibility — Registering Custom Entities

The library exposes `ExtractorRegistry` to register custom entity extractors without modifying the source code:

```python
from text_similarity.entities.base import EntityExtractor, EntityMatch
from text_similarity.entities.registry import ExtractorRegistry

class CPFExtractor(EntityExtractor):
    """Example: CPF extractor for HR systems."""

    def extract(self, text: str) -> list[EntityMatch]:
        import re
        matches = []
        for m in re.finditer(r"\d{3}\.\d{3}\.\d{3}-\d{2}", text):
            matches.append(EntityMatch(
                entity_type="cpf",
                text_matched=m.group(),
                value=m.group().replace(".", "").replace("-", ""),
                start=m.start(),
                end=m.end(),
            ))
        return matches

# Register the custom extractor
ExtractorRegistry.register("cpf", CPFExtractor)

# Instantiate Comparator activating only your extractor
comp = Comparator.smart(entities=["cpf"])
score = comp.compare("019.283.847-09", "documento cpf 01928384709")
```

Default available extractors:

| Name | Examples detected |
|---|---|
| `money` | `R$ 30,00`, `50 reais`, `USD 100` |
| `date` | `12/03/2023`, `ontem`, `amanhã`, `25 de abril` |
| `dimension` | `2kg`, `1.5l`, `30cm`, `10m²` |
| `number` | `3`, `três`, `1000` |
| `product_model` | `S22 Ultra`, `iPhone 13`, `XJ-900`, `QS.250.08`, `QG.418.17` |

---

## 🤝 Contributing

Quality standards strictly followed: `Ruff` (lint + format) and `MyPy` (strong typing).

### Workflow

- **Development branch:** `dev` — all development happens here
- Create feature branches from `dev` and open PRs back to `dev`
- Merges to `main` are done only on releases

### Before Opening a PR

```bash
# Lint and formatting
uv run ruff check src tests
uv run ruff format src tests

# Type checking
uv run mypy src

# Tests
uv run pytest tests/
```

### Reporting Bugs / Suggestions

Open a [GitHub issue](https://github.com/joscelino/text_similarity/issues) describing:
- Library version (`pip show text-similarity-br`)
- Python version
- Minimal reproducible example
