# Text Similarity PT-BR

[![CI Pipeline](https://github.com/joscelino/text_similarity/actions/workflows/pipeline.yaml/badge.svg)](https://github.com/joscelino/text_similarity/actions/workflows/pipeline.yaml)
[![Docs](https://readthedocs.org/projects/text-similarity/badge/?version=latest)](https://text-similarity.readthedocs.io/pt-br/latest/)
[![PyPI](https://img.shields.io/pypi/v/text-similarity-br)](https://pypi.org/project/text-similarity-br/)
[![Python](https://img.shields.io/badge/python-3.8%20|%203.9%20|%203.10%20|%203.11%20|%203.12-blue)](https://www.python.org)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Uma biblioteca Python otimizada e especializada na comparação de similaridade de textos em português brasileiro (PT-BR). Ideal para sistemas de NLP, chatbots, análise de sentimento e cruzamento de dados onde as peculiaridades do idioma, formatação de dinheiro, fonética regional e medidas influenciam a real intenção e semelhança dos textos.

## ✨ Recursos Principais

- **Limpeza Especializada (TextCleaner):** Expansão de contrações modernas ("vc" -> "você", "fds" -> "fim de semana") e tratamento de acentos focado no nosso idioma.
- **Detecção de Entidades (EntityNormalizer):** Extração e preservação inteligente de grandezas antes da "limpeza bruta" que as destruiria. (Ex: converte `R$ 30,00` para a tag única `<money:30.0>`).
  - Dinheiro (`R$ 30,00`, `30 reais`)
  - Datas (`12/03/2023`, `ontem`)
  - Dimensões/Pesos (`2kg`, `10 m`)
  - Modelos de Produto (`S22 Ultra`, `iPhone 13 Pro`)
- **Pré-processamento Avançado:** Tokenização, remoção de _stopwords_ do português, e Lematização (com suporte nativo ao SpaCy `pt_core_news_sm`).
- **Comparações Híbridas:** Algoritmos combinados para ir além das palavras (Bag-of-Words).
  - *Cosseno (TF-IDF)*: Para variação lexical.
  - *BM25 (Okapi BM25)*: Alternativa ao TF-IDF, superior para textos curtos (produtos, modelos). Selecionável via `indexing_strategy="bm25"`.
  - *Índice Denso (sentence-transformers)*: Filtro inicial por similaridade semântica densa, capturando sinônimos sem sobreposição lexical. Selecionável via `indexing_strategy="dense"`.
  - *Distância de Edição (Levenshtein)*: Rápido, usando `rapidfuzz` para detectar erros de digitação.
  - *Fonética (Metaphone PT-BR adaptado)*: Trata "cassaa" e "caça" como pesos idênticos.
  - *Interseção de Entidades*: Lógica de "Curto-Circuito" que garante correspondência (score altíssimo) se a entidade de busca essencial (ex: `GN500`) for validada intacta em textos mais longos.
- **Pipeline Otimizada (Joblib Cache):** Suporte a cache em disco nativo. Textos volumosos já mastigados nas etapas de Regex/SpaCy não gastam processamento de novo.
- **Performance Otimizada para Alto Volume:** Regex pré-compilados, pré-processamento paralelo via `ProcessPoolExecutor`, batch spaCy com `nlp.pipe()`, cache persistente de catálogos em disco e LRU cache para dateparser.

---

## Requisitos

- **Python:** \>= 3.8

---

## 🚀 Instalação

```bash
# Com uv (recomendado)
uv add text-similarity-br

# Com pip
pip install text-similarity-br
```

A partir da versão 0.4.0, o pacote já inclui `sentence-transformers` como dependência, habilitando **Similaridade Semântica** sem instalação adicional.

Com suporte a lematização via SpaCy (opcional):

```bash
# Com uv
uv add "text-similarity-br[nlp]"
uv run python -m spacy download pt_core_news_sm

# Com pip
pip install "text-similarity-br[nlp]"
python -m spacy download pt_core_news_sm
```

---

## 📖 Como Usar

A API pública foi desenhada em torno da fachada `Comparator`, garantindo facilidade sem esconder o poder customizável.

### Modo Básico (Rápido e Simples)
Opera apenas sobre Bag-of-Words e correções de grafia (Levenshtein/Cosseno). Ideal para volume de dados altos e textos curtos.

```python
from text_similarity.api import Comparator

comp = Comparator.basic()

score = comp.compare("iphone 13 pro", "iphone pro 13")
print(f"Similaridade: {score:.2f}") # Output ~0.8 a 1.0 dependendo do peso
```

### Modo "Smart" (Entidades e Fonética)
Ativa nativamente os extratores de Moeda, Data, Dimensões, Modelos de Produto e aplica cálculos fonéticos. Aceita os parâmetros `fusion_strategy` (`"linear"` ou `"rrf"`) e `rrf_k` para controlar a fusão dos rankings em operações batch.

```python
from text_similarity.api import Comparator

comp = Comparator.smart()

novo_score = comp.compare("Foi me cobrado 30 reais", "O preço é R$ 30,00")

print(f"Similaridade Smart: {novo_score:.2f}") 
# Resultado alto por conta da identificação da entidade financeira exata

# --- Interseção Perfeita de Modelos (Short-circuit) ---
score_modelo = comp.compare("GN500", "Temos as peças GN 500, GN 1000 e SK 200")
print(f"Score Modelo Embutido: {score_modelo:.2f}")
# Resultado: ~0.95. Ao localizar o modelo procurado "GN500" isolado no meio do 
# texto longo alvo, o algoritmo de intersecção assegura diretamente uma alta 
# pontuação, ignorando todo o resto da string longa que causaria diluição.
```

#### Filtrando Entidades Específicas

Por padrão, o modo `smart` ativa **todos** os extratores (`money`, `date`, `dimension`, `number`, `product_model`). Você pode restringir apenas às entidades relevantes para o seu domínio passando o parâmetro `entities`:

```python
from text_similarity.api import Comparator

# Apenas modelos de produto — ideal para catálogos de peças técnicas
comp = Comparator.smart(entities=["product_model"])

# Apenas valores monetários — ideal para sistemas financeiros
comp_fin = Comparator.smart(entities=["money", "number"])

# Datas e dimensões — ideal para laudos e fichas técnicas
comp_lab = Comparator.smart(entities=["date", "dimension"])
```

> **Dica:** Filtrar entidades melhora a precisão evitando falsos positivos. Um extrator de `date` ativo num catálogo de produtos pode mapear incorretamente SKUs contendo dígitos de ano.

### Modo Semântico (Word Embeddings)
Para capturar a real intenção semântica entre sinônimos que não compartilham nenhuma letra (ex: `"veículo"` vs `"carro"`), você pode ativar o motor de **Sentence-Transformers**.

```python
from text_similarity.api import Comparator

# Habilita o uso de modelos densos por debaixo dos panos
comp = Comparator.smart(use_embeddings=True)

score = comp.compare("automóvel bicombustível", "carro flex")
print(f"Similaridade Semântica: {score:.2f}") # Alto score, diferentemente do TF-IDF puro.
```
*Atenção: A primeira chamada em cada processo isolado pode demorar alguns milisegundos a mais para carregar o modelo PyTorch na RAM. Nos métodos de Lote (`compare_batch` / `strategy="parallel"`), a Similaridade Semântica age como uma avaliação final super otimizada apenas nos `top_n` retornados pelo TF-IDF.*

### Processamento em Lote (Batch)
Para casos de uso onde é necessário comparar uma *query* contra centenas ou milhares de candidatos, utilize o método `compare_batch`. Ele é altamente otimizado aplicando matrizes esparsas via Scikit-Learn e descartes (short-circuit) matemáticos. Entregando resultados consolidados até **~48x mais rápido** dependendo do volume.

```python
from text_similarity.api import Comparator
comp = Comparator.smart()

busca = "Notebook Dell Inspiron 15"
candidatos = [
    "Dell Inspiron 15 polegadas i5",
    "Notebook Lenovo Thinkpad",
    "Mouse sem fio logitech",
    # ... 10,000 outros itens
]

# Filtra rapidamente por TF-IDF mínimo (0.1) e extrai os 5 melhores
resultados = comp.compare_batch(busca, candidatos, top_n=5, min_cosine=0.1)

for r in resultados:
    print(f"Score: {r['score']:.2f} | Match: {r['candidate']}")
```

### Comparação Multi-Query (`compare_many_to_many`)
Quando você precisa comparar **múltiplas buscas** contra o mesmo catálogo de candidatos, use `compare_many_to_many`. Ele pré-computa a matriz TF-IDF dos candidatos **uma única vez**, eliminando recálculos redundantes e entregando speedups significativos em cenários de alto volume.

```python
from text_similarity.api import Comparator
comp = Comparator.smart()

buscas = [
    "Notebook Dell Inspiron 15",
    "Mouse sem fio logitech",
    "Monitor Samsung 27 polegadas",
]
candidatos = [
    "Dell Inspiron 15 polegadas i5",
    "Notebook Lenovo Thinkpad",
    "Mouse logitech wireless",
    "Monitor Samsung 27'' 4K",
    # ... milhares de itens
]

# Retorna uma lista de resultados para CADA query
todos_resultados = comp.compare_many_to_many(
    buscas, candidatos, top_n=5, min_cosine=0.1
)

for query, resultados in zip(buscas, todos_resultados):
    print(f"\n🔍 Query: {query}")
    for r in resultados:
        print(f"  Score: {r['score']:.2f} | {r['candidate']}")
```

> **Quando usar qual?**
> - `compare_batch()` → 1 query × N candidatos (ex: busca textual de um usuário).
> - `compare_many_to_many()` → M queries × N candidatos (ex: deduplicação em lote, cruzamento de bases).

### Fusão de Rankings via RRF (`fusion_strategy="rrf"`)
Por padrão, o `Comparator` combina os scores dos algoritmos por **soma ponderada** (estratégia `"linear"`). Para cenários onde os scores brutos dos algoritmos possuem escalas muito diferentes (ex: mistura de léxico com semântica), você pode usar **Reciprocal Rank Fusion (RRF)**, que baseia-se na **posição** dos candidatos em cada ranking em vez dos scores brutos:

```python
from text_similarity.api import Comparator

# RRF: combina rankings por posição, eliminando problemas de escala
comp = Comparator.smart(fusion_strategy="rrf")

resultados = comp.compare_batch(
    "Notebook Dell Inspiron",
    candidatos,
    top_n=10,
    min_cosine=0.1,
)

# Cada resultado inclui detalhes do RRF: rank e contribuição de cada algoritmo
for r in resultados:
    print(f"Score: {r['score']:.2f} | {r['candidate']}")
    print(f"  Detalhes: {r['details']}")
```

O parâmetro `rrf_k` (padrão 60) controla a suavização: valores maiores atenuam a diferença entre posições no ranking.

```python
# RRF com suavização mais agressiva
comp = Comparator.smart(fusion_strategy="rrf", rrf_k=100)
```

#### Opções de Pesos e Algoritmos

Ao utilizar o modo `smart`, você pode equilibrar os seguintes algoritmos através do parâmetro `weights` (no construtor) ou `rrf_weights` (nas funções de média/batch):

| Opção | Nome Técnico | O que avalia | Melhor uso |
| :--- | :--- | :--- | :--- |
| **`cosine`** | Cosseno (TF-IDF) | Frequência e raridade das palavras. | Detectar palavras-chave idênticas. |
| **`bm25`** | Okapi BM25 | Relevância com saturação de frequência. | Textos curtos (produtos, SKUs). Ativado via `indexing_strategy="bm25"`. |
| **`edit`** | Levenshtein | Proximidade de caracteres (escrita). | Capturar erros de digitação (typos). |
| **`phonetic`** | Fonética (PT-BR) | Pronúncia das palavras em português. | Capturar trocas de letras com som igual (ex: S/Z/X). |
| **`semantic`** | Semântica | Significado e contexto (Embeddings). | Encontrar sinônimos (ex: "carro" vs "veículo"). |
| **`entity`** | Entidades | Identificadores específicos. | Garantir que códigos e modelos coincidam. |

#### Pesos por Algoritmo (`rrf_weights`)

Por padrão, todos os algoritmos contribuem igualmente no RRF. Use `rrf_weights` para dar mais importância a algoritmos específicos — por exemplo, priorizando similaridade semântica sobre busca léxica:

```python
from text_similarity.api import Comparator

# Prioriza semântica (70%) sobre léxico (30%) no ranking final
comp = Comparator.smart(
    use_embeddings=True,
    fusion_strategy="rrf",
    rrf_weights={"cosine": 0.3, "semantic": 0.7},
)

# Prioriza fonética para domínios com erros de digitação frequentes
comp_fon = Comparator.smart(
    fusion_strategy="rrf",
    rrf_weights={"cosine": 0.3, "edit": 0.2, "phonetic": 0.5},
)
```

A fórmula aplicada é: `score = Σ weight_i * 1/(k + rank_i)`. Algoritmos não listados em `rrf_weights` recebem peso `1.0` por padrão.

> **Quando usar `"rrf"` vs `"linear"`:**
> - `fusion_strategy="linear"` (padrão) → Quando os algoritmos operam em escalas similares e os pesos foram calibrados para o seu domínio.
> - `fusion_strategy="rrf"` → Quando mistura algoritmos com escalas distintas (ex: TF-IDF + Semântico), ou quando candidatos consistentemente bem posicionados em múltiplos rankings devem ser priorizados, independentemente do score absoluto.
> - `rrf_weights` → Quando, além de usar RRF, você quer que determinado algoritmo tenha mais influência na posição final do ranking.

Também disponível via import direto para uso avançado — útil quando você já possui rankings próprios (ex: vindos de Elasticsearch, banco vetorial, ou algoritmos customizados) e quer fundi-los:

```python
from text_similarity import RRFusion

# Cada sublista é o ranking de UM algoritmo, ordenado por score descendente.
# A estrutura é: [{"candidate": str, "score": float}, ...]
rankings_por_algoritmo = [
    # Ranking do algoritmo "cosine"
    [
        {"candidate": "Dell Inspiron 15 i5", "score": 0.92},
        {"candidate": "Notebook Lenovo", "score": 0.45},
        {"candidate": "Mouse Logitech", "score": 0.10},
    ],
    # Ranking do algoritmo "semantic"
    [
        {"candidate": "Dell Inspiron 15 i5", "score": 0.85},
        {"candidate": "Mouse Logitech", "score": 0.30},
        {"candidate": "Notebook Lenovo", "score": 0.20},
    ],
]

# Nomes dos algoritmos, na MESMA ORDEM das sublistas acima
nomes_algoritmos = ["cosine", "semantic"]

# Pesos iguais (padrão)
rrf = RRFusion(k=60)

# Ou com pesos por algoritmo
rrf = RRFusion(k=60, weights={"cosine": 0.4, "semantic": 0.6})

ranking_fundido = rrf.fuse(rankings_por_algoritmo, nomes_algoritmos)

for item in ranking_fundido:
    print(f"Score RRF: {item['score']:.3f} | {item['candidate']}")
    # Cada item inclui detalhes: rank, raw_score, rrf_contribution, weight
```

> **Nota:** No uso padrão via `Comparator.smart(fusion_strategy="rrf")`, esses rankings são montados automaticamente pelo `Comparator`. O import direto do `RRFusion` é para cenários onde você quer fundir rankings de fontes externas.

### Execução Paralela (`strategy="parallel"`)
Para cenários de **alto volume** (50+ queries × 10k+ candidatos), ative a estratégia paralela que distribui as queries entre múltiplos processos via `ProcessPoolExecutor`:

```python
from text_similarity.api import Comparator
comp = Comparator.smart()

# Distribui entre 4 processos (padrão: os.cpu_count())
resultados = comp.compare_many_to_many(
    buscas, candidatos, top_n=5, min_cosine=0.1,
    strategy="parallel", n_workers=4,
)

# Funciona também com compare_batch
resultado = comp.compare_batch(
    "busca única", candidatos, top_n=10,
    strategy="parallel", n_workers=4,
)
```

> **⚠️ Quando NÃO usar `parallel`:** Para poucos queries (< 20) ou poucos candidatos (< 5k), o overhead de criação de processos pode superar o ganho. Use `strategy="vectorized"` (padrão) nesses casos.

### Integração Async (FastAPI, aiohttp)
Para **web servers assíncronos**, use os métodos `_async` que offloadam o trabalho CPU-bound para um `ProcessPoolExecutor`, mantendo o event loop livre:

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

> **Métodos async disponíveis:** `compare_batch_async()` e `compare_many_to_many_async()`. Ambos usam `strategy="parallel"` internamente.


### Re-Ranking de Resultados de Bancos Vetoriais
Quando você já possui resultados de um banco vetorial (Pinecone, Qdrant, Milvus, PGVector, Elasticsearch) e quer **re-ordenar** usando validação linguística PT-BR (edição, fonética, entidades), use o `rerank_vector_results`. Ele funciona como um **Cross-Encoder linguístico brasileiro**, aplicando os algoritmos do `HybridSimilarity` sobre os resultados já filtrados pelo banco.

```python
from text_similarity.api import Comparator

comp = Comparator.smart(entities=["product_model"])

# Resultados vindos do seu banco vetorial (Qdrant, Pinecone, etc.)
vector_results = [
    {"id": "doc1", "text": "Peças industriais variadas", "score": 0.90},
    {"id": "doc2", "text": "Ferramentas GN série completa", "score": 0.80},
    {"id": "doc3", "text": "Motor elétrico trifásico", "score": 0.70},
    {"id": "doc4", "text": "Peças GN500 originais", "score": 0.45},
]

# Re-rankeia usando validação linguística
reranked = comp.rerank_vector_results(
    "GN500",
    vector_results,
    preprocess_query=True,        # pipeline na query do usuário
    preprocess_candidates=True,   # pipeline nos textos (se brutos)
)

for r in reranked:
    print(f"Score: {r['score']:.2f} (vetorial: {r['vector_score']:.2f}) | {r['candidate']}")
# "Peças GN500 originais" sobe da posição #4 para #1 via short-circuit de entidade
```

O resultado inclui:
- `id` — identificador do documento (preservado do input, se presente)
- `candidate` — texto original
- `score` — score final do HybridSimilarity
- `vector_score` — score original do banco vetorial
- `details` — detalhes por algoritmo (cosine, edit, phonetic, entity)

> **Formato de entrada:** Cada candidato deve ter pelo menos `"text"` (str) e `"score"` (float). O campo `"id"` é opcional.

> **Pré-processamento:** Use `preprocess_candidates=False` (padrão) quando os textos do banco já estão normalizados. Use `True` quando os textos são brutos e precisam de limpeza/extração de entidades.

> **Compatível com RRF:** Funciona com `fusion_strategy="rrf"` para combinar rankings por posição:
> ```python
> comp = Comparator.smart(entities=["product_model"], fusion_strategy="rrf")
> reranked = comp.rerank_vector_results("GN500", vector_results)
> ```

### Entendendo "Por que" deram Match (Explain)
Às vezes você precisa debugar a intenção do usuário ou mostrar evidências de que o cruzamento de algoritmos detectou semelhança. Use o `.explain()`:

```python
from text_similarity.api import Comparator
comp = Comparator.smart()

detalhes = comp.explain("televisão samsung 55 polegadas", "tv samsung 55\"")

print(detalhes["score"])
# 0.85
print(detalhes["details"])
# {'cosine': 0.82, 'edit': 0.80, 'phonetic': 0.95} -> Foneticamente altíssimo e detectada dimensão de 55.
```

> **Comportamento com strings vazias:** `explain("", "qualquer texto")` retorna `{"score": 0.0, "details": {}}` sem lançar exceção.

> **Short-circuit no `explain()`:** Quando uma entidade é detectada com interseção total (ex: busca por `<productmodel:GN500>` encontrada no texto alvo), `explain()` retorna `{"score": 0.95, "details": {"entity": {..., "short_circuit": True}}}`, igualmente ao `compare()`.

> **`compare_batch()` com lista vazia:** `comp.compare_batch("qualquer", [])` retorna `[]` imediatamente, sem processamento.

## ⚡ Performance para Alto Volume

A biblioteca foi otimizada para cenários de alto volume (100+ queries x 100k+ candidatos) com múltiplas técnicas que reduzem significativamente o tempo de processamento.

### Cache Persistente de Catálogos (`preprocess_catalog`)

Quando o mesmo catálogo de candidatos é reutilizado entre execuções (ex: rodadas diárias de matching contra uma base de produtos), use `preprocess_catalog()` para salvar os textos pré-processados em disco. Na primeira execução, processa e salva. Nas seguintes, carrega direto — economia de ~80% do tempo total.

```python
from text_similarity.api import Comparator
comp = Comparator.smart()

# Primeira execução: processa + salva em disco
candidatos = ["Dell Inspiron 15", "Mouse Logitech MX", ...]  # 150k itens
p_candidatos = comp.preprocess_catalog(candidatos, cache_path="meu_catalogo.pkl")

# Execuções seguintes: carrega do disco instantaneamente
p_candidatos = comp.preprocess_catalog(candidatos, cache_path="meu_catalogo.pkl")

# Use com compare_many_to_many + preprocess=False nos candidatos já processados
resultados = comp.compare_many_to_many(
    queries, p_candidatos, top_n=10, preprocess=False,
)
```

A invalidação é automática via hash SHA-256: se o catálogo mudar (itens adicionados, removidos ou alterados), o cache é reprocessado automaticamente.

### Pré-processamento Paralelo Automático

Para lotes com mais de 1.000 textos, o `_process_batch()` distribui automaticamente o trabalho entre múltiplos processos via `ProcessPoolExecutor`, sem necessidade de configuração. Compatível com Windows (`spawn`).

### Otimizações Internas

As seguintes otimizações são aplicadas automaticamente e não requerem mudanças no código do usuário:

| Otimização | Impacto | Descrição |
|---|---|---|
| Regex pré-compilados | ~15-25% | Todos os 12 patterns de regex são compilados uma única vez no nível de classe |
| Pré-processamento paralelo | ~40-60% | Lotes grandes (>1k textos) são distribuídos entre múltiplos processos |
| Batch spaCy (`nlp.pipe()`) | ~20-40% | Lematização via spaCy usa batch processing ao invés de chamadas individuais |
| Cache persistente | ~80% (re-exec) | Catálogos processados são salvos em disco e reutilizados entre execuções |
| LRU cache dateparser | ~5-10% | Datas já resolvidas são cacheadas em memória (até 1024 entradas) |
| Fonética otimizada | ~5-10% | Substituições fonéticas via regex compilado + mapa ao invés de `.replace()` sequenciais |

### Indexação BM25 (`indexing_strategy="bm25"`)

Por padrão, o pipeline de filtragem usa TF-IDF + cosseno. Para cenários com **textos curtos** (produtos, modelos, SKUs de 3-15 tokens), o BM25 (Okapi BM25) oferece ranking superior graças à saturação de term frequency e normalização por comprimento de documento.

```python
from text_similarity.api import Comparator

# BM25 como estratégia de indexação
comp = Comparator.smart(
    entities=["product_model"],
    indexing_strategy="bm25",
)

# Uso idêntico — toda a API funciona transparentemente
resultados = comp.compare_batch("samsung galaxy s22", candidatos, top_n=10)

# Multi-query também suportado
todos = comp.compare_many_to_many(buscas, candidatos, top_n=5)
```

Os parâmetros `bm25_k1` (saturação de frequência) e `bm25_b` (normalização por comprimento) podem ser ajustados para o seu domínio. Para produtos curtos (3-8 tokens), `bm25_k1=1.5` e `bm25_b=0.3` reduzem a penalização por comprimento:

```python
# Otimizado para catálogos de produtos curtos
comp = Comparator.smart(
    indexing_strategy="bm25",
    bm25_k1=1.5,
    bm25_b=0.3,
)
```

#### Estimativa de Impacto: TF-IDF vs BM25 vs Dense

| Métrica | TF-IDF | BM25 | Dense |
|---|---|---|---|
| Qualidade de ranking (textos curtos) | Baseline | **+10-20% precision@10** | Variável por domínio |
| Recall semântico (sinônimos) | Baixo | Baixo | **Alto** |
| Tempo de indexação (150k candidatos) | ~2s | ~1-3s (comparável) | Não recomendado* |
| Tempo por query | ~5ms (sparse matmul) | ~15-30ms (loop) | ~5-20ms |
| Memória | ~50MB (sparse matrix) | ~80-100MB (dicts) | ~200-500MB |

*\*Em CPU, o `DenseIndex` leva ~5-10 minutos para indexar 150k candidatos. É adequado apenas para catálogos pequenos/médios (até ~10k itens).*

**Recomendação:** use BM25 para catálogos de produtos/SKUs, TF-IDF para bases de texto longo ou volume extremo de queries, e Dense apenas para catálogos pequenos/médios (até ~10k itens) com alta variação lexical entre query e candidatos.

> **Compatível com todas as features:** as três estratégias funcionam com `strategy="parallel"`, `fusion_strategy="rrf"`, `preprocess=False`, métodos async e `rerank_vector_results`. A troca é transparente — apenas mude o `indexing_strategy`.

### Indexação Densa (`indexing_strategy="dense"`)

Para cenários onde a query e os candidatos são **semanticamente equivalentes mas não compartilham palavras** (ex: `"veículo flex"` vs `"carro bicombustível"`), o índice denso usa embeddings do `sentence-transformers` como filtro inicial, capturando similaridade semântica antes mesmo do `HybridSimilarity` entrar em ação.

```python
from text_similarity.api import Comparator

# Índice denso — resolve o gap de recall de sinônimos
comp = Comparator.smart(
    indexing_strategy="dense",
)

# Candidato será encontrado mesmo sem sobreposição lexical
resultados = comp.compare_batch("veículo flex", candidatos, top_n=10)
```

Por padrão utiliza o modelo `paraphrase-multilingual-MiniLM-L12-v2` (multilingual, inclui PT-BR). Para usar outro modelo:

```python
comp = Comparator.smart(
    indexing_strategy="dense",
    dense_model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
)
```

> **⚠️ Limitação importante:** O `DenseIndex` roda em CPU e leva ~5-10 minutos para indexar 150k documentos. **Use apenas para catálogos pequenos/médios (até ~10k itens).** Para grandes volumes com recall semântico, use `rerank_vector_results` combinado com um banco vetorial externo (Qdrant, Pinecone, etc.).

> **Quando usar `"dense"`:** Catálogos de até ~10k itens com alta variação lexical — sinônimos, linguagem informal, suporte ao cliente.

> **Compatível com todas as features:** Dense funciona com `strategy="parallel"`, `fusion_strategy="rrf"`, `preprocess=False` e métodos async. A troca é transparente — apenas mude o `indexing_strategy`.

### Liberando o modelo da memória (`unload_embeddings_model`)

Após uma sessão de inferência intensa, você pode liberar o modelo semântico da RAM/VRAM:

```python
comp = Comparator.smart(use_embeddings=True)

# ... processamento ...

# Libera o modelo da memória global
comp.unload_embeddings_model()

# O modelo será recarregado automaticamente na próxima comparação semântica
```

---

## 🎯 Interpretação dos Scores

O score retornado varia entre `0.0` (completamente diferentes) e `1.0` (idênticos).

| Faixa | Interpretação |
|---|---|
| `>= 0.85` | Match muito forte — provável duplicata ou variação mínima de descrição |
| `0.60 – 0.84` | Match provável — mesmo item com descrição diferente (ex: código com/sem espaço) |
| `0.35 – 0.59` | Match incerto — requer revisão manual |
| `< 0.35` | Sem relação semântica relevante |

> **Dica:** Para domínios com códigos de produto (materiais, SKUs, peças técnicas), um threshold de `>= 0.60` é um bom ponto de partida. Calibre com pares conhecidos do seu domínio para ajustar precisão × recall.

### Uso Apenas para Tratamento de Texto
Se o seu objetivo não for realizar comparações, mas apenas aproveitar o robusto motor de processamento em português (para limpar bases de dados, treinar modelos, remover acentos, expandir contrações e lematizar), você pode instanciar as etapas da `Pipeline` de forma autônoma e oficial:

```python
from text_similarity.pipeline.pipeline import PreprocessingPipeline
from text_similarity.pipeline.backends import CleanTextStage, TokenizerStage, StopwordsStage

# Monte seu pipeline customizado apenas com o que precisa:
pipeline = PreprocessingPipeline([
    CleanTextStage(),  # Expansão de contrações ("vc" -> "você"), sem acentos, lowercase
    TokenizerStage(),  # Tokenização segura
    StopwordsStage()   # Remoção de conectivos inúteis do PT-BR
])

texto_bruto = "Limpando meeu texto, crz... vc viu a promo???"
texto_tratado, stats = pipeline.process(texto_bruto)

print(texto_tratado)
# Saída esperada (bag of words tratado): "limpar texto crz ver promo"
```

### Bypass do Pré-processamento (`preprocess=False`)
Quando seus textos **já foram limpos externamente** (ex: vindos de um pipeline ETL, banco de dados normalizado ou outro sistema de NLP), você pode desativar o pré-processamento para evitar transformações redundantes e ganhar performance:

```python
from text_similarity.api import Comparator
comp = Comparator.smart()

# Textos já normalizados pelo seu pipeline externo
clean1 = "samsung galaxy s22 ultra 256gb"
clean2 = "samsung galaxy s22 ultra 256gb preto"

# Bypassa limpeza, tokenização, stopwords e lematização
score = comp.compare(clean1, clean2, preprocess=False)
print(f"Score: {score:.2f}")

# Também funciona com explain
detalhes = comp.explain(clean1, clean2, preprocess=False)
```

Funciona em **todos os métodos** de comparação:

```python
# Batch — 1 query × N candidatos já limpos
resultados = comp.compare_batch(
    "galaxy s22", candidatos_limpos,
    top_n=10, min_cosine=0.1, preprocess=False,
)

# Multi-query — M queries × N candidatos já limpos
todos = comp.compare_many_to_many(
    queries_limpas, candidatos_limpos,
    top_n=5, preprocess=False,
)

# Async
resultados = await comp.compare_batch_async(
    "galaxy s22", candidatos_limpos,
    top_n=10, preprocess=False,
)
```

> **Quando usar `preprocess=False`:**
> - Dados vindos de pipelines ETL que já normalizam texto.
> - Re-ranking de resultados já processados por outro sistema (ex: Elasticsearch, banco vetorial).
> - Benchmarks onde você quer isolar o custo dos algoritmos de similaridade sem overhead do pipeline.
>
> **Atenção:** Com `preprocess=False`, o cache in-memory **não é utilizado** (não há hash nem armazenamento), e nenhuma etapa do pipeline é executada — incluindo extração de entidades. Certifique-se de que seus textos estão no formato esperado pelos algoritmos.

---

## 📈 Calibração de Pesos (Grid Search)

Para obter a melhor precisão em domínios específicos, você pode calibrar os pesos do algoritmo `HybridSimilarity` usando o `WeightCalibrator`. Ele permite testar múltiplas combinações de pesos contra um dataset "Gold Standard" (anotado manualmente) e gera um relatório detalhado de performance comparativa entre precisão e custo de tempo (latência).

```python
from text_similarity.api import Comparator
from text_similarity.tuning.calibrator import WeightCalibrator

comp = Comparator.smart()

# Dataset de teste (Gold Standard)
gold_standard = [
    {"query": "casa", "target": "caza", "match": True},
    {"query": "celular", "target": "fone", "match": False},
]

# Configurações de pesos que você deseja comparar
configs = [
    {"cosine": 0.5, "edit": 0.5},
    {"edit": 1.0},
    {"phonetic": 0.8, "cosine": 0.2},
]

calibrator = WeightCalibrator(comp, configs)
report = calibrator.evaluate(gold_standard)

# Exibe o dashboard de resultados (requer extra 'tuning')
report.summary()
```

Para habilitar a visualização rica (rich terminal dashboard):
```bash
# Com uv
uv add "text-similarity-br[tuning]"

# Com pip
pip install "text-similarity-br[tuning]"
```

---

## ⚙️ Configuração do Cache

A biblioteca mantém um cache in-memory (SHA-256) para evitar reprocessar o mesmo texto várias vezes pelo pipeline. Por padrão, o cache está **ativado**.

```python
from text_similarity.api import Comparator

# Cache ativado por padrão (padrão)
comp = Comparator.smart(use_cache=True)

# Desativar o cache (útil em ambientes com memória limitada ou testes)
comp_no_cache = Comparator.smart(use_cache=False)
```

### Cache Persistente em Disco

Para cenários de alto volume com catálogos reutilizáveis, use `preprocess_catalog()` para salvar em disco e eliminar reprocessamento entre execuções. Veja a seção [Cache Persistente de Catálogos](#cache-persistente-de-catálogos-preprocess_catalog) para detalhes.

### Limpando o Cache Manualmente

Use `clear_cache()` quando precisar forçar o reprocessamento — por exemplo, depois de alterar as entidades ativas ou ao liberar memória após um lote grande:

```python
comp = Comparator.smart()

# Processa e armazena em cache
comp.compare("produto A", "produto B")

# Libera toda a memória do cache in-memory e limpa o cache em disco (Joblib)
comp.clear_cache()
```

---

## 🔌 Extensibilidade — Registrando Entidades Customizadas

A biblioteca expõe o `ExtractorRegistry` para registrar extratores de entidade personalizados, sem precisar alterar o código-fonte:

```python
from text_similarity.entities.base import EntityExtractor, EntityMatch
from text_similarity.entities.registry import ExtractorRegistry

class CPFExtractor(EntityExtractor):
    """Exemplo: extrator de CPF para sistemas de RH."""

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

# Registra o extrator customizado
ExtractorRegistry.register("cpf", CPFExtractor)

# Instancia o Comparator ativando apenas o seu extrator
comp = Comparator.smart(entities=["cpf"])
score = comp.compare("019.283.847-09", "documento cpf 01928384709")
```

Extratores disponíveis por padrão:

| Nome | Exemplos detectados |
|---|---|
| `money` | `R$ 30,00`, `50 reais`, `USD 100` |
| `date` | `12/03/2023`, `ontem`, `amanhã`, `25 de abril` |
| `dimension` | `2kg`, `1.5l`, `30cm`, `10m²` |
| `number` | `3`, `três`, `1000` |
| `product_model` | `S22 Ultra`, `iPhone 13`, `XJ-900` |


## 🤝 Contribuindo

Padrões de qualidade seguidos rigorosamente: `Ruff` (lint + format) e `MyPy` (tipagem forte).

### Fluxo de Trabalho

- **Branch de desenvolvimento:** `dev` — todo desenvolvimento acontece aqui
- Crie branches de feature a partir de `dev` e abra PRs de volta para `dev`
- Merges para `main` são feitos apenas em releases

### Antes de Abrir um PR

```bash
# Lint e formatação
uv run ruff check src tests
uv run ruff format src tests

# Tipagem
uv run mypy src

# Testes
uv run pytest tests/
```

### Reportando Bugs / Sugestões

Abra uma [issue no GitHub](https://github.com/joscelino/text_similarity/issues) descrevendo:
- Versão da biblioteca (`pip show text-similarity-br`)
- Versão do Python
- Exemplo mínimo reproduzível
