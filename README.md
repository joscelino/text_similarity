# Text Similarity PT-BR

[![CI Pipeline](https://github.com/joscelino/text_similarity/actions/workflows/pipeline.yaml/badge.svg)](https://github.com/joscelino/text_similarity/actions/workflows/pipeline.yaml)
[![Docs](https://readthedocs.org/projects/text-similarity/badge/?version=latest)](https://text-similarity.readthedocs.io/pt-br/latest/)
[![PyPI](https://img.shields.io/pypi/v/text-similarity-br)](https://pypi.org/project/text-similarity-br/)
[![Python](https://img.shields.io/badge/python-3.8%20|%203.9%20|%203.10%20|%203.11%20|%203.12-blue)](https://www.python.org)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Uma biblioteca Python otimizada e especializada na comparação de similaridade de textos em português brasileiro (PT-BR). Ideal para sistemas de NLP, chatbots, análise de sentimento e cruzamento de dados onde as peculiaridades do idioma, formatação de dinheiro, fonética regional e medidas influenciam a real intenção e semelhança dos textos.

## Recursos Principais

- **Limpeza Especializada (TextCleaner):** Expansão de contrações modernas ("vc" -> "você", "fds" -> "fim de semana") e tratamento de acentos focado no nosso idioma.
- **Detecção de Entidades (EntityNormalizer):** Extração e preservação inteligente de grandezas antes da "limpeza bruta" que as destruiria. (Ex: converte `R$ 30,00` para a tag única `<money:30.0>`).
  - Dinheiro (`R$ 30,00`, `30 reais`)
  - Datas (`12/03/2023`, `ontem`)
  - Dimensões/Pesos (`2kg`, `10 m`)
  - Modelos de Produto (`S22 Ultra`, `iPhone 13 Pro`)
- **Pré-processamento Avançado:** Tokenização, remoção de _stopwords_ do português, e Lematização (com suporte nativo ao SpaCy `pt_core_news_sm`).
- **Comparações Híbridas:** Algoritmos combinados para ir além das palavras (Bag-of-Words).
  - *Cosseno (TF-IDF)*: Para variação lexical.
  - *Distância de Edição (Levenshtein)*: Rápido, usando `rapidfuzz` para detectar erros de digitação.
  - *Fonética (Metaphone PT-BR adaptado)*: Trata "cassaa" e "caça" como pesos idênticos.
  - *Interseção de Entidades*: Lógica de "Curto-Circuito" que garante correspondência (score altíssimo) se a entidade de busca essencial (ex: `GN500`) for validada intacta em textos mais longos.
- **Pipeline Otimizada (Joblib Cache):** Suporte a cache em disco nativo. Textos volumosos já mastigados nas etapas de Regex/SpaCy não gastam processamento de novo.

---

## Requisitos

- **Python:** \>= 3.8

---

## 🚀 Instalação

```bash
pip install text-similarity-br
```

Com suporte a lematização via SpaCy (opcional):

```bash
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
print(f"Similaridade: {score:.2f}") # Output ~0.8 a 1.0 depending on weight
```

### Modo "Smart" (Entidades e Fonética)
Ativa nativamente os extratores de Moeda, Data, Dimensões, Modelos de Produto e aplica cálculos fonéticos.

```python
from text_similarity.api import Comparator

comp = Comparator.smart()

novo_score = comp.compare("Foi me cobrado 30 reais", "O preço é R$ 30,00")

print(f"Similaridade Smart: {novo_score:.2f}") 
# Resultado alto por conta da identificação da entidade financeira exata

# --- Novo Recurso: Interseção Perfeita de Modelos (Short-circuit) ---
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


## Contribuindo
Padrões de Qualidade seguidos rigorosamente: `Ruff` (Lint+Format) e `MyPy` (Tipoção Forte).
Para garantir suas alterações, digite:
```bash
uv run ruff check src tests
uv run pytest tests/
```
