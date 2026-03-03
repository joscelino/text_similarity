# Text Similarity PT-BR

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
- Instalação via [uv](https://github.com/astral-sh/uv) (Recomendado) ou `pip`.

---

## 🚀 Instalação

```bash
# Clone ou baixe o projeto
uv sync

# Ative o ambiente virtual
# Windows:
.venv\Scripts\activate
# Linux/MacOS:
source .venv/bin/activate
```

*(Nota: Adicione o modelo SpaCy opcional e de alta precisão com o comando `python -m spacy download pt_core_news_sm`)*

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

## 🎯 Interpretação dos Scores

O score retornado varia entre `0.0` (completamente diferentes) e `1.0` (idênticos).

| Faixa | Interpretação |
|---|---|
| `>= 0.85` | Match muito forte — provável duplicata ou variação mínima de descrição |
| `0.60 – 0.84` | Match provável — mesmo item com descrição diferente (ex: código com/sem espaço) |
| `0.35 – 0.59` | Match incerto — requer revisão manual |
| `< 0.35` | Sem relação semântica relevante |

> **Dica:** Para domínios com códigos de produto (materiais, SKUs, peças técnicas), um threshold de `>= 0.60` é um bom ponto de partida. Calibre com pares conhecidos do seu domínio para ajustar precisão × recall.

---

## Configuração do Cache Local

A biblioteca expõe opções de cache na largada de configuração (Memory de Disco usando hashlib/Joblib).  Ao passar `use_cache=True` para os construtores de classe, hashes em SHA-256 previnem as longas travadas do motor de Regex ou SpaCy de recalcular um payload que o serviço de NLP submeteu recentemente.


## Contribuindo
Padrões de Qualidade seguidos rigorosamente: `Ruff` (Lint+Format) e `MyPy` (Tipoção Forte).
Para garantir suas alterações, digite:
```bash
uv run ruff check src tests
uv run pytest tests/
```
