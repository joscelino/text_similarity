# Instalação

## Requisitos

- Python >= 3.8
- (Recomendado) `uv` para gestão ultrarrápida de pacotes.

## Como instalar

[Placeholder: Instruções do PyPI entrarão aqui na Fase 6.5]

```bash
# Para clonar do Github e instalar via UV:
git clone https://github.com/joscelino/text_similarity.git
cd text_similarity
uv sync
```

## Dependências Opcionais (NLP Pesado)
Se você for utilizar lematização avançada e normalização gramatical perfeita, instale o pacote `spaCy` do português:

```bash
python -m spacy download pt_core_news_sm
```
