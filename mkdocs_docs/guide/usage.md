# Como Usar

A forma primária de interagir com a lib é através da classe `Comparator`.

## Modo "Basic" (Volume / Velocidade)
Para bases de dados intensas, use apenas o corretor de distâncias (Levenshtein e TF-IDF Cosseno), pulando a pesada extração de Entidades em PT-BR.

```python
from text_similarity.api import Comparator

comp = Comparator.basic()
print(comp.compare("iphone 13 pro", "iphone pro 13"))
```

## Modo "Smart" (Linguagem Natural)
O modo Smart ativa todos os filtros de pipeline, lematiza as palavras e invoca nossa correção fonética antes do _scoring_ final.

```python
from text_similarity.api import Comparator

comp = Comparator.smart()
print(comp.compare("Gastei uns 50 reais", "Custou R$ 50,00"))
```

## Explorando os Detalhes (.explain)
Pede para a biblioteca justificar por que dois textos combinaram (útil em testes unitários).

```python
detalhes = comp.explain("televisão samsung 55 polegadas", "tv samsung 55\"")
print(detalhes)
# {'score': 0.85, 'details': {'cosine': ..., 'edit': ..., 'phonetic': ...}}
```
