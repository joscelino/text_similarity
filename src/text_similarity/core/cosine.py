"""Módulo de similaridade baseada em Distância de Cosseno usando Bag of Words."""

from __future__ import annotations

# scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

from .base import SimilarityAlgorithm


class CosineSimilarity(SimilarityAlgorithm):
    """Calcula similaridade cosseno utilizando TF-IDF.
    
    Bom para avaliar sobreposição de vocabulário e contexto global.
    """

    def __init__(self, ngram_range: tuple[int, int] = (1, 2)) -> None:
        """Inicializa configurações de tokenização TF-IDF."""
        # Usa bigramas por padrão para capturar contexto local ("s22 ultra")
        self.ngram_range = ngram_range

    def compare(self, text1: str, text2: str) -> float:
        """Extrai os tokens TF-IDF e os avalia por distância Vetorial."""
        if not text1 or not text2:
            return 0.0

        # Utilizamos o TfidfVectorizer para computar a similaridade dos dois textos.
        # Em cenários de busca massiva um "VectorSpaceModel" pré-treinado seria
        # injetado, mas como algoritmo atômico fazemos o fit nele mesmo.
        vectorizer = TfidfVectorizer(ngram_range=self.ngram_range, min_df=1)

        try:
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            # tfidf_matrix[0:1] calcula contra matriz inteira,
            # [0][1] extrai o cruzamento.
            score = sklearn_cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)[0][1]
            # cast garante o tipo pois sklearn retorna numpy.float64
            return float(score)
        except ValueError:
            # Caso "After pruning, no terms remain." ou outras exceções de vocabulário
            return 0.0
