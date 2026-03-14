"""text-similarity-ptbr.

Biblioteca Python para comparação de similaridade de textos otimizada para PT-BR.
"""

from .api import Comparator
from .core.bm25 import BM25Index
from .core.dense import DenseIndex
from .core.fusion import RRFusion
from .exceptions import (
    PipelineError,
    StageConfigError,
    StageProcessingError,
    TextSimilarityError,
)

__version__ = "0.6.0"
__all__ = [
    "BM25Index",
    "Comparator",
    "DenseIndex",
    "RRFusion",
    "TextSimilarityError",
    "PipelineError",
    "StageProcessingError",
    "StageConfigError",
]
