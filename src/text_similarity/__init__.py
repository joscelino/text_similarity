"""text-similarity-ptbr.

Biblioteca Python para comparação de similaridade de textos otimizada para PT-BR.
"""

from .api import Comparator
from .core.bm25 import BM25Index
from .core.fusion import RRFusion
from .exceptions import (
    PipelineError,
    StageConfigError,
    StageProcessingError,
    TextSimilarityError,
)

__version__ = "0.4.0"
__all__ = [
    "BM25Index",
    "Comparator",
    "RRFusion",
    "TextSimilarityError",
    "PipelineError",
    "StageProcessingError",
    "StageConfigError",
]
