"""text-similarity-ptbr.

Biblioteca Python para comparação de similaridade de textos otimizada para PT-BR.
"""

from .api import Comparator
from .exceptions import (
    PipelineError,
    StageConfigError,
    StageProcessingError,
    TextSimilarityError,
)

__version__ = "0.2.0"
__all__ = [
    "Comparator",
    "TextSimilarityError",
    "PipelineError",
    "StageProcessingError",
    "StageConfigError",
]
