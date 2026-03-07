"""Exceções customizadas para a biblioteca text_similarity."""

from __future__ import annotations


class TextSimilarityError(Exception):
    """Exceção base para todos os erros do domínio text_similarity."""
    pass


class PipelineError(TextSimilarityError):
    """Exceção base para falhas ocorridas durante a execução do Pipeline."""
    pass


class StageProcessingError(PipelineError):
    """Lançada quando um estágio individual falha ao processar dados.
    
    Attributes:
        stage_name: Nome da classe do estágio que falhou.
        original_error: A exceção original capturada que causou a falha.
    """
    
    def __init__(self, stage_name: str, original_error: Exception, message: str | None = None) -> None:
        self.stage_name = stage_name
        self.original_error = original_error
        
        msg = message or f"Falha no estágio {stage_name}: {original_error.__class__.__name__} - {str(original_error)}"
        super().__init__(msg)


class StageConfigError(PipelineError):
    """Lançada quando um estágio apresenta erro de inicialização ou configuração.
    
    (ex: Modelos NLP ausentes, arquivos de dados corrompidos).
    """
    pass
