import os
import re

import pytest

# Lista de termos que NÃO devem aparecer em mensagens de log ou print (erros comuns)
FORBIDDEN_TERMS = [
    r"encotrado",  # Erro corrigido: encontrado
    r"processameto",  # Erro comum: processamento
    r"atribuido",  # Falta acento em contextos PT-BR (atribuído)
    r"invalido",  # Falta acento: inválido
    r"parametros",  # Falta acento: parâmetros
    r"configuracao",  # Falta cedilha/acento: configuração
]


def get_all_python_files(root_dir):
    python_files = []
    for root, _, files in os.walk(root_dir):
        if "tests" in root or ".venv" in root or "__pycache__" in root:
            continue
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
    return python_files


@pytest.mark.parametrize("file_path", get_all_python_files("src"))
def test_no_typos_in_logs_and_prints(file_path):
    """Verifica erros de digitação em chamadas de logger ou print.

    Garante que arquivos Python no diretório src não contêm termos
    com grafia incorreta em mensagens de log.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Regex para capturar conteúdo dentro de logger.something(...) ou print(...)
    log_patterns = [
        r"logger\.(?:info|warning|error|debug|critical|exception)\s*\(\s*(['\"].*?['\"])\s*",
        r"print\s*\(\s*(['\"].*?['\"])\s*",
    ]

    for pattern in log_patterns:
        matches = re.finditer(pattern, content, re.DOTALL)
        for match in matches:
            log_message = match.group(1).lower()
            for forbidden in FORBIDDEN_TERMS:
                if re.search(forbidden, log_message):
                    line_no = content.count("\n", 0, match.start()) + 1
                    pytest.fail(
                        f"Erro de digitação '{forbidden}' em"
                        f" {file_path}:{line_no}\n"
                        f"Mensagem: {log_message}"
                    )
