from text_similarity.api import Comparator

def test_cpf_matching():
    comp = Comparator.smart()
    
    cpf_busca = "19992185821"
    texto_longo = "O cliente portador do documento 19992185821 realizou a compra ontem."
    
    score = comp.compare(cpf_busca, texto_longo)
    print(f"\nScore CPF Test: {score}")
    assert score > 0.65, "Score muito baixo para correspondência de CPF"

def test_cnpj_matching():
    comp = Comparator.smart()
    
    cnpj_busca = "49716046000109"
    texto_longo_misturado = "[EMP1 12345678000199, EMP2 49716046000109, EMP3 98765432000111]"
    
    score = comp.compare(cnpj_busca, texto_longo_misturado)
    print(f"\nScore CNPJ Test: {score}")
    assert score > 0.65, "Score muito baixo para correspondência de CNPJ"

if __name__ == '__main__':
    test_cpf_matching()
    test_cnpj_matching()
