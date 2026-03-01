from text_similarity.preprocessing.lemmatization import Lemmatizer
from text_similarity.preprocessing.stopwords import StopwordsFilter
from text_similarity.preprocessing.text_cleaner import TextCleaner
from text_similarity.preprocessing.tokenizer import Tokenizer


def test_text_cleaner() -> None:
    cleaner = TextCleaner()
    # Expansão de contrações, remoção de acentos e pontuação, lowercase
    text = "Ele não gostou do João, que pagou R$ 50,00 na loja."
    # do -> de o
    # na -> em a
    # não -> nao, joão -> joao
    # vírgula e ponto são removidos. O R$ perde $. Mas vamos checar o exato.
    cleaned = cleaner.clean(text)

    assert "joao" in cleaned
    assert "nao" in cleaned
    assert "de o" in cleaned
    assert "em a" in cleaned
    assert "," not in cleaned
    assert (
        "r$ 5000" in cleaned
    )  # Porque R$ 50,00 -> r$ 5000 (sem acento, virgula vai embora)
    # Vamos ver o regex atual: [^a-z0-9\s$<>\-:] . Logo o $ fica. A vírgula sai, o ponto sai.
    assert "r$ 5000" in cleaned


def test_text_cleaner_keeps_entities() -> None:
    cleaner = TextCleaner()
    text = "Custa <money:30.5> na feira!"
    cleaned = cleaner.clean(text)
    # na -> em a
    # ! removida
    assert cleaned == "custa <money:30.5> em a feira"


def test_tokenizer() -> None:
    tokenizer = Tokenizer()
    text = "custa <money:30.5> em a feira s22-ultra"
    tokens = tokenizer.tokenize(text)

    assert len(tokens) == 6
    assert tokens[0] == "custa"
    assert tokens[1] == "<money:30.5>"  # Mantida inteira!
    assert tokens[2] == "em"
    assert tokens[3] == "a"
    assert tokens[4] == "feira"
    assert tokens[5] == "s22-ultra"  # Hífens no meio da palavra são mantidos pelo regex


def test_stopwords_filter() -> None:
    filter_sw = StopwordsFilter()
    tokens = ["ele", "nao", "gostou", "de", "o", "joao", "<money:10>"]
    filtered = filter_sw.filter(tokens)

    assert "ele" not in filtered
    assert "nao" not in filtered
    assert "de" not in filtered
    assert "o" not in filtered
    assert "gostou" in filtered
    assert "joao" in filtered
    assert "<money:10>" in filtered  # Preservado


def test_lemmatization_fallback() -> None:
    lemmatizer = Lemmatizer()
    tokens = ["gostou", "meninos", "<date:2024>"]

    lemmas = lemmatizer.lemmatize(tokens)
    assert len(lemmas) == 3
    assert "<date:2024>" in lemmas

    # Dependendo do que tem instalado na máquina de teste (provavelmente 'none')
    if lemmatizer.backend == "none":
        assert lemmas[0] == "gostou"
