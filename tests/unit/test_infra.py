from text_similarity import Comparator, __version__


def test_version() -> None:
    assert isinstance(__version__, str)


def test_comparator_interface() -> None:
    # Testando se os métodos da interface estão disponíveis
    cmp_basic = Comparator.basic()
    assert cmp_basic.mode == "basic"

    cmp_smart = Comparator.smart(entities=["date"])
    assert cmp_smart.mode == "smart"

    assert hasattr(cmp_basic, "compare")
    assert hasattr(cmp_smart, "explain")
