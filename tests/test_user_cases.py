from text_similarity.api import Comparator


def test_example_1():
    comp = Comparator.smart()
    t1 = "GN500"
    t2 = (
        "[GN 500, GN 1000, GN 1001, GN 1500, GN 2000, GN 2500, GN 2501, GN 3000, "
        "GN 3500, SK 200, SK 201, SK 400, SK 401, SK 700, SK 701, MC 4000, "
        "MC 4001, MC 6000, MC 6001]"
    )
    score = comp.compare(t1, t2)
    print(f"Example 1 Score: {score}")
    assert score > 0.65


def test_example_2():
    comp = Comparator.smart()
    t1 = "104401"
    t2 = "lista de reagentes DiaClon que não contém nenhum código semelhante"
    score = comp.compare(t1, t2)
    print(f"Example 2 Score: {score}")
    assert score < 0.4


def test_example_3():
    comp = Comparator.smart()
    t1 = "PPW38002"
    t2 = (
        "[PPW38000tPROFEMUR61650, HASTE 135 mmtReta Cone tamanho "
        "10rnPPW38001tPROFEMUR61650, HASTE 135 mmtReta Cone tamanho "
        "11rnPPW38002tPROFEMUR61650, HASTE 135 mmtReta Cone tamanho 12]"
    )
    score = comp.compare(t1, t2)
    print(f"Example 3 Score: {score}")
    assert score > 0.65
