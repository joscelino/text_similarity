"""Testes da assinatura pública de ``Comparator.__init__``.

Garante:
    - ``use_embeddings`` é parâmetro tipado explícito.
    - **kwargs NÃO existe mais na assinatura pública.
    - Parâmetros desconhecidos levantam ``TypeError`` com mensagem clara.
"""

from __future__ import annotations

import inspect

import pytest

from text_similarity.api import Comparator


class TestNoKwargsInSignature:
    """A assinatura pública não pode aceitar ``**kwargs`` genéricos."""

    def test_init_has_no_var_keyword(self) -> None:
        sig = inspect.signature(Comparator.__init__)
        var_keyword = [
            p
            for p in sig.parameters.values()
            if p.kind is inspect.Parameter.VAR_KEYWORD
        ]
        assert not var_keyword, (
            "Comparator.__init__ não deve mais aceitar **kwargs; "
            f"encontrado: {var_keyword}"
        )

    def test_use_embeddings_is_typed_parameter(self) -> None:
        sig = inspect.signature(Comparator.__init__)
        assert "use_embeddings" in sig.parameters
        param = sig.parameters["use_embeddings"]
        assert param.default is False
        # A anotação existe (não é Parameter.empty).
        assert param.annotation is not inspect.Parameter.empty


class TestUseEmbeddingsAsNamedParameter:
    """``use_embeddings`` funciona como argumento nomeado."""

    def test_basic_with_use_embeddings_true(self) -> None:
        comp = Comparator(mode="basic", use_embeddings=True)
        assert comp.use_embeddings is True
        assert "semantic" in comp.algorithm.weights

    def test_smart_with_use_embeddings_true(self) -> None:
        comp = Comparator(mode="smart", use_embeddings=True)
        assert comp.use_embeddings is True
        assert "semantic" in comp.algorithm.weights

    def test_default_use_embeddings_is_false(self) -> None:
        comp = Comparator(mode="basic")
        assert comp.use_embeddings is False
        assert "semantic" not in comp.algorithm.weights


class TestUnknownParameterRaisesTypeError:
    """Passar um parâmetro desconhecido deve levantar ``TypeError``."""

    def test_unknown_param_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            Comparator(mode="basic", parametro_inexistente=True)  # type: ignore[call-arg]

    def test_unknown_param_error_message_mentions_argument(self) -> None:
        with pytest.raises(TypeError) as excinfo:
            Comparator(mode="basic", nao_existe=42)  # type: ignore[call-arg]
        # Mensagem padrão do Python inclui o nome do argumento.
        assert "nao_existe" in str(excinfo.value)


class TestNoKwargsGetInSource:
    """Garantia estática: código não usa mais kwargs.get('use_embeddings')."""

    def test_api_source_has_no_kwargs_get(self) -> None:
        import text_similarity.api as api_module

        source = inspect.getsource(api_module)
        assert "kwargs.get" not in source, (
            "kwargs.get(...) foi encontrado em api.py — deveria ser substituído "
            "por self.use_embeddings."
        )

    def test_api_source_no_hardcoded_weight_literals_in_init(self) -> None:
        """No corpo executável de __init__ não pode haver literais de pesos.

        A docstring do método pode referenciar chaves como ``"cosine"`` em
        exemplos — esses casos são ignorados. Verificamos apenas o código
        executável.
        """
        src = inspect.getsource(Comparator.__init__)
        # Remove a docstring do método para não confundir com exemplos
        # dentro dela.
        doc = Comparator.__init__.__doc__ or ""
        src_no_doc = src.replace(doc, "")

        for key in ('"cosine"', '"edit"', '"phonetic"', '"entity"', '"semantic"'):
            assert key not in src_no_doc, (
                f"Literal {key} encontrado em Comparator.__init__; use "
                "as constantes de text_similarity.config.default_weights."
            )
