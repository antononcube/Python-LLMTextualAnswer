import os

import pytest

from LLMTextualAnswer import LLMTextualAnswer


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_OLLAMA_TESTS") != "1",
    reason="Set RUN_OLLAMA_TESTS=1 to run Ollama integration tests.",
)


def test_llm_textual_answer_with_ollama():
    from langchain_ollama import ChatOllama

    llm = ChatOllama(model=os.getenv("OLLAMA_MODEL", "gemma3:1b"))
    text = (
        "Born and raised in the Austrian Empire, Tesla studied engineering and physics "
        "in the 1870s without receiving a degree."
    )
    result = LLMTextualAnswer(text, ["Where born?"], llm=llm, form=dict)
    assert result.get("Where born?")
