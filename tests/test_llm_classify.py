import os
from LLMTextualAnswer import llm_classify
from langchain_ollama import ChatOllama


def test_llm_classify_index_label():
    llm = ChatOllama(model=os.getenv("OLLAMA_MODEL", "gemma3:1b"))
    result = llm_classify(
        "Some short distance run.",
        ["News", "Sports", "Tech"],
        llm=llm,
    )
    assert result == "Sports"


def test_llm_classify_substring_match():
    llm = ChatOllama(model=os.getenv("OLLAMA_MODEL", "gemma3:1b"))
    result = llm_classify(
        "Some short twit before elections.",
        ["Sports", "Politics"],
        llm=llm,
    )
    assert result == "Politics"


def test_llm_classify_batch():
    llm = ChatOllama(model=os.getenv("OLLAMA_MODEL", "gemma3:1b"))
    result = llm_classify(
        ["Beta essay.", "Alpha text.", "Gamma poen."],
        ["Alpha", "Beta", "Gamma"],
        llm=llm,
    )
    assert result == ["Beta", "Alpha", "Gamma"]
