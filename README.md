# LLMTextualAnswer

Python package for finding textual answers via LLMs. This is a Python port of the Wolfram Language `LLMTextualAnswer` function, focused on building prompts, wiring LangChain models, and parsing structured outputs.

## Install

```bash
pip install LLMTextualAnswer
```

## Usage

```python
from LLMTextualAnswer import LLMTextualAnswer
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

text = (
    "Born and raised in the Austrian Empire, Tesla studied engineering and physics "
    "in the 1870s without receiving a degree."
)

questions = ["Where born?"]

result = LLMTextualAnswer(
    text,
    questions,
    llm=llm,
    form=dict,
)

print(result)
```

## Notes

- `LLMTextualAnswer` accepts LangChain chat/text models that support `.invoke`.
- Use `prompt_style="chat"` or `prompt_style="text"` if auto-detection is not desired.
- When you want only the prompt template, pass `form="StringTemplate"`.
