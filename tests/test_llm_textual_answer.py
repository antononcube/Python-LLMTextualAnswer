from LLMTextualAnswer import LLMTextualAnswer


class DummyLLM:
    is_chat_model = True

    def invoke(self, messages, config=None):
        return '{"Where born?": "Austrian Empire"}'


class DummyTextLLM:
    def invoke(self, prompt, config=None):
        return '{"Where born?": "Austrian Empire"}'


def test_llm_textual_answer_with_chat_model():
    llm = DummyLLM()
    text = (
        "Born and raised in the Austrian Empire, Tesla studied engineering and physics "
        "in the 1870s without receiving a degree."
    )
    result = LLMTextualAnswer(text, ["Where born?"], llm=llm, form=dict)
    assert result == {"Where born?": "Austrian Empire"}


def test_llm_textual_answer_with_text_model():
    llm = DummyTextLLM()
    text = (
        "Born and raised in the Austrian Empire, Tesla studied engineering and physics "
        "in the 1870s without receiving a degree."
    )
    result = LLMTextualAnswer(
        text,
        ["Where born?"],
        llm=llm,
        form=dict,
        prompt_style="text",
    )
    assert result == {"Where born?": "Austrian Empire"}
