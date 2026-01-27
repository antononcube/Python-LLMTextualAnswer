"""Python port of the Wolfram Language LLMTextualAnswer function."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

try:
    from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
except ImportError:  # pragma: no cover - optional dependency
    JsonOutputParser = None
    StrOutputParser = None
    ChatPromptTemplate = None
    PromptTemplate = None


class _AutomaticType:
    def __repr__(self) -> str:  # pragma: no cover - trivial
        return "Automatic"


Automatic = _AutomaticType()

DEFAULT_PROMPT = (
    "You examine texts and can answers questions about them. "
    "The answers you give are amenable for further computer programming processing. "
    'Answer the questions concisely. DO NOT use the word "and" as a list separator. '
    "Separate list elements only with commas. DO NOT number the list or the items of the list. "
    "When possible give numerical results. If an answer is date give it in the ISO-8601 format. "
    'If a question is not applicable give "N/A" as its answer. '
    "Your responses should be in the form of question-answer pairs. "
    "Put the question-answer pairs in a JSON object format. "
    " In the result JSON object the questions are the keys, the answers are the values. "
)

DEFAULT_PRELUDE = "Given the text:"


class LLMTextualAnswerError(ValueError):
    pass


@dataclass
class LLMFunction:
    template: str
    prompt: str
    response_format: str
    llm: Any
    llm_options: Dict[str, Any]
    prompt_style: str = "auto"

    def __call__(self, text: str) -> Any:
        if StrOutputParser is None or JsonOutputParser is None:
            raise LLMTextualAnswerError(
                "langchain-core is required to use LLMFunction."
            )

        query = self.template.format(text=text)

        if callable(self.llm) and not hasattr(self.llm, "invoke"):
            return self.llm(
                query=query,
                prompt=self.prompt,
                response_format=self.response_format,
                **self.llm_options,
            )

        parser = (
            JsonOutputParser()
            if self.response_format == "json"
            else StrOutputParser()
        )

        use_chat = _select_prompt_style(self.llm, self.prompt_style)
        if use_chat:
            if ChatPromptTemplate is None:
                raise LLMTextualAnswerError(
                    "langchain-core prompts are required for chat-style prompts."
                )
            prompt_template = ChatPromptTemplate.from_messages(
                [("system", "{prompt}"), ("human", "{query}")]
            )
        else:
            if PromptTemplate is None:
                raise LLMTextualAnswerError(
                    "langchain-core prompts are required for text-style prompts."
                )
            prompt_template = PromptTemplate.from_template("{prompt}\n{query}")

        chain = prompt_template | self.llm | parser
        config = self.llm_options or None
        return chain.invoke({"prompt": self.prompt, "query": query}, config=config)


FormType = Union[str, type, _AutomaticType]


def _normalize_questions(questions: Union[str, Sequence[str]]) -> List[str]:
    if isinstance(questions, str):
        return [questions]
    if not isinstance(questions, Sequence) or isinstance(questions, (bytes, bytearray)):
        raise LLMTextualAnswerError(
            "Questions must be a string or a sequence of strings."
        )
    if not questions:
        raise LLMTextualAnswerError("Questions cannot be empty.")
    for item in questions:
        if not isinstance(item, str):
            raise LLMTextualAnswerError("All questions must be strings.")
    return list(questions)


def _normalize_form(form: FormType) -> str:
    if form is Automatic or form is None:
        return "auto"
    if isinstance(form, type):
        mapping = {str: "string", dict: "dict", list: "list"}
        if form in mapping:
            return mapping[form]
    if isinstance(form, str):
        mapping = {
            "String": "string",
            "string": "string",
            "str": "string",
            "JSON": "dict",
            "json": "dict",
            "Association": "dict",
            "dictionary": "dict",
            "dict": "dict",
            "List": "list",
            "list": "list",
            "Automatic": "auto",
            "Whatever": "auto",
            "auto": "auto",
            "Function": "function",
            "function": "function",
            "LLMFunction": "llm_function",
            "llm_function": "llm_function",
            "StringTemplate": "template",
            "stringtemplate": "template",
            "template": "template",
        }
        if form in mapping:
            return mapping[form]
    raise LLMTextualAnswerError(
        "Form must be one of Association, List, String, StringTemplate, "
        "LLMFunction, Function, or Automatic."
    )


def _remove_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.lower().startswith("```json") and stripped.endswith("```"):
        return stripped[7:-3].strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        return stripped[3:-3].strip()
    return stripped


def _default_request(questions: Sequence[str]) -> str:
    plural = "s" if len(questions) > 1 else ""
    if len(questions) == 1:
        prefix = "give the shortest answer"
    else:
        prefix = "list the shortest answers"
    return f"{prefix} of the question{plural}:" + "".join(questions)


def _build_template(prelude: str, request_text: str) -> str:
    return f"{prelude}{{text}}{request_text}"


def _select_prompt_style(llm: Any, prompt_style: str) -> bool:
    if prompt_style == "chat":
        return True
    if prompt_style == "text":
        return False
    if getattr(llm, "is_chat_model", False):
        return True
    name = llm.__class__.__name__.lower()
    if "chat" in name:
        return True
    return False


def LLMTextualAnswer(
    text: str,
    questions: Union[str, Sequence[str]],
    form: FormType = Automatic,
    *,
    prelude: Optional[Union[str, _AutomaticType]] = Automatic,
    prompt: Optional[Union[str, _AutomaticType]] = Automatic,
    request: Optional[Union[str, _AutomaticType]] = Automatic,
    llm: Optional[Any] = None,
    llm_call: Optional[Callable[..., Any]] = None,
    llm_call_form: Optional[Union[str, _AutomaticType]] = Automatic,
    prompt_style: Optional[Union[str, _AutomaticType]] = Automatic,
    **llm_options: Any,
) -> Any:
    """Find textual answers for questions using an LLM call.

    The llm argument should be a LangChain LLM or chat model (invoke-compatible).
    The llm_call callable should accept: query, prompt, response_format, and **llm_options.
    """

    if not isinstance(text, str):
        raise LLMTextualAnswerError("Text must be a string.")

    qs = _normalize_questions(questions)
    normalized_form = _normalize_form(form)

    if prelude is None:
        prelude = Automatic
    if prelude is Automatic:
        prelude_value = DEFAULT_PRELUDE
    elif isinstance(prelude, str):
        prelude_value = prelude
    else:
        raise LLMTextualAnswerError("Prelude must be a string or Automatic.")

    if prompt is None:
        prompt = Automatic
    if prompt is Automatic:
        prompt_value = DEFAULT_PROMPT
    elif isinstance(prompt, str):
        prompt_value = prompt
    else:
        raise LLMTextualAnswerError("Prompt must be a string or Automatic.")

    if request is None:
        request = Automatic
    if request is Automatic:
        request_text = _default_request(qs)
    elif isinstance(request, str):
        request_text = request + "".join(qs)
    else:
        raise LLMTextualAnswerError("Request must be a string or Automatic.")

    if llm_call_form is None:
        llm_call_form = Automatic
    if llm_call_form is Automatic:
        if normalized_form in {"dict", "list", "auto"} and prompt_value == DEFAULT_PROMPT:
            response_format = "string"
        elif normalized_form in {"dict", "list", "auto"}:
            response_format = "json"
        else:
            response_format = "string"
    else:
        response_format = str(llm_call_form).lower()

    template = _build_template(prelude_value, request_text)

    if normalized_form == "template":
        return template

    llm_value = llm or llm_call
    if llm_value is None:
        raise LLMTextualAnswerError(
            "llm (or llm_call) must be provided to invoke the LLM or build the LLMFunction."
        )

    if prompt_style is None:
        prompt_style = Automatic
    if prompt_style is Automatic:
        prompt_style_value = "auto"
    else:
        prompt_style_value = str(prompt_style).lower()
        if prompt_style_value not in {"auto", "chat", "text"}:
            raise LLMTextualAnswerError(
                "prompt_style must be 'auto', 'chat', or 'text'."
            )

    llm_func = LLMFunction(
        template=template,
        prompt=prompt_value,
        response_format=response_format,
        llm=llm_value,
        llm_options=llm_options,
        prompt_style=prompt_style_value,
    )

    if normalized_form == "llm_function":
        return llm_func
    if normalized_form == "function":
        return llm_func.__call__

    result = llm_func(text)

    if (
        normalized_form in {"dict", "list", "auto"}
        and isinstance(result, str)
        and response_format == "string"
    ):
        try:
            parsed = json.loads(_remove_code_fences(result))
        except json.JSONDecodeError:
            parsed = result
    else:
        parsed = result

    if normalized_form in {"dict", "auto"}:
        if isinstance(parsed, dict):
            return parsed
        return parsed

    if normalized_form == "list":
        if isinstance(parsed, dict):
            return list(parsed.values())
        if isinstance(parsed, list):
            return parsed
        return parsed

    if normalized_form == "string":
        if isinstance(parsed, dict) and len(parsed) == 1:
            return next(iter(parsed.values()))
        if isinstance(parsed, str):
            return parsed
        return json.dumps(parsed)

    return parsed


def llm_textual_answer(*args: Any, **kwargs: Any) -> Any:
    return LLMTextualAnswer(*args, **kwargs)


__all__ = [
    "Automatic",
    "DEFAULT_PRELUDE",
    "DEFAULT_PROMPT",
    "LLMFunction",
    "LLMTextualAnswer",
    "LLMTextualAnswerError",
    "llm_textual_answer",
]
