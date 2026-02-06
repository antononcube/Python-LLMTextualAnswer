"""Python port of the Wolfram Language LLMTextualAnswer function."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import re

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
    echo: bool = False

    def __call__(self, text: str) -> Any:
        if StrOutputParser is None or JsonOutputParser is None:
            raise LLMTextualAnswerError(
                "langchain-core is required to use LLMFunction."
            )

        query = self.template.format(text=text)

        if self.echo:
            print(query)

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


def _default_request(questions: Sequence[str], n: int = 1) -> str:
    plural = "s" if len(questions) > 1 else ""
    if n > 1:
        if len(questions) == 1:
            prefix = f"give the top {n} answers for"
        else:
            prefix = f"list the top {n} answers for each of"
        return f"{prefix} the question{plural}:" + "".join(questions)
    else:
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

# =========================================================
# LLM textual answer
# =========================================================

def LLMTextualAnswer(
    text: str,
    questions: Union[str, Sequence[str]],
    n: int = 1,
    form: FormType = Automatic,
    *,
    prelude: Optional[Union[str, _AutomaticType]] = Automatic,
    prompt: Optional[Union[str, _AutomaticType]] = Automatic,
    request: Optional[Union[str, _AutomaticType]] = Automatic,
    llm: Optional[Any] = None,
    llm_call: Optional[Callable[..., Any]] = None,
    llm_call_form: Optional[Union[str, _AutomaticType]] = Automatic,
    prompt_style: Optional[Union[str, _AutomaticType]] = Automatic,
    echo: bool = False,
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
        request_text = _default_request(qs, n)
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
        echo=echo,
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

# =========================================================
# LLM classification helpers
# =========================================================

def _normalize_class_labels(class_labels: Sequence[Any]) -> List[str]:
    if not isinstance(class_labels, Sequence) or isinstance(
        class_labels, (str, bytes, bytearray)
    ):
        raise LLMTextualAnswerError("class_labels must be a sequence of labels.")
    if not class_labels:
        raise LLMTextualAnswerError("class_labels cannot be empty.")
    return [str(label) for label in class_labels]


def _build_classify_question(class_labels: Sequence[str], epilog: Any, sep: str="\n") -> str:
    question_lines = [
        f"{index}) {label}" for index, label in enumerate(class_labels, start=1)
    ]
    question = sep.join(question_lines)
    if epilog is None or epilog is Automatic:
        question += "\nYour answer should have one of the labels and nothing else."
    elif isinstance(epilog, str):
        question += epilog
    else:
        raise LLMTextualAnswerError("epilog must be a string or Automatic.")
    return question


def _extract_classification(result: Any, class_labels: Sequence[str]) -> Any:
    if isinstance(result, dict) and result:
        return next(iter(result.values()))
    elif isinstance(result, (list, tuple, set)) and result:
        return next(iter(result))

    if not isinstance(result, str):
        return result

    match = re.match(r"^\s*(\d+)", result)
    if match:
        index = int(match.group(1))
        if 1 <= index <= len(class_labels):
            return class_labels[index - 1]
        return match.group(1)

    matches = [label for label in class_labels if label in result]
    if matches:
        return matches
    return result

# =========================================================
# LLM classify
# =========================================================

def llm_classify(
    text: Union[str, Sequence[str]],
    class_labels: Sequence[Any],
    *,
    epilog: Optional[Union[str, _AutomaticType]] = Automatic,
    request: Optional[Union[str, _AutomaticType]] = "which of these labels characterizes it:",
    form: FormType = "string",
    llm: Optional[Any] = None,
    llm_call: Optional[Callable[..., Any]] = None,
    echo: bool = False,
    sep: str = " ;\n",
    **llm_options: Any,
) -> Any:
    """Classify text into the given labels using an LLM."""

    labels = _normalize_class_labels(class_labels)
    question = _build_classify_question(labels, epilog, sep=sep)

    if isinstance(text, Sequence) and not isinstance(text, (str, bytes, bytearray)):
        return [
            llm_classify(
                item,
                labels,
                epilog=epilog,
                request=request,
                form=form,
                llm=llm,
                llm_call=llm_call,
                echo=echo,
                **llm_options,
            )
            for item in text
        ]

    if not isinstance(text, str):
        raise LLMTextualAnswerError("Text must be a string.")

    result = llm_textual_answer(
        text,
        question,
        n = 1,
        form=form,
        request=request,
        llm=llm,
        llm_call=llm_call,
        **llm_options,
    )

    if echo:
        print("llm_textual_answer result:\n" + str(result))

    return _extract_classification(result, labels)

# =========================================================
# Package symbols
# =========================================================

__all__ = [
    "Automatic",
    "DEFAULT_PRELUDE",
    "DEFAULT_PROMPT",
    "LLMFunction",
    "LLMTextualAnswer",
    "LLMTextualAnswerError",
    "llm_classify",
    "llm_textual_answer",
]
