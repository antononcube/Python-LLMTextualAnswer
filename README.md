# LLMTextualAnswer

Python package for finding textual answers via LLMs. This is a Python port of the Wolfram Language `LLMTextualAnswer` function, focused on building prompts, wiring LangChain models, and parsing structured outputs.

-----

## Install

```bash
pip install LLMTextualAnswer
```

-----

## Usage

### Question answering

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

For more detailed examples see the notebook ["https://github.com/antononcube/Python-LLMTextualAnswer/blob/main/docs/Basic-usage.ipynb"](https://github.com/antononcube/Python-LLMTextualAnswer/blob/main/docs/Basic-usage.ipynb).


### Classification

Here is a list of workflow construction specifications:

```python
queries = [
    'Make a classifier with the method RandomForest over the data dfTitanic; show precision and accuracy; plot True Positive Rate vs Positive Predictive Value.',
    'Make a recommender over the data frame dfOrders. Give the top 5 recommendations for the profile year:2022, type:Clothing, and status:Unpaid',
    'Create an LSA object over the text collection aAbstracts; extract 40 topics; show statistical thesaurus for "notebook", "equation", "changes", and "prediction"',
    'Compute quantile regression for dfTS with interpolation order 3 and knots 12 for the probabilities 0.2, 0.4, and 0.9.'
]
```

Here are possible workflows names:

```python
workflows = ['Classification', 'Latent Semantic Analysis', 'Quantile Regression', 'Recommendations']
```

For each workflow spec give the corresponding (most likely) workflow name:

```python
for q in queries:
    print("Spec  : " + q)
    print("Class : " + llm_classify(q, workflows, llm = llm, form=dict) + "\n")
```

```
# Spec  : Make a classifier with the method RandomForest over the data dfTitanic; show precision and accuracy; plot True Positive Rate vs Positive Predictive Value.
# Class : Classification
# 
# Spec  : Make a recommender over the data frame dfOrders. Give the top 5 recommendations for the profile year:2022, type:Clothing, and status:Unpaid
# Class : Recommendations
# 
# Spec  : Create an LSA object over the text collection aAbstracts; extract 40 topics; show statistical thesaurus for "notebook", "equation", "changes", and "prediction"
# Class : Latent Semantic Analysis
# 
# Spec  : Compute quantile regression for dfTS with interpolation order 3 and knots 12 for the probabilities 0.2, 0.4, and 0.9.
# Class : Quantile Regression
# 

```

-----

## Notes

- `LLMTextualAnswer` accepts LangChain chat/text models that support `.invoke`.
- Use `prompt_style="chat"` or `prompt_style="text"` if auto-detection is not desired.
- When you want only the prompt template, pass `form="StringTemplate"`.
