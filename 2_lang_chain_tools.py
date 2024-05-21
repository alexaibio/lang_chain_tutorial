"""
Agentic workflow with external tools with LangChain
https://python.langchain.com/v0.2/docs/how_to/tools_chain/
Brief:
    As a part of an answer LLM has to generate a string for API calling an external tools.
    Those API calls shall be executed later on and their result has to be looped back to LLM
    to get a final answer
"""

from langchain_core.tools import tool


# create a custom tool from a function (might be a REST API call
@tool
def multiply(first_int: int, second_int: int) -> int:
    """Multiply two integers together."""
    return first_int * second_int


print(multiply.description)
print(multiply.args)
res = multiply.invoke({"first_int": 4, "second_int": 5})
print(res)