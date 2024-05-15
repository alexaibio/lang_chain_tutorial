"""
LangChain with OpenAI
https://medium.com/@avra42/getting-started-with-langchain-a-powerful-tool-for-working-with-large-language-models-286419ba0842

# to use LLAMA instead of OpenAI:
 - https://python.langchain.com/v0.1/docs/integrations/llms/llamacpp/
 - https://github.com/ggerganov/llama.cpp

Here is simple Reflection Agentic design pattern
"""
from settings import config
from langchain.chains import LLMChain, SimpleSequentialChain  # import LangChain libraries
from langchain.llms import OpenAI  # import OpenAI model
from langchain.prompts import PromptTemplate # import PromptTemplate


llm = OpenAI(model_name=config["COMPLETION_MODEL"], max_tokens=300, temperature=0.9, openai_api_key=config['OPENAI_API_KEY'])

user_question = "Cyanobacteria can perform photosynthetsis, are they considered as plants?"


# Ask this question directly
template = """{question}\n\n"""
prompt_template = PromptTemplate(input_variables=["question"], template=template)
question_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True)
ans = question_chain.run(user_question)


# Chain 1: Generating assumptions made to help answering
template_assumption = """
    Here is a statement:
    {statement}
    Make a bullet point list of at least 5 assumptions you made when producing the above statement.\n\n
    """
prompt_template = PromptTemplate(input_variables=["statement"], template=template_assumption)
assumptions_chain = LLMChain(llm=llm, prompt=prompt_template)
ans = assumptions_chain.run(user_question)
assumptions_chain_seq = SimpleSequentialChain(chains=[question_chain, assumptions_chain], verbose=True)


# Chain 2: Fact checking the assumptions
template_check = """
    Here is a bullet point list of assertions:
    {assertions}
    For each assertion, determine whether it is true or false. If it is false, explain why.\n\n
    """
prompt_template = PromptTemplate(input_variables=["assertions"], template=template_check)
fact_checker_chain = LLMChain(llm=llm, prompt=prompt_template)
fact_checker_chain_seq = SimpleSequentialChain(chains=[question_chain, assumptions_chain, fact_checker_chain], verbose=True)


# Final Chain: Generating the final answer to the user's question based on the facts and assumptions
template_final = """
    In light of the above facts, how would you answer the question '{}'
    """.format(user_question)
template_final = """{facts}\n""" + template

prompt_template = PromptTemplate(input_variables=["facts"], template=template_final)
answer_chain = LLMChain(llm=llm, prompt=prompt_template)
overall_chain = SimpleSequentialChain(
    chains=[question_chain, assumptions_chain, fact_checker_chain, answer_chain],
    verbose=True,
)

print(f'QUESTION: {user_question}')
ans = overall_chain.run(user_question)



