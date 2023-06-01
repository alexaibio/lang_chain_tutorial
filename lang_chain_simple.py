"""
LangChain with OpenAI
https://medium.com/@avra42/getting-started-with-langchain-a-powerful-tool-for-working-with-large-language-models-286419ba0842


"""
import openai
from langchain.chains import LLMChain, SimpleSequentialChain  # import LangChain libraries
from langchain.llms import OpenAI  # import OpenAI model
from langchain.prompts import PromptTemplate # import PromptTemplate
from settings import config

openai.api_key = config["OPENAI_API_KEY"]
llm = OpenAI(model_name="text-davinci-003", max_tokens=300, temperature=0.9, openai_api_key=config['OPENAI_API_KEY'])

user_question = "Cyanobacteria can perform photosynthetsis, are they considered as plants?"

# Chain 1: Generating a rephrased version of the user's question
template = """{question}\n\n"""
prompt_template = PromptTemplate(input_variables=["question"], template=template)
question_chain = LLMChain(llm=llm, prompt=prompt_template)

# Chain 2: Generating assumptions made in the statement
template = """Here is a statement:
    {statement}
    Make a bullet point list of at least 5 assumptions you made when producing the above statement.\n\n"""
prompt_template = PromptTemplate(input_variables=["statement"], template=template)
assumptions_chain = LLMChain(llm=llm, prompt=prompt_template)
assumptions_chain_seq = SimpleSequentialChain(chains=[question_chain, assumptions_chain], verbose=True)

# Chain 3: Fact checking the assumptions
template = """Here is a bullet point list of assertions:
{assertions}
For each assertion, determine whether it is true or false. If it is false, explain why.\n\n"""
prompt_template = PromptTemplate(input_variables=["assertions"], template=template)
fact_checker_chain = LLMChain(llm=llm, prompt=prompt_template)
fact_checker_chain_seq = SimpleSequentialChain(chains=[question_chain, assumptions_chain, fact_checker_chain], verbose=True)

# Final Chain: Generating the final answer to the user's question based on the facts and assumptions
template = """In light of the above facts, how would you answer the question '{}'""".format(
    user_question
)
template = """{facts}\n""" + template
prompt_template = PromptTemplate(input_variables=["facts"], template=template)
answer_chain = LLMChain(llm=llm, prompt=prompt_template)
overall_chain = SimpleSequentialChain(
    chains=[question_chain, assumptions_chain, fact_checker_chain, answer_chain],
    verbose=True,
)

print(f'QUESTION: {user_question}')
ans = overall_chain.run(user_question)

#print(ans)

