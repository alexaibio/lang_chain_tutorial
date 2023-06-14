from langchain.llms import OpenAI
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.agents import load_tools
from dotenv import load_dotenv
load_dotenv('.env')


####### task 1: use PAL math tool to solve math question
llm = OpenAI(temperature=0)
tools = load_tools(["pal-math"], llm=llm)

agent = initialize_agent(tools,
                         llm,
                         agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                         verbose=True)
print('--------------------')
print(agent.agent.llm_chain.prompt.template)
#agent.run("If my age is half of my dad's age and he is going to be 60 next year, what is my current age?")


###### task 2: use serpapi to access real time information
# model does not know the current age of Demi Moor, so an error
#agent.run("My age is half of my dad's age. Next year he is going to be same age as Demi Moore. What is my current age?")

# we need to assess a tool for answering questions about current events - serpapi
tools = load_tools(["pal-math", "serpapi"], llm=llm)
agent = initialize_agent(tools,
                         llm,
                         agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                         verbose=True)
agent.run("My age is half of my dad's age. Next year he is going to be same age as Demi Moore. What is my current age?")
