"""
Taken from https://www.datacamp.com/tutorial/llama-cpp-tutorial

"""
from llama_cpp import Llama


##### from local file
my_model_path = "../models/zephyr-7b-beta.Q4_0.gguf"
CONTEXT_SIZE = 512

# LOAD THE MODEL
zephyr_model = Llama(model_path=my_model_path, n_ctx=CONTEXT_SIZE)

user_prompt = 'Give me a short marketing plan for a service oriented software company.'

model_output = zephyr_model(
    user_prompt,
    max_tokens=300,
    temperature=0.1,
    top_p=0.1,
    echo=False,
)

final_result = model_output["choices"][0]["text"].strip()

print(final_result)