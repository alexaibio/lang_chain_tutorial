"""
How to run llama.cpp as an OpenAI Compatible Server
    https://llama-cpp-python.readthedocs.io/en/latest/server/
    pip install llama-cpp-python[server]
    server options: python3 -m llama_cpp.server --help
    run server: python3 -m llama_cpp.server --model ./models/mistral-7b-instruct-v0.2.Q3_K_L.gguf
    api_base="http://localhost:8000/v1"
"""



