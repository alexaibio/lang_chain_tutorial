"""
https://github.com/ggerganov/llama.cpp/tree/master/examples/server
    - clone
    - build / make
    - run: ./server -m models/7B/ggml-model.gguf -c 2048

with docker
    - docker run -p 8080:8080 -v /path/to/models:/models ghcr.io/ggerganov/llama.cpp:server -m models/7B/ggml-model.gguf -c 512 --host 0.0.0.0 --port 8080

with docker and CUDA:
    docker run -p 8080:8080 -v /path/to/models:/models --gpus all ghcr.io/ggerganov/llama.cpp:server-cuda -m models/7B/ggml-model.gguf -c 512 --host 0.0.0.0 --port 8080 --n-gpu-layers 99
"""

