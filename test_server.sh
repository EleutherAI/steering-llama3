#!/bin/bash

# export CUDA_VISIBLE_DEVICES=1
python -m outlines.serve.serve --model="casperhansen/llama-3-70b-instruct-awq" --port 8005
