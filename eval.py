from argparse import ArgumentParser
import json
import subprocess
from tqdm import tqdm
from time import sleep

from transformers import AutoTokenizer
import shlex

from common import Settings, parse_settings_args
from utils import force_save
from refusal_test_open_ended import eval_prompt, EVAL_MODEL


def evaluate(settings, mults, port):
    tokenizer = AutoTokenizer.from_pretrained(EVAL_MODEL)

    for mult in mults:
        print(f"Evaluating steering by {mult}...")
        with open(settings.response_path(mult)) as f:
            prompts = json.load(f)

        if "scores" in prompts[0]:
            print("Already evaluated. Skipping...")
            continue

        for prompt in tqdm(prompts):
            scores = []
            for response in prompt["responses"]:
                question = prompt["prompt"]
                score = eval_prompt(question, response, tokenizer, url=f"http://127.0.0.1:{port}/generate")
                scores.append(score)
            prompt["scores"] = scores

        force_save(
            prompts,
            settings.response_path(mult),
            mode="json"
        )

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--mults", type=float, nargs="+", required=True)

    parser.add_argument("--port", type=int, default=8005)
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--wait", type=int, default=120)

    args, settings = parse_settings_args(parser)

    if args.server:
        command = shlex.split(f'python -m outlines.serve.serve --model="casperhansen/llama-3-70b-instruct-awq" --port {args.port}')
        
        process = subprocess.Popen(command)
        
        print(f"Server started on port {args.port}. Waiting {args.wait} sec for it to be ready...")
        sleep(args.wait)
    try:
        evaluate(settings, args.mults, args.port)
        print("Done!")
    finally:
        if args.server:
            process.terminate()
            print("Server terminated. Waiting 5s...")
            sleep(5)
            if process.poll() is None:
                process.kill()
                print("Server killed. Waiting...")
                process.wait()
                print("Server process finished.")