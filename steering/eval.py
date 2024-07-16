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
from caa_test_open_ended import eval_prompt as caa_eval_prompt


def evaluate(settings, mults, port, overwrite):
    tokenizer = AutoTokenizer.from_pretrained(EVAL_MODEL)

    score_field = "scores" if settings.behavior is None else "num_scores"
    eval_prompt_fn = eval_prompt if settings.behavior is None else lambda q, r, t, url: caa_eval_prompt(settings.behavior, q, r, t, url)

    for mult in mults:
        print(f"Evaluating steering by {mult}...")
        with open(settings.response_path(mult)) as f:
            prompts = json.load(f)

        if score_field in prompts[0] and not overwrite:
            print("Already evaluated. Skipping...")
            continue

        for prompt in tqdm(prompts):
            scores = []
            for response in prompt["responses"]:
                question = prompt["prompt"]
                score = eval_prompt_fn(question, response, tokenizer, url=f"http://127.0.0.1:{port}/generate")
                scores.append(score)
            prompt[score_field] = scores

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
    parser.add_argument("--wait", type=int, default=150)

    parser.add_argument("--overwrite", action="store_true")

    args, settings = parse_settings_args(parser)

    if args.server:
        command = shlex.split(f'python -m outlines.serve.serve --model="casperhansen/llama-3-70b-instruct-awq" --port {args.port}')
        
        process = subprocess.Popen(command)
        
        print(f"Server started on port {args.port}. Waiting {args.wait} sec for it to be ready...")
        sleep(args.wait)
    try:
        evaluate(settings, args.mults, args.port, args.overwrite)
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