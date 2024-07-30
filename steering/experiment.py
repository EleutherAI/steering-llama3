from generate_vectors import generate_vectors
from steer import steer
from eval import evaluate
from common import Settings
from caa_test_open_ended import ALL_BEHAVIORS

import numpy as np
import shlex
import subprocess
from time import sleep

import sys

configs = []
port = 8000

for model in ["llama2", "llama3"]:
    for dataset in ["ab", "openr", "opencon", "caa", "abcon", "prompts"]:
        for behavior in ((ALL_BEHAVIORS + [None]) if dataset in ["ab", "openr"] else [None]):
            for layer in ["all", "one"]:
                for logit in [False, True]:
                    if logit and dataset not in ["ab", "abcon"]:
                        continue

                    for leace in [None, "orth", "leace"]:
                        if logit and leace is not None:
                            continue

                        maxmult = {
                            "ab": 4,
                            "openr": 2,
                            "opencon": 1,
                            "caa": 1.5,
                            "abcon": 3,
                            "prompts": 1.5,
                        }[dataset]

                        if logit:
                            maxmult *= 4
                        if layer == 15:
                            maxmult *= 2.5

                        maxmult *= 1.5
                        multstep = maxmult / 15

                        configs.append((model, dataset, behavior, layer, logit, leace, maxmult, multstep, port))
                        port += 1


def main(cfg: int):
    model, dataset, behavior, layer, logit, leace, maxmult, multstep, port = configs[cfg]

    if layer == "one":
        if model == "llama2":
            layer = 13
        else:
            layer = 15

    residual = (layer == "all")

    settings = Settings(
        model=model,
        dataset=dataset, 
        residual=residual,
        logit=logit,
        behavior=behavior,
        scrub=False,
    )
    generate_vectors(settings)


    settings = Settings(
        model=model,
        dataset=dataset, 
        residual=residual,
        logit=logit,
        behavior=behavior,
        scrub=False,
        layer=layer, 
        temp=1.5,
        leace=leace,
        toks=None,
    )

    mults = list(np.round(np.arange(-maxmult, maxmult + multstep, multstep), 5))
    
    steer(settings, mults)

    command = shlex.split(f'python -m outlines.serve.serve --model="casperhansen/llama-3-70b-instruct-awq" --port {port}')
        
    process = subprocess.Popen(command)
    
    print(f"Server started on port {port}. Waiting {150} sec for it to be ready...")
    sleep(150)

    try:
        evaluate(settings, mults, port)
    finally:
        process.terminate()
        print("Server terminated. Waiting 5s...")
        sleep(5)
        if process.poll() is None:
            process.kill()
            print("Server killed. Waiting...")
            process.wait()
            print("Server process finished.")


if __name__ == "__main__":
    # 0..275 inclusive
    main(int(sys.argv[1]))