import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from steering.gpu_graph import run

import numpy as np

name = "0715_lallfine"

commands = {}
port = 8000

starts = list(np.round(np.arange(-0.2, 0.21, .04), 3))[7:]

for dataset in ["prompts"]:
    for leace in [None]:
        for start in starts:
            cmd = f"python steering/steer.py --dataset {dataset} --mults {start} {start+0.02} --layer all"
            if leace is not None:
                cmd += f" --leace {leace}"
            commands[f"steer_{dataset}_{leace}_{start}"] = cmd

            cmd = f"python steering/eval.py --dataset {dataset} --mults {start} {start+0.02} --layer all --port {port} --server"
            if leace is not None:
                cmd += f" --leace {leace}"
            commands[f"eval_{dataset}_{leace}_{start}"] = cmd
            port += 1

dependencies = {job: [] for job in commands}

for job in commands:
    if job.startswith("eval_"):
        dependencies[job].append(f"steer_{job[5:]}")

for job in commands:
    commands[job] += f" > logs/{name}_{job}.log 2> logs/{name}_{job}.err"

if __name__ == "__main__":
    # usage: python run_whatever.py 1,2,3,4,5,6,7
    if len(sys.argv) == 1:
        # default to all GPUs
        gpus = list(range(8))
    else:
        gpus = [int(gpu) for gpu in sys.argv[1].split(",")]

    for job in commands:
        print(f"{job}: {commands[job]} <-- {dependencies[job]}")
    print()
    print(f"Running on GPUs: {gpus}")

    run(gpus, commands, dependencies)