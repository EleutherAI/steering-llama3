from gpu_graph import run
import sys

name = "0710pm"

commands = {}
port = 8000

for dataset in ["ab", "caa", "prompts"]:
    for leace in ["quad"]:
        cmd = f"python steer.py --dataset {dataset} --mults $(seq -2 .25 2)"
        if leace is not None:
            cmd += f" --leace {leace}"
        commands[f"steer_{dataset}_{leace}"] = cmd

        cmd = f"python eval.py --dataset {dataset} --mults $(seq -2 .25 2)"
        if leace is not None:
            cmd += f" --leace {leace}"
        commands[f"eval_{dataset}_{leace}"] = cmd
        port += 1


for dataset in ["ab"]:
    for leace in [None, "leace", "orth"]:
        cmd = f"python steer.py --dataset {dataset} --mults $(seq -2 .25 2) --layer all --residual"
        if leace is not None:
            cmd += f" --leace {leace}"
        commands[f"steer_{dataset}_{leace}"] = cmd

        cmd = f"python eval.py --dataset {dataset} --mults $(seq -2 .25 2) --layer all --residual"
        if leace is not None:
            cmd += f" --leace {leace}"
        commands[f"eval_{dataset}_{leace}"] = cmd
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