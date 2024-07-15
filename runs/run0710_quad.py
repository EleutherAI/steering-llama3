from gpu_graph import run
import sys

name = "0710quad"

commands = {}

for dataset in ["ab", "caa", "prompts"]:
    cmd = f"python steer.py --dataset {dataset} --mults -1 1 --leace quad"
    commands[f"steer_{dataset}"] = cmd

port = 8000
for dataset in ["ab", "caa", "prompts"]:
    cmd = f"python eval.py --dataset {dataset} --mults -1 1 -port {port} --server --leace quad"
    commands[f"eval_{dataset}"] = cmd
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