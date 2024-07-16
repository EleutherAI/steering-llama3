import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from steering.gpu_graph import run
from steering.caa_test_open_ended import ALL_BEHAVIORS

import numpy as np

name = "0716_beh"

commands = {}
dependencies = {}
port = 8000

# starts = list(np.round(np.arange(-0.2, 0.21, .04), 3))[7:]


for dataset in ["ab"]:
    for behavior in ALL_BEHAVIORS + [None]:
        for layer in ["all", 15]:
            cmd_suffix = f"--dataset {dataset}"
            if behavior is not None:
                cmd_suffix += "--behavior {behavior}"
            if layer == "all":
                cmd_suffix += " --residual"
            cmd = f"python steering/generate_vectors.py {cmd_suffix}"

            commands[f"gen_{dataset}_{behavior}_{layer}"] = cmd
            dependencies[f"gen_{dataset}_{behavior}_{layer}"] = []

            for leace in [None, "orth"]:
                steer_suffix = f"{cmd_suffix} --layer {layer} --mults -2 -1 -.5 -.2 -.1 0 .1 .2 .5 1 2 --overwrite"
                if leace is not None:
                    steer_suffix += f" --leace {leace}"
                cmd = f"python steering/steer.py {steer_suffix}"
                commands[f"steer_{dataset}_{behavior}_{layer}_{leace}"] = cmd
                dependencies[f"steer_{dataset}_{behavior}_{layer}_{leace}"] = [f"gen_{dataset}_{behavior}_{layer}"]

                cmd = f"python steering/eval.py {steer_suffix} --port {port} --server"
                commands[f"eval_{dataset}_{behavior}_{layer}_{leace}"] = cmd
                dependencies[f"eval_{dataset}_{behavior}_{layer}_{leace}"] = [f"steer_{dataset}_{behavior}_{layer}_{leace}"]
                port += 1

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