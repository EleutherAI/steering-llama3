from multiprocessing import Manager, Process, Lock
import subprocess
from datetime import datetime
from dataclasses import dataclass
from typing import List

@dataclass
class Context:
    lock: Lock
    claimed: List[str]
    completed: List[str]
    free_gpus: List[int]
    jobs: List[str]
    commands: dict
    dependencies: dict

# Function that runs the job on a GPU
def run_on_gpu(gpu: int, cmd: str):
    print()
    print(f"Starting on GPU {gpu}: {cmd}")
    print("at time:", datetime.now())
    command = f"CUDA_VISIBLE_DEVICES={gpu} {cmd}"
    try:
        subprocess.run(command, shell=True, check=True)
    except Exception as e:
        print()
        for _ in range(3):
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print()
        print(f"[WARN] Error on GPU {gpu}: {cmd}")
        print(e)
        print()
        # append to log file
        with open(f"logs/gpu_graph.err", "a") as f:
            f.write(f"[{datetime.now()}]\n{gpu}\n{cmd}\n{e}\n\n")
    else:
        print(f"Finished on GPU {gpu}: {cmd}")
    finally:
        print("at time:", datetime.now())
        print() 


def worker(gpu, job, ctx):
    run_on_gpu(gpu, ctx.commands[job])
    with ctx.lock:
        ctx.completed.append(job)
        ctx.free_gpus.append(gpu)
        print(f"GPU {gpu} released from {job}")
    
    launch_available(ctx)
        

def launch_available(ctx: Context):
    with ctx.lock:
        avail_jobs = [
            job for job in ctx.jobs if (
                job not in ctx.claimed and
                all(dep in ctx.completed for dep in ctx.dependencies[job])
            )
        ]

        processes = []
        while ctx.free_gpus and avail_jobs:
            gpu = ctx.free_gpus.pop(0)
            job = avail_jobs.pop(0)
            ctx.claimed.append(job)
            p = Process(target=worker, args=(gpu, job, ctx))
            processes.append(p)
            print(f"GPU {gpu} claimed for {job}")

        if not processes:
            if all(job in ctx.claimed for job in ctx.jobs):
                print("All jobs running or completed :)")
            elif avail_jobs:
                print("[WARN] No free GPUs -- this should not happen! :(")
            else:
                print("Blocked on dependencies :|")
            print(f"Running: {[j for j in ctx.claimed if j not in ctx.completed]}")
            print(f"Free GPUs: {ctx.free_gpus}")

    for p in processes:
        p.start()

    # Wait for all worker processes to finish
    for p in processes:
        p.join()


def run(gpus, commands: dict, dependencies: dict):
    """
    Run a list of commands on a list of GPUs, respecting dependencies.

    Args:
        gpus (list): List of GPU IDs to use.
        commands (dict): Dictionary of command names and their corresponding shell commands.
        dependencies (dict): Dictionary of command names and their lists of dependencies.
    """
    # Create a shared job list and a lock
    manager = Manager()
    claimed = manager.list([])
    completed = manager.list([])
    free_gpus = manager.list(gpus.copy())
    lock = manager.Lock()

    jobs = list(commands.keys())

    print("Jobs:", jobs)

    ctx = Context(
        lock=lock,
        claimed=claimed,
        completed=completed,
        free_gpus=free_gpus,
        jobs=jobs,
        commands=commands,
        dependencies=dependencies
    )

    launch_available(ctx)