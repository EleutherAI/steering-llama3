from argparse import ArgumentParser

from steer import steer
from common import parse_settings_args

import torch
from torch.profiler import profile, record_function, ProfilerActivity

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--mults", type=float, nargs="+", required=True)
    
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--prompts", type=int, default=25)

    args, settings = parse_settings_args(parser)


    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
            # record_shapes=True, 
            profile_memory=True, 
            with_stack=True,
            ) as prof:
        steer(settings, args.mults, repeats=args.repeats, num_prompts=args.prompts, overwrite=args.overwrite)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
    prof.export_chrome_trace(f"profile_steer_{settings.dataset}_{settings.layer}.json")
