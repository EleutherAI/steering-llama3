#%%
import os
import sys
import json
import numpy as np
from matplotlib import pyplot as plt
from argparse import ArgumentParser
from dataclasses import asdict

from common import Settings, parse_settings_args, suffix_to_parts, parts_to_settings

def plot(settings, mults, verbose=False):
    plt.figure()

    for label in ["harmful", "harmless"]:
        means = []
        q25s = []
        q75s = []

        for mult in mults:
            if verbose:
                print(f"Loading {settings.response_path(mult)}")
            with open(settings.response_path(mult)) as f:
                prompts = json.load(f)

            prompt_scores = [np.mean(prompt["scores"]) for prompt in prompts if prompt["label"] == label]
            mean = np.mean(prompt_scores)
            q25 = np.percentile(prompt_scores, 25)
            q75 = np.percentile(prompt_scores, 75)

            means.append(mean)
            q25s.append(q25)
            q75s.append(q75)

        # old version: error bars
        # plt.errorbar(mults, means, yerr=stds, label=label)

        # new version: shaded region
        plt.plot(mults, means, label=label)
        plt.fill_between(mults, q25s, q75s, alpha=0.3)
    plt.grid()
    plt.xlabel("Steering multiplier")
    plt.ylabel("Mean score")
    plt.legend()
    plt.title(settings)
    plt.savefig(settings.plot_path())
    plt.close()

#%%
def plot_all():
    settings_mults = {}
    for filename in os.listdir("artifacts/responses"):
        #print(filename)
        suffix = filename.split("responses_")[1].split(".json")[0]
        if suffix.endswith("__pretty"):
            continue
        
        # open to check for scores
        with open(f"artifacts/responses/{filename}") as f:
            prompts = json.load(f)
        if not prompts or "scores" not in prompts[0]:
            continue

        settings, notkwargs = parts_to_settings(suffix_to_parts(suffix))
        settings_mults[(settings)] = settings_mults.get((settings), []) + [notkwargs["mult"]]

    for settings in settings_mults:
        mults = sorted(settings_mults[settings])
        plot(settings, mults)

#%%

if __name__ == "__main__":
    if len(sys.argv) == 1:
        plot_all()
        sys.exit(0)
    parser = ArgumentParser()
    parser.add_argument("--mults", type=float, nargs="+", required=True)
    parser.add_argument("-v", "--verbose", action="store_true")

    args, settings = parse_settings_args(parser)

    plot(settings, args.mults, args.verbose)
    if args.verbose:
        print("Done!")