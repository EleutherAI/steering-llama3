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

    if settings.behavior is None:
        labels = ["harmful", "harmless"]
    else:
        labels = [None]

    for label in labels:
        means = []
        q25s = []
        q75s = []
        good_mults = []

        for mult in mults:
            if verbose:
                print(f"Loading {settings.response_path(mult)}")
            with open(settings.response_path(mult)) as f:
                prompts = json.load(f)

            score_field = "scores" if "scores" in prompts[0] else "num_scores"

            for prompt in prompts:
                if label is None:
                    if "label" in prompt:
                        raise ValueError(f"Unexpected label in prompt: {prompt}")
                    if not all(isinstance(score, int) and 0 <= score <= 10 for score in prompt[score_field]):
                        raise ValueError(f"Unexpected scores in prompt: {prompt}")
                if label is not None:
                    if "label" not in prompt:
                        raise ValueError(f"Missing label in prompt: {prompt}")
                    if prompt["label"] not in ["harmful", "harmless"]:
                        raise ValueError(f"Unexpected label in prompt: {prompt}")
                    if not all(isinstance(score, bool) for score in prompt[score_field]):
                        raise ValueError(f"Unexpected scores in prompt: {prompt}")

            if label is not None:
                prompt_scores = [np.mean(prompt[score_field]) for prompt in prompts if prompt["label"] == label]
            else:
                prompt_scores = [np.mean(prompt[score_field]) for prompt in prompts]

            mean = np.mean(prompt_scores)
            q25 = np.percentile(prompt_scores, 25)
            q75 = np.percentile(prompt_scores, 75)

            if verbose:
                print(f"Mult {mult}: mean {mean}, 25th {q25}, 75th {q75}")

            means.append(mean)
            q25s.append(q25)
            q75s.append(q75)
            good_mults.append(mult)

        plt.plot(good_mults, means, label=label)
        plt.fill_between(good_mults, q25s, q75s, alpha=0.3)

    # xlim special cases:
    if settings.dataset == "prompts":
        if settings.residual:
            plt.xlim(-1.5, 1)
        elif settings.layer == "all":
            plt.xlim(-.25, .2)
    plt.grid()
    plt.xlabel("Steering multiplier")
    plt.ylabel("Mean score")
    if settings.behavior is None:
        plt.ylim(0, 1)
        plt.legend()
    else:
        plt.ylim(0, 10)
    plt.title(settings)
    plt.savefig(settings.plot_path())
    plt.close()
    if verbose:
        print(f"Saved {settings.plot_path()}")

#%%
def plot_all(filter_settings=None, verbose=False):
    settings_mults = {}
    for filename in os.listdir("artifacts/responses"):
        # skip .gitkeep
        if filename == ".gitkeep":
            continue
        suffix = filename.split("responses_")[1].split(".json")[0]
        if suffix.endswith("__pretty"):
            continue
        
        # open to check for scores
        with open(f"artifacts/responses/{filename}") as f:
            prompts = json.load(f)
        if not prompts or ("scores" not in prompts[0] and "num_scores" not in prompts[0]):
            continue

        settings, notkwargs = parts_to_settings(suffix_to_parts(suffix))
        settings_mults[(settings)] = settings_mults.get((settings), []) + [notkwargs["mult"]]

    if filter_settings is None:
        for settings in settings_mults:
            mults = sorted(settings_mults[settings])
            plot(settings, mults, verbose)
    else:
        mults = sorted(settings_mults[filter_settings])
        plot(filter_settings, mults, verbose)

#%%

if __name__ == "__main__":
    if len(sys.argv) == 1:
        plot_all()
        sys.exit(0)
    parser = ArgumentParser()
    parser.add_argument("--mults", type=float, nargs="+", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")

    args, settings = parse_settings_args(parser)

    if args.mults is None:
        plot_all(settings, args.verbose)
    else:
        plot(settings, args.mults, args.verbose)
    if args.verbose:
        print("Done!")