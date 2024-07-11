import json
import numpy as np
from common import Settings, parse_settings_args
from matplotlib import pyplot as plt
from argparse import ArgumentParser


def plot(settings, mults, verbose=False):

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
    plt.xlabel("Steering multiplier")
    plt.ylabel("Mean score")
    plt.legend()
    plt.title(settings)
    plt.savefig(settings.plot_path())


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--mults", type=float, nargs="+", required=True)
    parser.add_argument("-v", "--verbose", action="store_true")

    args, settings = parse_settings_args(parser)

    plot(settings, args.mults, args.verbose)
    if args.verbose:
        print("Done!")