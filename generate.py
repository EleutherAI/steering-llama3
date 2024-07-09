from argparse import ArgumentParser
from dataclasses import asdict, dataclass, field
from pathlib import Path
import json

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import cached_property, get_layer_list
from settings import Settings


@dataclass
class ActivationAdder:
    # [num samples, num features]
    positives: list = field(default_factory=list)
    negatives: list = field(default_factory=list)

    @cached_property
    def steering_vector(self):
        # [num features]
        u = torch.mean(self.positives, dim=0) - torch.mean(self.negatives, dim=0)

        return u

    def record_activations(self, positive: bool = True):
        def hook(model, input, output):
            act = output[0]
            act = act.detach()[:, -1, :].cpu()
            if positive:
                self.positives.append(act)
            else:
                self.negatives.append(act)

        return hook


def main():
    pass


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--dataset", type=str, choices=["prompts", "rimskyab"], default="prompts")

    main(settings)