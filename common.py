from dataclasses import dataclass
from typing import Optional, Union

from utils import cached_property


def parts_to_suffix(parts):
    return "_".join(f"{k}{v}" for k, v in parts.items() if v is not None)

def suffix_to_parts(suffix):
    parts = {}
    for part in suffix.split("_"):
        k, v = part[0], part[1:]
        parts[k] = v
    return parts

def parts_to_settings(parts):
    keymap = {
        'M': 'model',
        'D': 'dataset',
        'L': 'layer',
        'T': 'temp',
        'C': 'leace',
        'R': 'residual',
        'X': 'mult',
        'P': 'positive',
    }
    valmap = {
        'L': lambda x: int(x) if x != "all" else x,
        'T': float,
        'X': float,
        'P': lambda x: x == "+",
        'R': lambda x: x == "res",
    }
    both = {keymap[k]: valmap.get(k, str)(v) for k, v in parts.items()}
    skip = {'mult', 'positive'}
    kwargs = {k: v for k, v in both.items() if k not in skip}
    notkwargs = {k: v for k, v in both.items() if k in skip}
    return Settings(**kwargs), notkwargs


@dataclass
class Settings:
    model: str = "llama3"
    dataset: str = "prompts"
    leace: Optional[str] = None
    temp: float = 1.5
    layer: Optional[Union[int, str]] = None
    residual: bool = False

    @cached_property
    def model_id(self):
        if self.model == "llama3":
            return "meta-llama/Meta-Llama-3-8B-Instruct"
        else:
            raise ValueError(f"Unknown model: {self.model}")

    def vec_parts(self, layer=None):
        return {
            'M': self.model,
            'D': self.dataset,
            'R': 'res' if self.residual else None,
            'L': self.layer if layer is None else layer,
        }

    def response_parts(self):
        return self.vec_parts() | {
            'T': self.temp,
            'C': self.leace,
        }

    def acts_path(self, positive, layer):
        parts = self.vec_parts(layer) | {
            'P': "+" if positive else "-",
        }
        return f"artifacts/acts/acts_{parts_to_suffix(parts)}.pt"

    def vec_path(self, layer):
        parts = self.vec_parts(layer)
        return f"artifacts/vecs/vec_{parts_to_suffix(parts)}.pt"

    def response_path(self, mult):
        parts = self.response_parts() | {
            'X': mult,
        }
        return f"artifacts/responses/responses_{parts_to_suffix(parts)}.json"

    def plot_path(self):
        parts = self.response_parts()
        return f"plots/plot_{parts_to_suffix(parts)}.png"


    def __str__(self):
        parts = self.response_parts()
        return ' '.join(f"{k}:{v}" for k, v in parts.items() if v is not None)


def parse_settings_args(parser, generate=False):
    parser.add_argument("--dataset", type=str, choices=["prompts", "caa", "ab", "opencon", "openr", "abcon"], default="prompts")
    parser.add_argument("--residual", action="store_true")

    if not generate:
        parser.add_argument("--layer", type=lambda x: int(x) if x != "all" else x, default=15)
        parser.add_argument("--temp", type=float, default=1.5)
        parser.add_argument("--leace", type=str, default=None, choices=["leace", "orth", "quad"])

    args = parser.parse_args()

    if generate:
        settings = Settings(dataset=args.dataset, residual=args.residual)
    else:
        settings = Settings(
            dataset=args.dataset, 
            residual=args.residual,
            layer=args.layer, 
            temp=args.temp,
            leace=args.leace,
        )

    return args, settings