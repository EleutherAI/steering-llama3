from dataclasses import dataclass
from typing import Optional

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
        'X': 'mult',
        'P': 'positive',
    }
    valmap = {
        'L': int,
        'T': float,
        'X': float,
        'P': lambda x: x == "+",
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
    layer: Optional[int] = None

    @cached_property
    def model_id(self):
        if self.model == "llama3":
            return "meta-llama/Meta-Llama-3-8B-Instruct"
        else:
            raise ValueError(f"Unknown model: {self.model}")

    def vec_parts(self):
        return {
            'M': self.model,
            'D': self.dataset,
            'L': self.layer,
            'C': self.leace,
        }

    def response_parts(self):
        return self.vec_parts() | {
            'T': self.temp,
        }

    def acts_path(self, positive):
        parts = self.vec_parts() | {
            'P': "+" if positive else "-",
        }
        return f"artifacts/acts/acts_{parts_to_suffix(parts)}.pt"

    def vec_path(self):
        parts = self.vec_parts()
        return f"artifacts/vecs/vec_{parts_to_suffix(parts)}.pt"

    def response_path(self, mult):
        parts = self.response_parts() | {
            'X': mult,
        }
        return f"artifacts/responses/responses_{parts_to_suffix(parts)}.json"


def parse_settings_args(parser, generate=False):
    parser.add_argument("--dataset", type=str, choices=["prompts", "caa", "ab"], default="prompts")

    if not generate:
        parser.add_argument("--layer", type=int, default=15)
        parser.add_argument("--temp", type=float, default=1.5)

    args = parser.parse_args()

    if generate:
        settings = Settings(dataset=args.dataset)
    else:
        settings = Settings(
            dataset=args.dataset, 
            layer=args.layer, 
            temp=args.temp
        )

    return args, settings