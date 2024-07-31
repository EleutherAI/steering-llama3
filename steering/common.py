from dataclasses import dataclass
from typing import Optional, Union

from utils import cached_property

from caa_test_open_ended import ALL_BEHAVIORS


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
        'B': 'behavior',
        'L': 'layer',
        'T': 'temp',
        'C': 'leace',
        'K': 'toks',
        'R': 'residual',
        'G': 'logit',
        'X': 'mult',
        'P': 'positive',
        'S': 'scrub',
    }
    valmap = {
        'L': lambda x: int(x) if x != "all" else x,
        'T': float,
        'X': float,
        'P': lambda x: x == "+",
        'R': lambda x: x == "res",
        'G': lambda x: x == "log",
        'S': lambda x: x == "scrub",
    }
    both = {keymap[k]: valmap.get(k, str)(v) for k, v in parts.items()}
    skip = {'mult', 'positive'}
    kwargs = {k: v for k, v in both.items() if k not in skip}
    notkwargs = {k: v for k, v in both.items() if k in skip}
    return Settings(**kwargs), notkwargs


@dataclass(unsafe_hash=True)
class Settings:
    model: str = "llama3"
    dataset: str = "prompts"
    leace: Optional[str] = None
    temp: float = 1.5
    layer: Optional[Union[int, str]] = None
    residual: bool = False
    logit: bool = False
    toks: Optional[str] = None
    behavior: Optional[str] = None
    scrub: bool = False

    @cached_property
    def model_id(self):
        if self.model == "llama3":
            return "meta-llama/Meta-Llama-3-8B-Instruct"
        elif self.model == "llama31":
            return "meta-llama/Meta-Llama-3.1-8B-Instruct"
        elif self.model == "llama2":
            return "meta-llama/Llama-2-7b-chat-hf"
        else:
            raise ValueError(f"Unknown model: {self.model}")

    def vec_parts(self, layer=None):
        return {
            'M': self.model,
            'D': self.dataset,
            'B': self.behavior,
            'R': 'res' if self.residual else None,
            'G': 'log' if self.logit else None,
            'L': self.layer if layer is None else layer,
            'S': 'scrub' if self.scrub else None,
        }

    def response_parts(self):
        return self.vec_parts() | {
            'T': self.temp,
            'C': self.leace,
            'K': self.toks,
        }

    def acts_path(self, positive, layer):
        parts = self.vec_parts(layer) | {
            'P': "+" if positive else "-",
        }
        return f"artifacts/acts/acts_{parts_to_suffix(parts)}.pt"

    def vec_path(self, layer):
        parts = self.vec_parts(layer)
        return f"artifacts/vecs/vec_{parts_to_suffix(parts)}.pt"

    def eraser_path(self, layer):
        parts = self.vec_parts(layer)
        return f"artifacts/vecs/eraser_{parts_to_suffix(parts)}.pt"

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
    parser.add_argument("--dataset", type=str, choices=["prompts", "caa", "openr", "ab", "opencon", "abcon"], default="prompts")
    parser.add_argument("--residual", action="store_true")
    parser.add_argument("--logit", action="store_true")
    parser.add_argument("--behavior", type=str, default=None, choices=ALL_BEHAVIORS)
    parser.add_argument("--model", type=str, default="llama3", choices=["llama3", "llama31", "llama2"])
    parser.add_argument("--scrub", action="store_true")

    if not generate:
        parser.add_argument("--layer", type=lambda x: int(x) if x != "all" else x, default=15)
        parser.add_argument("--temp", type=float, default=1.5)
        parser.add_argument("--leace", type=str, default=None, choices=["leace", "orth", "quad", "quadlda"])
        parser.add_argument("--toks", type=str, default=None, choices=["before", "after"])

    args = parser.parse_args()

    if generate:
        settings = Settings(
            model=args.model,
            dataset=args.dataset, 
            residual=args.residual,
            logit=args.logit,
            behavior=args.behavior,
            scrub=args.scrub,
        )
    else:
        settings = Settings(
            model=args.model,
            dataset=args.dataset, 
            residual=args.residual,
            logit=args.logit,
            behavior=args.behavior,
            scrub=args.scrub,
            layer=args.layer, 
            temp=args.temp,
            leace=args.leace,
            toks=args.toks,
        )

    return args, settings