from dataclasses import asdict, dataclass, field
from typing import Optional

from utils import cached_property


@dataclass
class Settings:
    dataset: str = "prompts"

    @cached_property
    def save_suffix(self):
        parts = {
            'D': self.dataset,
        }

        return '_'.join(f"{k}{v}" for k, v in parts.items() if v is not None)

    def acts_path(self):
        return f"artifacts/acts/acts_{self.save_suffix}.pt"

    def vec_path(self):
        return f"artifacts/vecs/vec_{self.save_suffix}.pt"