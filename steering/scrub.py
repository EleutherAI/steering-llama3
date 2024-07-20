# %%
from types import MethodType
from typing import Callable, ContextManager
from contextlib import contextmanager
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from datasets import Dataset
from tqdm.auto import tqdm
from transformers import (
    LlamaForCausalLM,
    LlamaModel,
)
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    apply_rotary_pos_emb,
    LlamaSdpaAttention,
    LlamaMLP,
)
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset, ClassLabel, Sequence

from concept_erasure import ErasureMethod, LeaceFitter, LeaceEraser
from concept_erasure.utils import assert_type, is_norm_layer, mangle_module_path

from generate_vectors import *


RESIDUAL_CLASSES = (LlamaSdpaAttention, LlamaMLP)

class CAAScrubber:
    """Wrapper for a dictionary mapping module paths to `LeaceEraser` objects and CAA vectors."""

    def __init__(self, pre_hook: bool = False):
        super().__init__()

        self.erasers: dict[str, LeaceEraser] = {}
        self.vectors: dict[str, torch.Tensor] = {}  # Includes multiplier!
        self.pre_hook = pre_hook

    @contextmanager
    def scrub(self, model):
        """Add hooks to the model which apply the erasers during a forward pass."""

        def scrub_hook(key: str, x: Tensor):
            eraser = assert_type(LeaceEraser, self.erasers[key])
            return eraser(x).type_as(x) + self.vectors[key]

        with self.apply_hook(model, scrub_hook):
            yield self

    @contextmanager
    def apply_hook(
        self,
        model: nn.Module,
        hook_fn: Callable[[str, Tensor], Tensor | None],
    ):
        """Apply a `hook_fn` to each submodule in `model` that we're scrubbing."""

        def post_wrapper(_, __, output, name: str) -> Tensor | None:
            key = mangle_module_path(name)
            return hook_fn(key, output)

        def pre_wrapper(_, inputs, name: str) -> tuple[Tensor | None, ...]:
            x, *extras = inputs
            key = mangle_module_path(name)
            return hook_fn(key, x), *extras

        # Unwrap the base model if necessary. This is needed to ensure we don't try to
        # scrub right before the unembedding layer
        from transformers import PreTrainedModel

        if isinstance(model, PreTrainedModel):
            model = assert_type(PreTrainedModel, model.base_model)

        handles = [
            (
                mod.register_forward_pre_hook(partial(pre_wrapper, name=name))
                if self.pre_hook
                else mod.register_forward_hook(partial(post_wrapper, name=name))
            )
            for name, mod in model.named_modules()
            if type(mod) in RESIDUAL_CLASSES and mangle_module_path(name) in self.erasers
        ]
        assert len(handles) == len(self.erasers), "Not all erasers can be applied"

        try:
            yield self
        finally:
            # Make sure to remove the hooks even if an exception is raised
            for handle in handles:
                handle.remove()


@torch.no_grad()
def scrub_llama(
    model: LlamaForCausalLM,
    train: Dataset,
    multiplier: float,
    z_column: str | None,
    batch_size: int = 1,
    method: ErasureMethod = "leace",
    sublayers: bool = True,
    affine: bool = True,
) -> tuple[CAAScrubber | None, float]:
    base = assert_type(LlamaModel, model.base_model)
    d = assert_type(int, base.config.hidden_size)

    if z_column is None:
        k = -1
        scrubber = None
    else:
        k = assert_type(int, train.features[z_column].feature.num_classes)
        scrubber = CAAScrubber()

    losses = []

    xs = []
    zs = []
    N = len(train) // batch_size
    train = train.with_format("torch", device=model.device)

    # Embed all the inputs
    embed_fn = base.get_input_embeddings()
    for i, batch in enumerate(tqdm(train.iter(batch_size), desc="Embedding", total=N)):
        assert isinstance(batch, dict)

        tokens = assert_type(torch.Tensor, batch["input_ids"])
        x = embed_fn(tokens)
        xs.append(x.to("cpu", non_blocking=True))

        # We don't actually need to move these to the CPU since they're small
        if z_column is not None:
            zs.append(F.one_hot(batch[z_column], num_classes=k))

    # Enumerate the layers
    for j, layer in enumerate(tqdm(base.layers, unit="layer")):
        assert isinstance(layer, LlamaDecoderLayer)

        attn_eraser = None
        if scrubber is not None:
            attn_fitter = LeaceFitter(
                d, k, affine=affine, device=model.device, method=method
            )

            class_sums = torch.zeros(k, d, device=model.device)

            # Fit the next eraser and vector on the previous hidden states
            for x, z in tqdm(zip(xs, zs), desc="Fitting (attn)", total=N):
                assert scrubber is not None

                # Discard post-LN output and recompute during application to save RAM
                x = layer.input_layernorm(x.to(model.device))
                attn_fitter.update(x[..., -1, :], z)
                # print(x.shape, z.shape, x.dtype, z.dtype)
                class_sums += z.squeeze(1).type_as(x).T @ x[..., -1, :]

            class_means = class_sums / N
            attn_vector = multiplier * (class_means[1] - class_means[0]).type_as(x)
            scrubber.vectors[f"layers-{j}-input_layernorm"] = attn_vector

            attn_eraser = attn_fitter.eraser
            scrubber.erasers[f"layers-{j}-input_layernorm"] = attn_eraser
            del attn_fitter  # Save VRAM

        # Run attention & MLP with the erasers we just fit
        for i, x in tqdm(enumerate(xs), desc="Applying (attn)", total=N):
            # Bring back to the accelerator
            x = x.to(model.device)
            h = layer.input_layernorm(x)  # Recomputing from above

            # Apply the eraser
            if attn_eraser is not None and scrubber is not None:
                h = attn_eraser(h).type_as(h) + attn_vector

            pos_ids = torch.arange(0, h.shape[-2], device=h.device, dtype=torch.long)
            pos_ids = pos_ids.unsqueeze(0).view(-1, h.shape[-2])

            h, _, __ = layer.self_attn(h, position_ids=pos_ids)
            h = x = x + h  # Post-attention residual connection

            # We're not scrubbing the sublayers, so run the rest of the layer
            if not sublayers:
                h = layer.post_attention_layernorm(h)
                h = layer.mlp(h)
                h = x + h  # Post-MLP residual connection

            xs[i] = h.to("cpu", non_blocking=True)

        # Skip the part where we scrub the MLP if we're not doing that
        if not sublayers:
            continue

        mlp_eraser = None
        if scrubber is not None:
            mlp_fitter = LeaceFitter(
                d, k, affine=affine, device=model.device, method=method
            )

            class_sums = torch.zeros(k, d, device=model.device)

            # Fit the next eraser on the previous hidden states
            for x, z in tqdm(zip(xs, zs), desc="Fitting (MLP)", total=N):
                assert scrubber is not None

                # Discard post-LN output and recompute during application to save RAM
                x = layer.post_attention_layernorm(x.to(model.device))
                mlp_fitter.update(x[..., -1, :], z)
                # print(x.shape, z.shape, x.dtype, z.dtype)
                class_sums += z.squeeze(1).type_as(x).T @ x[..., -1, :]

            class_means = class_sums / N
            mlp_vector = multiplier * (class_means[1] - class_means[0]).type_as(x)
            scrubber.vectors[f"layers-{j}-post_attention_layernorm"] = mlp_vector

            mlp_eraser = mlp_fitter.eraser
            scrubber.erasers[f"layers-{j}-post_attention_layernorm"] = mlp_eraser
            del mlp_fitter  # Save VRAM

        for i, x in tqdm(enumerate(xs), desc="Applying (MLP)", total=N):
            # Bring back to the accelerator
            x = x.to(model.device)
            h = layer.post_attention_layernorm(x)  # Recomputing from above

            # Apply the eraser
            if mlp_eraser is not None and scrubber is not None:
                h = mlp_eraser(h).type_as(h) + mlp_vector

            h = layer.mlp(h)
            h = x + h  # Post-MLP residual connection
            xs[i] = h.to("cpu", non_blocking=True)

    # for batch, x in tqdm(zip(train.iter(batch_size), xs), total=N):
    #     assert isinstance(batch, dict)
    #     tokens = assert_type(torch.Tensor, batch["input_ids"])

    #     x = x.to(model.device)
    #     x = base.norm(x)
    #     logits = model.lm_head(x)

    #     labels = tokens.to(logits.device)
    #     shift_logits = logits[:, :-1, :].contiguous()
    #     labels = labels[:, 1:].contiguous()
    #     lm_loss = F.cross_entropy(
    #         shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1)
    #     )
    #     losses.append(lm_loss)

    return scrubber  #, torch.stack(losses).mean().item()


def scrub(
    settings: Settings
):
    tokenizer = AutoTokenizer.from_pretrained(settings.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        settings.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # load positive and negative prompts
    positives, negatives = get_prompts(settings)

    positive_ids = [tokenize(p, tokenizer)[0] for p in positives]
    negative_ids = [tokenize(n, tokenizer)[0] for n in negatives]

    # Create the dataset
    ds = Dataset.from_dict({
        "input_ids": positive_ids + negative_ids,
        "label": [[1]] * len(positive_ids) + [[0]] * len(negative_ids)
    })

    # Define the features explicitly
    ds = ds.cast_column("label", Sequence(ClassLabel(num_classes=2, names=["negative", "positive"])))
    
    scrubber = scrub_llama(
        model,
        ds,
        multiplier=1.0,
        z_column="label",
        batch_size=1,
        method="leace",
        sublayers=True,
        affine=True,
    )

    with scrubber.scrub(model):
        pass

    # layer_list = get_residual_layer_list(model) if settings.residual else get_layer_list(model)

# %%

scrub(Settings())