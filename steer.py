from dataclasses import dataclass
import numpy as np
import json
from tqdm import tqdm
from argparse import ArgumentParser
import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from common import Settings, parse_settings_args
from refusal_test_open_ended import get_harmful_test_prompts, get_harmless_test_prompts
from utils import get_layer_list, force_save


@dataclass
class ActivationSteerer:
    vector: torch.Tensor

    def steer_activations(self, multiplier: float):
        u = self.vector

        def hook(model, input, output):
            u_ = u.to(output[0].device)

            output[0][:, :, :] += multiplier * u_
            return output
            
        return hook


def load_steerer(settings: Settings, layer: int):
    path = settings.vec_path(layer)
    vec = torch.load(f"artifacts/vecs/{path}").cpu()
    steerer = ActivationSteerer(vec)
    return steerer

def get_prompts(settings: Settings, num):
    harmful_prompts = get_harmful_test_prompts(num)
    harmless_prompts = get_harmless_test_prompts(num)
    
    harmful = [
        {
            "label": "harmful",
            "prompt": p,
        } for p in harmful_prompts
    ]
    harmless = [
        {
            "label": "harmless",
            "prompt": p,
        } for p in harmless_prompts
    ]

    return harmful + harmless

def steer(
    settings: Settings, 
    mults: list[float], 
    repeats: int = 20,
    num_prompts: int = 25,
    ):
    token = os.getenv("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(settings.model_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        settings.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=token,
    )

    layer = settings.layer

    steerer = load_steerer(settings, layer)

    prompts = get_prompts(settings, num_prompts)

    layer_list = get_layer_list(model)

    for mult in mults:
        print(f"Steering activations by {mult}")

        if os.path.exists(settings.response_path(mult)):
            print(f"Responses already saved to {settings.response_path(mult)}. Skipping...")
            continue

        handle = layer_list[layer].register_forward_hook(steerer.steer_activations(mult))

        with torch.no_grad():
            for prompt in tqdm(prompts):
                responses = []

                for _ in range(repeats):
                    input_ids = tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt["prompt"]}], 
                        add_generation_prompt=True, 
                        return_tensors="pt",
                    ).to(model.device)

                    terminators = [
                        tokenizer.eos_token_id,
                        tokenizer.convert_tokens_to_ids("<|eot_id|>")
                    ]

                    outputs = model.generate(
                        input_ids,
                        max_new_tokens=50,
                        eos_token_id=terminators,
                        do_sample=True,
                        temperature=settings.temp,
                        use_cache=True,
                        #top_p=0.9,
                    )
                    response = outputs[0][input_ids.shape[-1]:]

                    responses.append(tokenizer.decode(response, skip_special_tokens=True))
                
                prompt["responses"] = responses

        handle.remove()

        force_save(
            prompts,
            settings.response_path(mult),
            mode="json"
        )

        print(f"Responses saved to {settings.response_path(mult)}. Done!")


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--mults", type=float, nargs="+", required=True)

    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--prompts", type=int, default=25)

    args, settings = parse_settings_args(parser)

    steer(settings, args.mults, repeats=args.repeats, num_prompts=args.prompts)