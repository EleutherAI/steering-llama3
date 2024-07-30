from dataclasses import dataclass
import numpy as np
import json
from tqdm import tqdm
from argparse import ArgumentParser
import os
from typing import Optional, Union

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from concept_erasure import LeaceEraser, QuadraticEditor, QuadraticFitter, LeaceFitter

from common import Settings, parse_settings_args
from refusal_test_open_ended import get_harmful_test_prompts, get_harmless_test_prompts
from utils import get_layer_list, force_save, cached_property, get_residual_layer_list
from generate_vectors import DATASET_ROOT
from scrub import get_scrubber, CAAScrubber, mangle_module_path

from torch.profiler import profile, record_function, ProfilerActivity


@dataclass
class ActivationSteerer:
    vector: torch.Tensor
    settings: Settings
    eraser: Optional[Union[LeaceEraser, QuadraticEditor]] = None
    threshold: Optional[float] = None
    class_vector: Optional[torch.Tensor] = None
    start: Optional[int] = None
    end: Optional[int] = None
    trivial: bool = False

    def steer_activations(self, multiplier: float, layer: int = None):
        if self.trivial:
            return lambda model, input, output: output

        if self.settings.scrub:
            u = self.vector.cuda()
        else:
            u = multiplier * self.vector.cuda()

        if self.settings.leace == "quad":
            target = 2 if multiplier == 0 else int(multiplier > 0)

            def hook(model, input, output):
                output[0][..., self.start:self.end, :] = self.eraser.transport(
                    output[0][..., self.start:self.end, :].to(torch.float64), 
                    2, 
                    target
                ).to(output[0].dtype)
                return output

            return hook

        if self.settings.leace == "quadlda":
            if multiplier == 0:
                return lambda model, input, output: output

            # get source classes with class_vector and threshold
            sources = (output[0] @ self.class_vector > self.threshold).to(int)
            target = int(multiplier > 0)

            def hook(model, input, output):
                output[0][..., self.start:self.end, :] = self.eraser(
                    output[0][..., self.start:self.end, :].to(torch.float64), 
                    sources, 
                    target
                ).to(output[0].dtype)
                return output

            return hook

        if self.eraser is not None:
            etype = self.eraser.proj_left.dtype
            def hook(model, input, output):
                if TEST:
                    output_copy = output[0].clone().detach()
                output[0][..., self.start:self.end, :] = self.eraser(output[0][..., self.start:self.end, :].to(etype)).to(output[0].dtype)
                if TEST:
                    print()
                    if layer is not None:
                        print(f"Layer {layer}")
                    print("DIFF:", (output_copy - output[0]).norm())
                    print("COPY:", output_copy.norm())
                    print("NEW:", output[0].norm())
                    print()
                output[0][..., self.start:self.end, :] += u
                # if TEST:
                #     print(self.eraser.proj_left.shape, u.shape)
                #     print(u.to(torch.float64) @ self.eraser.proj_left.to(torch.float64) / u.norm() / self.eraser.proj_left.norm())
                #     print(u.norm(), self.eraser.proj_left.norm())
                return output

            return hook

        # print("u", u.device, u.dtype, u.shape)

        def hook(model, input, output):
            # with record_function("steer_hook"):
            # with record_function("steer"):
            # print(type(model.mlp), type(model.self_attn))
            # print("output", output[0].device, output[0].dtype, output[0].shape)
            output[0][..., self.start:self.end, :] += u
            return output
            
        return hook


def load_scrub_steerers(settings: Settings, model, layer_list, mult):
    scrubber = get_scrubber(model, settings, mult)

    layer_names = {}
    for name, mod in model.named_modules():
        if mod in layer_list:
            layer_names[mod] = mangle_module_path(name).split('model-')[1]

    steerers = {}
    for i, layer in enumerate(layer_list):
        steerer = ActivationSteerer(
            scrubber.vectors[layer_names[layer]],
            settings,
            eraser=scrubber.erasers[layer_names[layer]],
        )
        steerers[i] = steerer

    return steerers


def load_steerer(settings: Settings, layer: int, trivial: bool = False):
    path = settings.vec_path(layer)
    vec = torch.load(path).cpu()

    if settings.leace == "quad":
        if settings.logit:
            raise ValueError("Logit steering not supported with quadratic eraser")
        print("Fitting quadratic editor...")
        # [N, 1, D] -> [N, D]
        pos = torch.load(settings.acts_path(positive=True, layer=layer)).cuda().squeeze(1).to(torch.float64)
        neg = torch.load(settings.acts_path(positive=False, layer=layer)).cuda().squeeze(1).to(torch.float64)

        fitter = QuadraticFitter(pos.shape[-1], 3, dtype=torch.float64, device=pos.device)
        fitter.update_single(pos, 2)
        fitter.update_single(neg, 2)
        fitter.update_single(pos, 1)
        fitter.update_single(neg, 0)

        editor = fitter.editor()
        print(f"Editor fitted for layer {layer}!")

        return ActivationSteerer(vec, settings, editor, trivial=trivial)

    if settings.leace == "quadlda":
        print("Fitting LEACE eraser for LDA vector...")
        # [N, 1, D] -> [N, D]
        pos = torch.load(settings.acts_path(positive=True, layer=layer)).cuda().squeeze(1)
        neg = torch.load(settings.acts_path(positive=False, layer=layer)).cuda().squeeze(1)
        # [2N, D]
        x = torch.cat([pos, neg]).to(torch.float64)
        z = torch.cat([torch.ones(len(pos)), torch.zeros(len(neg))]).cuda().to(torch.float64)

        eraser = LeaceEraser.fit(x, z, method=settings.leace)
        print(f"Eraser fitted for layer {layer}!")

        lda = eraser.proj_right.squeeze(0)
        act_mean = x.mean(dim=0)
        threshold = torch.dot(lda, act_mean)

        print("Fitting QLEACE editor...")
        # [N, 1, D] -> [N, D]
        pos = torch.load(settings.acts_path(positive=True, layer=layer)).cuda().squeeze(1).to(torch.float64)
        neg = torch.load(settings.acts_path(positive=False, layer=layer)).cuda().squeeze(1).to(torch.float64)

        fitter = QuadraticFitter(pos.shape[-1], 2, dtype=torch.float64, device=pos.device)
        fitter.update_single(pos, 1)
        fitter.update_single(neg, 0)

        editor = fitter.editor()
        print(f"Editor fitted for layer {layer}!")

        return ActivationSteerer(vec, settings, editor, threshold, lda, trivial=trivial)

    if settings.leace:
        if settings.logit:
            print(f"Loading logit eraser for layer {layer}...")
            eraser = torch.load(settings.eraser_path(layer))
            print(f"Eraser loaded for layer {layer}!")
        else:
            print(f"Fitting eraser for layer {layer}...")
            # [N, 1, D] -> [N, D]
            pos = torch.load(settings.acts_path(positive=True, layer=layer)).cuda().squeeze(1)
            neg = torch.load(settings.acts_path(positive=False, layer=layer)).cuda().squeeze(1)
            # [2N, D]
            x = torch.cat([pos, neg]).to(torch.float64)
            z = torch.cat([torch.ones(len(pos)), torch.zeros(len(neg))]).cuda().to(torch.float64)

            if TEST:
                fitter = LeaceFitter.fit(x, z, method=settings.leace)
                eraser = fitter.eraser

                if settings.leace == "leace":
                    sigma = fitter.sigma_xx
                    L, V = torch.linalg.eigh(sigma)
                    print("L", L)
            else:
                eraser = LeaceEraser.fit(x, z, method=settings.leace)
            print(f"Eraser fitted for layer {layer}!")

            if TEST:
                print(eraser.proj_left.shape, eraser.proj_right.shape)
                print(eraser.proj_right @ eraser.proj_left)
                print(eraser.proj_right.norm(), eraser.proj_left.norm())
                print(eraser.proj_right.norm() * eraser.proj_left.norm())
                # cosine sim
                print("cosine sim", eraser.proj_right @ eraser.proj_left / eraser.proj_right.norm() / eraser.proj_left.norm())
                print()

        return ActivationSteerer(vec, settings, eraser, trivial=trivial)

    return ActivationSteerer(vec, settings, trivial=trivial)


def get_prompts(settings: Settings, num):
    if settings.behavior is not None:
        with open(DATASET_ROOT + f"CAA/test/{settings.behavior}/test_dataset_open_ended.json") as f:
            questions = json.load(f)
        
        prompts = [{"prompt": q["question"]} for q in questions]

        return prompts[:num]

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
    overwrite: bool = False,
    trivial: bool = False,
    bias: bool = False,
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

    prompts = get_prompts(settings, num_prompts)

    layer_list = get_residual_layer_list(model) if settings.residual else get_layer_list(model)

    if settings.scrub:
        pass
    elif layer == "all":
        steerers = {i: load_steerer(settings, i, trivial) for i in range(len(layer_list))}
    else:
        steerers = {layer: load_steerer(settings, layer, trivial)}

    # if bias:
    #     # for i in steerers.keys():
    #     #     print(i, layer_list[i].mlp.down_proj.bias, layer_list[i].mlp.config.mlp_bias)
    #     orig_biases = {i: layer_list[i].mlp.down_proj.bias.data.clone() for i in steerers.keys()}

    for mult in mults:
        print(f"Steering activations by {mult}")

        if os.path.exists(settings.response_path(mult)) and not overwrite:
            print(f"Responses already saved to {settings.response_path(mult)}. Skipping...")
            continue

        if settings.scrub:
            steerers = load_scrub_steerers(settings, model, layer_list, mult)
        
        if bias:
            assert settings.leace is None
            assert settings.toks is None

            if settings.residual:
                raise NotImplementedError("Bias mode not implemented for residual layers")

            for i, steerer in steerers.items():
                layer_list[i].mlp.down_proj.bias = torch.nn.Parameter(mult * steerer.vector.cuda())
        else:
            handles = {
                i: layer_list[i].register_forward_hook(steerer.steer_activations(mult, i))
                for i, steerer in steerers.items()
            }

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

                    if settings.toks == "before":
                        for steerer in steerers.values():
                            steerer.start = None
                            steerer.end = input_ids.shape[-1] - 1
                    elif settings.toks == "after":
                        for steerer in steerers.values():
                            steerer.start = input_ids.shape[-1] - 1
                            steerer.end = None

                    # with record_function("model_generate"):
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

        if not bias:
            for handle in handles.values():
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
    
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--prompts", type=int, default=25)

    parser.add_argument("--trivial", action="store_true")
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--test", action="store_true")

    args, settings = parse_settings_args(parser)

    TEST = args.test

    steer(
        settings, 
        args.mults, 
        repeats=args.repeats, 
        num_prompts=args.prompts, 
        overwrite=args.overwrite,
        trivial=args.trivial,
        bias=args.bias,
    )