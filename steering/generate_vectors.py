from dataclasses import dataclass, field
import pandas as pd
from tqdm import tqdm
import json
import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from concept_erasure import LeaceFitter

from utils import cached_property, get_layer_list, force_save, get_residual_layer_list
from common import Settings, parse_settings_args


A_TOKEN = 4444  # '(A'
B_TOKEN = 5462  # '(B'

DATASET_ROOT = "datasets/"

@dataclass
class ActivationSaver:
    positives: list = field(default_factory=list)
    negatives: list = field(default_factory=list)

    @cached_property
    def positive_mean(self):
        p = torch.cat(self.positives).mean(0)
        return p

    @cached_property
    def negative_mean(self):
        n = torch.cat(self.negatives).mean(0)
        return n

    @cached_property
    def steering_vector(self):
        return self.positive_mean - self.negative_mean

    def logit_vector_and_eraser(self, logodds):
        x = torch.cat(self.positives).cuda().to(torch.float64)
        z = torch.cat(logodds).cuda().to(torch.float64)
        fitter = LeaceFitter.fit(x, z)
        # return sigma_xz / sigma_z
        return fitter.sigma_xz.squeeze(-1) / torch.std(z), fitter.eraser

    def record_activations(self, positive: bool):
        def hook(model, input, output):
            act = output[0]
            # if shape is [N, M], pad to [1, N, M]
            # I have made some poor choices in the past
            if len(act.shape) == 2:
                act = act.unsqueeze(0)
            act = act.detach()[:, -1, :].cpu()
            if positive:
                self.positives.append(act)
            else:
                self.negatives.append(act)

        return hook

def get_prompts_harm(
    settings: Settings,
):
    df = pd.read_csv(DATASET_ROOT+'harm_dataset.csv')

    positives = []
    negatives = []

    for index, row in df.iterrows():
        if settings.dataset == 'caa':
            p_messages = [
                {"role": "user", "content": row['prompt']},
                {"role": "assistant", "content": row['refusal']},
            ]
            n_messages = [
                {"role": "user", "content": row['prompt']},
                {"role": "assistant", "content": row['nonrefusal']},
            ]

            positives.append(p_messages)
            negatives.append(n_messages)
        elif settings.dataset == 'opencon':
            p_messages = [
                {"role": "user", "content": row['prompt']},
                {"role": "assistant", "content": "Sorry, I can't comply with that request. "},
            ]
            n_messages = [
                {"role": "user", "content": row['prompt']},
                {"role": "assistant", "content": "Sure, I can help with that. "},
            ]

            positives.append(p_messages)
            negatives.append(n_messages)
        elif settings.dataset == 'abcon':
            user_prompt = row['prompt'] + "\n\nChoices:\n(A) {a}\n(B) {b}"
            refusal = "Sorry, I can't comply with that request. "
            nonrefusal = "Sure, I can help with that. "

            messages = [
                {"role": "user", "content": user_prompt.format(a=refusal, b=nonrefusal)},
                {"role": "assistant", "content": "(A"},
            ]
            positives.append(messages)

            messages = [
                {"role": "user", "content": user_prompt.format(b=refusal, a=nonrefusal)},
                {"role": "assistant", "content": "(B"},
            ]
            positives.append(messages)

            messages = [
                {"role": "user", "content": user_prompt.format(a=refusal, b=nonrefusal)},
                {"role": "assistant", "content": "(B"},
            ]
            negatives.append(messages)

            messages = [
                {"role": "user", "content": user_prompt.format(b=refusal, a=nonrefusal)},
                {"role": "assistant", "content": "(A"},
            ]
            negatives.append(messages)
        elif settings.dataset == 'prompts':
            messages = [
                {"role": "user", "content": row['prompt']},
            ]

            if row['label'] == 'harmful':
                positives.append(messages)
            else:
                negatives.append(messages)
        else:
            raise ValueError(f"Unknown dataset: {settings.dataset}")

    return positives, negatives

def get_prompts_rimsky(
    settings: Settings,
):
    if settings.dataset == 'ab':
        with open(DATASET_ROOT+'CAA/generate/refusal/generate_dataset.json') as f:
            data = json.load(f)
    elif settings.dataset == 'openr':
        with open(DATASET_ROOT+'CAA/generate/refusal/generate_openresponse_dataset.json') as f:
            data = json.load(f)

    positives = []
    negatives = []

    for row in data:
        # remove last character which is a close-parenthesis
        messages = [
            {"role": "user", "content": row['question']},
            {"role": "assistant", "content": row['answer_matching_behavior'][:-1]},
        ]
        positives.append(messages)

        messages = [
            {"role": "user", "content": row['question']},
            {"role": "assistant", "content": row['answer_not_matching_behavior'][:-1]},
        ]
        negatives.append(messages)

    return positives, negatives

def get_prompts(
    settings: Settings,
):
    if settings.dataset in ['prompts', 'caa', 'opencon', 'abcon']:
        return get_prompts_harm(settings)
    elif settings.dataset in ['ab', 'openr']:
        return get_prompts_rimsky(settings)
    else:
        raise ValueError(f"Unknown dataset: {settings.dataset}")

def tokenize(messages, tokenizer):
    gen_prompt = (messages[-1]['role'] == 'user')
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=gen_prompt,
        return_tensors="pt",
    )
    if messages[-1]['role'] == 'assistant':
        input_ids = input_ids[:, :-1]  # remove eot token

    return input_ids

def generate_vectors(
    settings: Settings,
):
    # load model
    tokenizer = AutoTokenizer.from_pretrained(settings.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        settings.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # load positive and negative prompts
    positives, negatives = get_prompts(settings)

    layer_list = get_residual_layer_list(model) if settings.residual else get_layer_list(model)

    adders = {mod: ActivationSaver() for mod in layer_list}

    if settings.logit:
        logodds = []

    # set p hooks
    p_handles = [mod.register_forward_hook(
        adders[mod].record_activations(positive=True)
    ) for mod in layer_list]
    
    # fwd passes
    with torch.no_grad():
        for p in tqdm(positives):
            input_ids = tokenize(p, tokenizer)
            input_ids = input_ids.to(model.device)
            
            if settings.logit:
                last_tok = input_ids[:, -1]
                assert last_tok == A_TOKEN or last_tok == B_TOKEN
                match_tok, nonmatch_tok = (A_TOKEN, B_TOKEN) if last_tok == A_TOKEN else (B_TOKEN, A_TOKEN)

                output = model(input_ids[:, :-1])

                logodd = (output.logits[:, -1, match_tok] - output.logits[:, -1, nonmatch_tok]).cpu()
                logodds.append(logodd)
                
            else:
                model(input_ids)

    for handle in p_handles:
        handle.remove()

    if settings.logit:
        print("Saving activations and steering vectors...")

        # save activations and steering vectors
        for layer, mod in tqdm(enumerate(layer_list)):
            pos = torch.stack(adders[mod].positives)
            force_save(pos, settings.acts_path(positive=True, layer=layer))

            steering_vector, eraser = adders[mod].logit_vector_and_eraser(logodds)
            force_save(steering_vector, settings.vec_path(layer=layer))
            force_save(eraser, settings.eraser_path(layer=layer))
        
        print("Done!")
        return


    # set n hooks
    n_handles = [mod.register_forward_hook(
        adders[mod].record_activations(positive=False)
    ) for mod in layer_list]

    # fwd passes
    with torch.no_grad():
        for n in tqdm(negatives):
            input_ids = tokenize(n, tokenizer)
            input_ids = input_ids.to(model.device)
            model(input_ids)

    for handle in n_handles:
        handle.remove()

    print("Saving activations and steering vectors...")

    # save activations and steering vectors
    for layer, mod in tqdm(enumerate(layer_list)):
        pos = torch.stack(adders[mod].positives)
        neg = torch.stack(adders[mod].negatives)
        force_save(pos, settings.acts_path(positive=True, layer=layer))
        force_save(neg, settings.acts_path(positive=False, layer=layer))

        steering_vector = adders[mod].steering_vector
        force_save(steering_vector, settings.vec_path(layer=layer))
    
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    args, settings = parse_settings_args(parser, generate=True)
    generate_vectors(settings)