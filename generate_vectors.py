from dataclasses import dataclass, field
import pandas as pd
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import cached_property, get_layer_list, force_save
from common import Settings, parse_settings_args


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

    def record_activations(self, positive: bool):
        def hook(model, input, output):
            act = output[0]
            act = act.detach()[:, -1, :].cpu()
            if positive:
                self.positives.append(act)
            else:
                self.negatives.append(act)

        return hook

def get_prompts_caa(
    settings: Settings,
):
    df = pd.read_csv('caa_dataset.csv')

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

def get_prompts_ab(
    settings: Settings,
):
    with open('rimsky_refusal_generate_dataset.json') as f:
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
    if settings.dataset in ['prompts', 'caa']:
        return get_prompts_caa(settings)
    elif settings.dataset == 'ab':
        return get_prompts_ab(settings)
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

    layer_list = get_layer_list(model)

    adders = {mod: ActivationSaver() for mod in layer_list}

    # set p hooks
    p_handles = [mod.register_forward_hook(
        adders[mod].record_activations(positive=True)
    ) for mod in layer_list]
    
    # fwd passes
    with torch.no_grad():
        for p in tqdm(positives):
            input_ids = tokenize(p, tokenizer)
            input_ids = input_ids.to(model.device)
            model(input_ids)

    for handle in p_handles:
        handle.remove()

    # set n hooks
    n_handles = [mod.register_forward_hook(
        adders[mod].record_activations(positive=False)
    ) for mod in layer_list]

    # fwd passes
    with torch.no_grad():
        for n in tqdm(negatives):
            input_ids = tokenize(n, tokenizer, settings)
            input_ids = input_ids.to(model.device)
            model(input_ids)

    # save activations and steering vectors
    for layer, mod in enumerate(layer_list):
        pos = torch.stack(adders[mod].positives)
        neg = torch.stack(adders[mod].negatives)
        settings.layer = layer
        force_save(pos, settings.acts_path(positive=True))
        force_save(neg, settings.acts_path(positive=False))

        steering_vector = adders[mod].steering_vector
        force_save(steering_vector, settings.vec_path())
    
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    args, settings = parse_settings_args(parser, generate=True)
    generate_vectors(settings)