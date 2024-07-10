import pandas as pd
import io
import requests

from datasets import load_dataset


def get_harmful_gen_prompts(num=100):
    url = 'https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv'
    response = requests.get(url)

    dataset = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
    harmful_prompts = dataset['goal'].tolist()
    
    subset_prompts = harmful_prompts[:num]
    for i in range(len(subset_prompts)):
        subset_prompts[i] += '.'

    return subset_prompts


def get_harmless_gen_prompts(num=100):
    hf_path = 'tatsu-lab/alpaca'
    dataset = load_dataset(hf_path)

    harmless_prompts = []
    for i in range(len(dataset['train'])):
        if dataset['train'][i]['input'].strip() == '':
            harmless_prompts.append(dataset['train'][i]['instruction'])

    return harmless_prompts[:num]


def make_prompts_csv(num):
    harmful_prompts = get_harmful_gen_prompts(num)
    harmless_prompts = get_harmless_gen_prompts(num)

    df = pd.DataFrame({
        'prompt': harmful_prompts + harmless_prompts,
        'label': ['harmful'] * len(harmful_prompts) + ['harmless'] * len(harmless_prompts),
    })

    df.to_csv('caa_prompts.csv', index=False)


if __name__ == "__main__":
    make_prompts_csv(32)