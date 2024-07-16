import json
from tqdm import tqdm

from transformers import AutoTokenizer

from steering.caa_test_open_ended import eval_prompt, EVAL_MODEL, ALL_BEHAVIORS, SYCOPHANCY, REFUSAL

if __name__ == "__main__":
    behaviors = ALL_BEHAVIORS

    tokenizer = AutoTokenizer.from_pretrained(EVAL_MODEL)

    for behavior in tqdm(behaviors):

        with open(f"datasets/CAA/generate/{behavior}/generate_openresponse_dataset.json") as f:
            data = json.load(f)

        for d in data[:10]:
            print(d["question"])

            match_eval = eval_prompt(behavior, d["question"], d["answer_matching_behavior"], tokenizer)
            # if match_eval in [None, False]:
            #     print(d["answer_matching_behavior"])
            #     print("Match eval failed: ", match_eval)
            print(match_eval, d["answer_matching_behavior"])

            unmatch_eval = eval_prompt(behavior, d["question"], d["answer_not_matching_behavior"], tokenizer)
            # if unmatch_eval in [None, True]:
            #     print(d["answer_not_matching_behavior"])
            #     print("Unmatch eval failed: ", unmatch_eval)
            print(unmatch_eval, d["answer_not_matching_behavior"])

            print()