import json
from tqdm import tqdm

from transformers import AutoTokenizer

from refusal_test_open_ended import eval_prompt, EVAL_MODEL

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(EVAL_MODEL)

    with open("/home/adam/src/CAA/datasets/generate/refusal/generate_openresponse_dataset.json") as f:
        data = json.load(f)

    for d in (data):
        print(d["question"])

        match_eval = eval_prompt(d["question"], d["answer_matching_behavior"], tokenizer)
        # if match_eval in [None, False]:
        #     print(d["answer_matching_behavior"])
        #     print("Match eval failed: ", match_eval)
        print("REFUSE" if match_eval else "NOTREFUSE", d["answer_matching_behavior"])

        unmatch_eval = eval_prompt(d["question"], d["answer_not_matching_behavior"], tokenizer)
        # if unmatch_eval in [None, True]:
        #     print(d["answer_not_matching_behavior"])
        #     print("Unmatch eval failed: ", unmatch_eval)
        print("REFUSE" if unmatch_eval else "NOTREFUSE", d["answer_not_matching_behavior"])

        print()