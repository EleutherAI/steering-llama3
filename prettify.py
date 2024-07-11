# read every .json in artifacts/responses/,
# then write again with indent=2

import json
import os

def prettify_responses():
    responses_dir = "artifacts/responses"
    for filename in os.listdir(responses_dir):
        if filename.endswith(".json"):
            with open(os.path.join(responses_dir, filename)) as f:
                responses = json.load(f)
            with open(os.path.join(responses_dir, filename.replace('.json', '__pretty.json')), "w") as f:
                json.dump(responses, f, indent=2)


if __name__ == "__main__":
    prettify_responses()
    print("Done!")