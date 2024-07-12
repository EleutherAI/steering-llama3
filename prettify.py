

import json
import os

def prettify_responses():
    # read every .json in artifacts/responses/,
    # then write again with indent=2

    responses_dir = "artifacts/responses"
    for filename in os.listdir(responses_dir):
        if filename.endswith(".json"):
            with open(os.path.join(responses_dir, filename)) as f:
                responses = json.load(f)
            with open(os.path.join(responses_dir, filename.replace('.json', '__pretty.json')), "w") as f:
                json.dump(responses, f, indent=2)

def remove_duplicates():
    # for each file in artifacts/responses/,
    # check if the __pretty file is identical to the original

    responses_dir = "artifacts/responses"
    for filename in os.listdir(responses_dir):
        if filename.endswith(".json") and not filename.endswith("__pretty.json"):
            # if no __pretty file, skip
            if not os.path.exists(os.path.join(responses_dir, filename.replace('.json', '__pretty.json'))):
                continue
            with open(os.path.join(responses_dir, filename)) as f:
                # don't use json, we want to compare the raw text
                original = f.read()
            with open(os.path.join(responses_dir, filename.replace('.json', '__pretty.json'))) as f:
                pretty = f.read()
            if original == pretty:
                print(f"Removing {filename.replace('.json', '__pretty.json')}")
                os.remove(os.path.join(responses_dir, filename.replace('.json', '__pretty.json')))

if __name__ == "__main__":
    #prettify_responses()
    remove_duplicates()
    print("Done!")