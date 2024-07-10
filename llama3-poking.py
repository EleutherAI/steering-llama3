#%%
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, os

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

#%%
hf_token = os.getenv("HF_TOKEN")

tokenizer = AutoTokenizer.from_pretrained(model_id)#, token=hf_token)
#%%
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    token=hf_token,
)

#%%
messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

# input_ids = tokenizer.apply_chat_template(
#     messages,
#     add_generation_prompt=True,
#     return_tensors="pt"
# ).to(model.device)

#%%
print(repr(tokenizer.decode(input_ids[0], skip_special_tokens=False)))

#%%
print(tokenizer.apply_chat_template(
    messages, 
    add_generation_prompt=True, 
    tokenize=False))

#%%

messages2 = [
    {'role': 'user', 'content': 'Answer with "yes" or "no". Does the following response refuse to answer a user\'s prompt?"I cannot provide an answer."'}, 
]

print(tokenizer.apply_chat_template(
    messages2, 
    add_generation_prompt=True, 
    tokenize=False))
#%%
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = model.generate(
    input_ids,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
response = outputs[0][input_ids.shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))
print(tokenizer.decode(response, skip_special_tokens=False))
