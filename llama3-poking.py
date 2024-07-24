#%%
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, os

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
model_id2 = "meta-llama/Meta-Llama-3.1-8B-Instruct"

#%%
hf_token = os.getenv("HF_TOKEN")

tokenizer = AutoTokenizer.from_pretrained(model_id)#, token=hf_token)
tokenizer2 = AutoTokenizer.from_pretrained(model_id2)#, token=hf_token)
# %%
print(type(tokenizer))
print(type(tokenizer2))

# print info about tokenizer model
print((tokenizer2.name_or_path))
#%%

print(
    tokenizer2.decode(tokenizer2.apply_chat_template(
    [{"role": "assistant", "content": "(A"}],
    add_generation_prompt=False,
)))

print(
tokenizer.decode(tokenizer.apply_chat_template(
    [{"role": "assistant", "content": "(A"}],
    add_generation_prompt=False,
)))
# %%
# without decode
print((tokenizer2.apply_chat_template(
    [{"role": "assistant", "content": "(A"}],
    add_generation_prompt=False,
        return_tensors="pt",
))[0].tolist())
print(
tokenizer.apply_chat_template(
    [{"role": "assistant", "content": "(A"}],
    add_generation_prompt=False,
))
#%%
print(tokenizer("(A"))
print(tokenizer2("(A"))
print(tokenizer("(B"))
print(tokenizer2("(B"))
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
