#%%
#import os

from transformers import AutoModelForCausalLM, AutoTokenizer

#%%
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

#%%
model.model.layers
# %%
model.model.layers[3]