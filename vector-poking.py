#%%
import torch
import os
import numpy as np

vecs = {}
for filename in os.listdir("artifacts/vecs"):
    if filename.startswith("vec_") and "L15" in filename and not "Dab_B" in filename:
        vec = torch.load(f"artifacts/vecs/{filename}")
        vecs[filename[13:-7]] = vec


# %%

#pairwise cosine similarity

def cos_sim(a, b):
    a, b = a.to(torch.float32).cpu(), b.to(torch.float32).cpu()
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

sims = {}

for key1 in vecs:
    for key2 in vecs:
        if key1 < key2:
            sims[(key1, key2)] = cos_sim(vecs[key1], vecs[key2])


# print ten largest and smallest similarities

sorted_sims = sorted(sims.items(), key=lambda x: x[1])

print("Largest similarities:")
for pair, sim in sorted_sims[-10:]:
    print(f"{sim}: {pair}")

print("\nSmallest similarities:")
for pair, sim in sorted_sims[:10]:
    print(f"{sim}: {pair}")

print()

print(f"{sims[('abcon', 'prompts')]}: abcon, prompts")