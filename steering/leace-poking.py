# %%
import torch
import os
import sys



from common import Settings
from concept_erasure import LeaceEraser
# %%

settings = Settings()

layer = 15

# change pwd to ..
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Fitting eraser...")
# [N, 1, D] -> [N, D]
pos = torch.load(settings.acts_path(positive=True, layer=layer)).cuda().squeeze(1)
neg = torch.load(settings.acts_path(positive=False, layer=layer)).cuda().squeeze(1)
# [2N, D]
x = torch.cat([pos, neg]).to(torch.float64)
z = torch.cat([torch.ones(len(pos)), torch.zeros(len(neg))]).cuda().to(torch.float64)

eraser = LeaceEraser.fit(x, z, method="leace")
print("Eraser fitted!")

# %%
print(eraser.proj_left.shape, eraser.proj_right.shape)

torch.arange(5).reshape([1, 5]).mH.shape