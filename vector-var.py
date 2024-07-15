# %%
import torch

from steering.common import Settings

# %%
settings = Settings()
# %%
settings.layer = 15
pos = torch.load(settings.acts_path(positive=True)).cuda().squeeze(1)
neg = torch.load(settings.acts_path(positive=False)).cuda().squeeze(1)
# %%
diffs = pos - neg

vec = diffs.mean(dim=0)

res = diffs - vec

vec.norm(), res.norm(dim=1).mean()
# %%
res[:16].mean(dim=0).norm()
# %%
import numpy as np
res.norm(dim=1).mean() / vec.norm() * 1 / np.sqrt(32)

# %%
settings = Settings(dataset="ab")
# %%
settings.layer = 15
pos = torch.load(settings.acts_path(positive=True)).cuda().squeeze(1)
neg = torch.load(settings.acts_path(positive=False)).cuda().squeeze(1)
# %%
diffs = pos - neg

vec = diffs.mean(dim=0)

res = diffs - vec

vec.norm(), res.norm(dim=1).mean(), res.norm(dim=1).mean() / vec.norm()
# %%
settings = Settings(dataset="caa")
# %%
settings.layer = 15
pos = torch.load(settings.acts_path(positive=True)).cuda().squeeze(1)
neg = torch.load(settings.acts_path(positive=False)).cuda().squeeze(1)
# %%
diffs = pos - neg

vec = diffs.mean(dim=0)

res = diffs - vec

vec.norm(), res.norm(dim=1).mean(), res.norm(dim=1).mean() / vec.norm()
# %%
