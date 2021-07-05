# %%
import os
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import torch.utils.data as data
import torch
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
import argparse
import json
#%%
from models.msvit import MsViT

#%%
model = MsViT(
    arch = 'l1,h3,d96,n1,s1,g1,p4,f7,a0_l2,h3,d192,n2,s1,g1,p2,f7,a0_l3,h6,d384,n8,s0,g1,p2,f7,a0_l4,h12,d768,n1,s0,g0,p2,f7,a0',
    img_size=64,
    in_chans=3,
    num_classes=10000
)
# %%
img = torch.rand(32, 3, 64, 64)
# %%
out = model(img)
# %%
print(out.shape)
# %%
