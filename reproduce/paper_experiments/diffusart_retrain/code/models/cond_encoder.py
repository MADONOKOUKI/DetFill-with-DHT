import math
from inspect import isfunction
from functools import partial

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from einops import rearrange

import torch
from torch import nn, einsum
import torch.nn.functional as F
from models.blocks import *

class cond_encod(nn.Module):
    def __init__(
            self,
            dim,
            init_dim=None,
            dim_mults=(1, 2, 4, 8),
            channels=3,
            with_time_emb=False,
            resnet_block_groups=8,
            use_convnext=False,
            convnext_mult=2,
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels

        init_dim = default(init_dim, dim // 2)
        self.init_conv = nn.Conv2d(channels, init_dim, 3, padding=1)

        self.init_down = nn.Conv2d(init_dim, init_dim, 3, padding=1, stride=2)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        if use_convnext:
            block_klass = partial(ConvNextBlock, mult=convnext_mult)
        else:
            block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        if with_time_emb:
            time_dim = dim * 4
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            time_dim = None
            self.time_mlp = None

        # layers
        self.downs = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.init_down (x)

        # t = self.time_mlp(time) if exists(self.time_mlp) else None

        h = []
        i = 0
        out = []
        # downsample
        for block1, block2, downsample in self.downs:
            i += 1
            x = block1(x)
            x = block2(x)
            h.append(x)
            x = downsample(x)
        x_out = self.mid_block2(x)
        out.append(h[i-2])
        out.append(x_out)
        return out

# f = Unet(dim = 128, channels= 3, dim_mults=(1, 2, 4,))
# print('ok')