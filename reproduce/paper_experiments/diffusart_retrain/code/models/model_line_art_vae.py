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



class Unet(nn.Module):
    def __init__(
            self,
            dim,
            init_dim=None,
            out_dim=None,
            dim_mults=(1, 2, 4,),
            channels=3,
            with_time_emb=True,
            resnet_block_groups=8,
            use_convnext=False,
            convnext_mult=2,
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels

        init_dim = default(init_dim, dim // 2)
        init_dim_0 = int(init_dim / 2)
        self.init_conv = nn.Conv2d(channels, init_dim_0, 3, padding=1)


        dims = [init_dim_0, init_dim, *map(lambda m: dim * m, dim_mults), 512]

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
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)



        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        # Residual_cross(PreNorm_cross(dim_out, CrossAttention(dim_out, dim_out))),
                        # PreNorm_cross(dim_out, CrossAttention(dim_out, dim_out)),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]

        self.mid_cross_attn_1 = PreNorm_cross(mid_dim, CrossAttention(mid_dim, mid_dim))
        self.mid_cross_attn_2 = PreNorm_cross(mid_dim, CrossAttention(mid_dim, mid_dim))
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn_1 = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_attn_2 = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_block3 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        # self.upsample_mid = Upsample(dim_out)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        # Residual(PreNorm(dim_in, EfficientAttention(dim_in))),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        out_dim = 4
        self.final_conv = nn.Sequential(
            block_klass(init_dim, init_dim),
            # nn.Conv2d(init_dim, out_dim, 1)
            nn.GroupNorm(32, init_dim),
            nn.Conv2d(init_dim, out_dim, kernel_size=(3, 3), padding=1)
        )

    def forward(self, x, time):


        t = self.time_mlp(time) if exists(self.time_mlp) else None

        x = self.init_conv(x)


        h = []
        i = 0
        b = 0

        # downsample
        for block1, block2, attn, downsample in self.downs:
            i += 1
            x = block1(x, t)
            x = block2(x, t)
            # print(i)
            if i >= 2:
                x = attn(x)
            h.append(x)

            x = downsample(x)

        # bottleneck
        # x = self.mid_cross_attn_1(x, feat2)
        x = self.mid_block1(x, t)
        x = self.mid_attn_1(x)
        x = self.mid_block2(x, t)
        # x = self.mid_cross_attn_2(x, feat_mid)
        # x = self.mid_attn_2(x)
        # x = self.mid_block3(x, t)
        # x = self.upsample_mid(x)






        # upsample
        for block1, block2, attn, upsample in self.ups:
            b += 1
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            if b <= 3:
                x = attn(x)
            x = upsample(x)

        out = self.final_conv(x)
        return out