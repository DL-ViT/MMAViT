""" Pooling-based Vision Transformer (PiT) in PyTorch

A PyTorch implement of Pooling-based Vision Transformers as described in
'Rethinking Spatial Dimensions of Vision Transformers' - https://arxiv.org/abs/2103.16302

This code was adapted from the original version at https://github.com/naver-ai/pit, original copyright below.

Modifications for timm by / Copyright 2020 Ross Wightman
"""
# PiT
# Copyright 2021-present NAVER Corp.
# Apache License v2.0

import math
import re
from copy import deepcopy
from functools import partial
from typing import Tuple

import torch
from torch import nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import build_model_with_cfg, overlay_external_default_cfg
from timm.models.layers import trunc_normal_, to_2tuple
from timm.models.layers import DropPath, to_2tuple, trunc_normal_, PatchEmbed, Mlp
try:
    from timm.models.registry import register_model
except ImportError:
    from .registry import register_model
from src.utils.lds import grassmanian_point
from src.utils.riemmanian_model import cov_frobenius_norm

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.conv', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    # deit models (FB weights)
    'manifold_pit_ti_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-pit-weights/pit_ti_730.pth'),
    'manifold_pit_xs_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-pit-weights/pit_xs_781.pth'),
    'manifold_pit_s_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-pit-weights/pit_s_809.pth'),
    'manifold_pit_b_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-pit-weights/pit_b_820.pth'),
    'pit_ti_distilled_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-pit-weights/pit_ti_distill_746.pth',
        classifier=('head', 'head_dist')),
    'pit_xs_distilled_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-pit-weights/pit_xs_distill_791.pth',
        classifier=('head', 'head_dist')),
    'pit_s_distilled_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-pit-weights/pit_s_distill_819.pth',
        classifier=('head', 'head_dist')),
    'pit_b_distilled_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-pit-weights/pit_b_distill_840.pth',
        classifier=('head', 'head_dist')),
    'manifold_pit_ti_32':{



        'num_classes': 1000, 'input_size': (3, 32,32), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.conv', 'classifier': 'head',

    },

    'manifold_pit_xti_32': {

        'num_classes': 1000, 'input_size': (3, 32, 32), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.conv', 'classifier': 'head',

    },
}



class EuclRiemGrassAtt(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, attention_dropout=0.1, projection_dropout=0.1,
                 seq_len=1,
                 ln_attention=True, return_map=False):
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5))
        self.riem_scale = nn.Parameter(torch.tensor(head_dim ** -0.5))
        self.grassman_scale = nn.Parameter(torch.tensor(head_dim ** -0.5))
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(projection_dropout)
        self.seq_len = seq_len

        self.conv_attn = nn.Sequential(
            #nn.LayerNorm((3 * num_heads,seq_len,seq_len),eps=1e-6),
            nn.Conv2d(in_channels=3 * self.num_heads, out_channels=num_heads, kernel_size=(1, 1)),
            nn.BatchNorm2d(num_heads)
        )

        self.return_map = return_map

    def forward(self, x):
       # print(f'x {x.shape}')
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        qgr, _ = grassmanian_point(q)
        kgr, _ = grassmanian_point(k)

        # kgr shape B, H, C', N
        kgr = kgr.permute(0, 1, 3, 2)

        dots = torch.matmul(qgr, kgr).unsqueeze(2)

        attn_grassmman = torch.linalg.norm(dots, dim=2) ** 2. * self.grassman_scale
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_riemmanian = (cov_frobenius_norm(q, k) * self.riem_scale)
        # print(attn_riemmanian.shape)
        attn_ = self.conv_attn(torch.cat((attn, attn_riemmanian, attn_grassmman), dim=1)).softmax(
            dim=-1)

        out = torch.matmul(self.attn_drop(attn_), v)
        # out = rearrange(out, 'b h n d -> b n (h d)')
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)

        return self.proj_drop(self.proj(out))



class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,seq_len =1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = EuclRiemGrassAtt(dim, num_heads=num_heads, qkv_bias=qkv_bias, attention_dropout=attn_drop,
                                     projection_dropout=drop,seq_len=seq_len)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class SequentialTuple(nn.Sequential):
    """ This module exists to work around torchscript typing issues list -> list"""
    def __init__(self, *args):
        super(SequentialTuple, self).__init__(*args)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        for module in self:
            x = module(x)
        return x


class Transformer(nn.Module):
    def __init__(
            self, base_dim, depth, heads, mlp_ratio, pool=None, drop_rate=.0, attn_drop_rate=.0, drop_path_prob=None,seq_len=1):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        embed_dim = base_dim * heads

        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim,
                num_heads=heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_prob[i],
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                seq_len=seq_len
            )
            for i in range(depth)])

        self.pool = pool

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x, cls_tokens = x
        B, C, H, W = x.shape
        token_length = cls_tokens.shape[1]

        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.blocks(x)

        cls_tokens = x[:, :token_length]
        x = x[:, token_length:]
        x = x.transpose(1, 2).reshape(B, C, H, W)

        if self.pool is not None:
            x, cls_tokens = self.pool(x, cls_tokens)
        return x, cls_tokens


class ConvHeadPooling(nn.Module):
    def __init__(self, in_feature, out_feature, stride, padding_mode='zeros'):
        super(ConvHeadPooling, self).__init__()

        self.conv = nn.Conv2d(
            in_feature, out_feature, kernel_size=stride + 1, padding=stride // 2, stride=stride,
            padding_mode=padding_mode, groups=in_feature)
        self.fc = nn.Linear(in_feature, out_feature)

    def forward(self, x, cls_token) -> Tuple[torch.Tensor, torch.Tensor]:

        x = self.conv(x)
        cls_token = self.fc(cls_token)

        return x, cls_token


class ConvEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size, stride, padding):
        super(ConvEmbedding, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=patch_size, stride=stride, padding=padding, bias=True)

    def forward(self, x):
        x = self.conv(x)
        return x


class PoolingVisionTransformer(nn.Module):
    """ Pooling-based Vision Transformer

    A PyTorch implement of 'Rethinking Spatial Dimensions of Vision Transformers'
        - https://arxiv.org/abs/2103.16302
    """
    def __init__(self, img_size, patch_size, stride, base_dims, depth, heads,
                 mlp_ratio, num_classes=1000, in_chans=3, distilled=False,
                 attn_drop_rate=.0, drop_rate=.0, drop_path_rate=.0,attention_type='all',seq_len=[256],**kwargs):
        super(PoolingVisionTransformer, self).__init__()

        padding = 0
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        height = math.floor((img_size[0] + 2 * padding - patch_size[0]) / stride + 1)
        width = math.floor((img_size[1] + 2 * padding - patch_size[1]) / stride + 1)

        self.base_dims = base_dims
        self.heads = heads
        self.num_classes = num_classes
        self.num_tokens = 2 if distilled else 1

        self.patch_size = patch_size
        self.pos_embed = nn.Parameter(torch.randn(1, base_dims[0] * heads[0], height, width))
        self.patch_embed = ConvEmbedding(in_chans, base_dims[0] * heads[0], patch_size, stride, padding)

        self.cls_token = nn.Parameter(torch.randn(1, self.num_tokens, base_dims[0] * heads[0]))
        self.pos_drop = nn.Dropout(p=drop_rate)

        transformers = []
        # stochastic depth decay rule
        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depth)).split(depth)]
        self.seq_len = [256 + 1, 64 + 1, 17]
        for stage in range(len(depth)):

            pool = None
            if stage < len(heads) - 1:
                pool = ConvHeadPooling(
                    base_dims[stage] * heads[stage], base_dims[stage + 1] * heads[stage + 1], stride=2)
            transformers += [Transformer(
                base_dims[stage], depth[stage], heads[stage], mlp_ratio, pool=pool,
                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_prob=dpr[stage],seq_len=self.seq_len[stage])
            ]
        self.transformers = SequentialTuple(*transformers)
        self.norm = nn.LayerNorm(base_dims[-1] * heads[-1], eps=1e-6)
        self.num_features = self.embed_dim = base_dims[-1] * heads[-1]


        # Classifier head
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        if self.head_dist is not None:
            return self.head, self.head_dist
        else:
            return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.head_dist is not None:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x + self.pos_embed)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x, cls_tokens = self.transformers((x, cls_tokens))
        cls_tokens = self.norm(cls_tokens)
        if self.head_dist is not None:
            return cls_tokens[:, 0], cls_tokens[:, 1]
        else:
            return cls_tokens[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            return self.head(x)


def checkpoint_filter_fn(state_dict, model):
    """ preprocess checkpoints """
    out_dict = {}
    p_blocks = re.compile(r'pools\.(\d)\.')
    for k, v in state_dict.items():
        # FIXME need to update resize for PiT impl
        # if k == 'pos_embed' and v.shape != model.pos_embed.shape:
        #     # To resize pos embedding when using model at different size from pretrained weights
        #     v = resize_pos_embed(v, model.pos_embed)
        k = p_blocks.sub(lambda exp: f'transformers.{int(exp.group(1))}.pool.', k)
        out_dict[k] = v
    return out_dict


def _create_pit(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    model = build_model_with_cfg(
        PoolingVisionTransformer, variant, pretrained,
        default_cfg=default_cfgs[variant],
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_strict=False,
        **kwargs)
    return model


@register_model
def manifold_pit_b_224(pretrained, **kwargs):
    model_kwargs = dict(
        patch_size=14,
        stride=7,
        base_dims=[64, 64, 64],
        depth=[3, 6, 4],
        heads=[4, 8, 16],
        mlp_ratio=4,
        **kwargs
    )
    return _create_pit('manifold_pit_b_224', pretrained, **model_kwargs)


@register_model
def manifold_pit_s_224(pretrained, **kwargs):
    model_kwargs = dict(
        patch_size=16,
        stride=8,
        base_dims=[48, 48, 48],
        depth=[2, 6, 4],
        heads=[3, 6, 12],
        mlp_ratio=4,
        **kwargs
    )
    return _create_pit('manifold_pit_s_224', pretrained, **model_kwargs)


@register_model
def manifold_pit_xs_224(pretrained, **kwargs):
    model_kwargs = dict(
        patch_size=16,
        stride=8,
        base_dims=[48, 48, 48],
        depth=[2, 6, 4],
        heads=[2, 4, 8],
        mlp_ratio=4,
        **kwargs
    )
    return _create_pit('manifold_pit_xs_224', pretrained, **model_kwargs)


@register_model
def manifold_pit_ti_224(pretrained, **kwargs):
    model_kwargs = dict(
        patch_size=16,
        stride=8,
        base_dims=[32, 32, 32],
        depth=[2, 6, 4],
        heads=[2, 4, 8],
        mlp_ratio=4,
        **kwargs
    )
    return _create_pit('manifold_pit_ti_224', pretrained, **model_kwargs)

@register_model
def manifold_pit_xti_32(pretrained, **kwargs):
    model_kwargs = dict(
        patch_size=2,
        stride=2,
        base_dims=[48, 48,48],
        depth=[2, 6, 4],
        heads=[2, 4, 8],
        mlp_ratio=2,
        **kwargs
    )
    return _create_pit('manifold_pit_xti_32', pretrained, **model_kwargs)


@register_model
def manifold_pit_ti_32(pretrained, **kwargs):
    model_kwargs = dict(
        patch_size=2,
        stride=2,
        base_dims=[48, 48,48],
        depth=[2, 6, 4],
        heads=[2, 4, 8],
        mlp_ratio=4,
        **kwargs
    )
    return _create_pit('manifold_pit_ti_32', pretrained, **model_kwargs)