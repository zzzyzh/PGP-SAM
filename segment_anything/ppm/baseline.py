# Copyright (c) Zhonghao Yan.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import fvcore.nn.weight_init as weight_init

from einops import rearrange

import torch
import torchvision
from torch import nn, Tensor
from torch.nn import functional as F

from .module import ConvLayer2d, ConvLayer1d, UpSample2d
from .common import MLP, get_norm, get_act


class Baseline(nn.Module):
    def __init__(
        self, 
        stage=4,
        embed_dim=768,
        feat_dim=256,
        num_heads=8,
        num_classes=9,
        num_tokens=8,
        norm='LN',
    ):  
        super().__init__()
        
        # module
        self.feature_refinement = nn.ModuleList()
        self.prototype_refinement = nn.ModuleList()
        self.prompt_generator = nn.ModuleList()
        
        for _ in range(stage):
            self.feature_refinement.append(
                FeatureRefinement(
                    in_dim=embed_dim,
                    conv_dim=feat_dim,
                    out_dim=feat_dim,
                    norm=norm
                )
            )
            
            self.prompt_generator.append(
                PromptGenerator(
                    feat_dim=feat_dim,
                    num_classes=num_classes,
                    num_tokens=num_tokens,
                    norm=norm,
                )
            )
            
        self.num_classes = num_classes
        self.num_tokens = num_tokens
        
    def forward(self, idx, interm_embed, out_embed=None, mask_feats=None, prototypes=None, class_prototypes=None, query_embed=None, masks=None):
                
        out_embed = self.feature_refinement[idx](interm_embed, out_embed)
        
        features, dense_embeddings, sparse_embeddings, prototype_tokens = self.prompt_generator[idx](out_embed, prototypes, class_prototypes, masks=masks)

        return features, None, out_embed, prototypes, class_prototypes, dense_embeddings, sparse_embeddings, prototype_tokens


class FeatureRefinement(nn.Module):
    def __init__(
        self, 
        in_dim=768,
        conv_dim=256,
        out_dim=256,
        norm='LN'
    ):  
        super().__init__()
        
        self.proj = ConvLayer2d(in_dim, conv_dim, 1, bias=False, norm=norm)
        
    def forward(self, interm_embed, out_embed):
        
        interm_features = self.proj(interm_embed.permute(0,3,1,2))

        x = interm_features + out_embed
                
        return x



# Adapted from: https://github.com/wenxi-yue/SurgicalSAM/blob/main/surgicalSAM/model.py
class PromptGenerator(nn.Module):
    def __init__(
        self, 
        feat_dim=256,
        feat_size=32,
        num_classes=9,
        num_tokens=8,
        norm='LN',
    ):
        super().__init__()
        
        self.linear = nn.Linear(feat_dim, feat_dim)
        self.norm = get_norm(norm, feat_dim, d=False)
        self.proj = nn.Linear(num_classes*num_tokens, num_classes)
        self.fusion = nn.Linear(feat_dim, feat_dim)
        self.alpha = nn.Parameter(torch.ones(feat_dim) * 1e-3)

        self.conv = nn.Sequential(
            ConvLayer2d(feat_dim*num_classes, feat_dim, 3, groups=feat_dim, bias=False, norm=norm, act_func='relu'),
            ConvLayer2d(feat_dim, feat_dim, 1)
        )

        self.dense_conv = nn.Sequential(
            ConvLayer2d(feat_dim, feat_dim//2, 1, bias=False, norm=norm, act_func='gelu'),
            ConvLayer2d(feat_dim//2, feat_dim, 1)
        )
        
        self.sparse_conv = nn.Sequential(
            ConvLayer1d((feat_size**2), feat_dim//2, feat_dim, 1, bias=False, norm=norm, act_func='gelu'),
            ConvLayer1d(feat_dim//2, num_classes, 1),
        )
        
        self.mask_embed = nn.Embedding(num_classes, feat_dim)
        self.alpha
        
        self.num_classes = num_classes
        self.num_tokens = num_tokens

    def forward(self, image_embed, prototypes, class_prototypes, query_prototypes=None, masks=None):
        res = image_embed
        b, c, h, w = image_embed.shape
        
        # compute prototype tokens
        if query_prototypes != None:
            prototype_tokens = query_prototypes + self.linear(query_prototypes * prototypes)
        else:
            prototype_tokens = self.linear(prototypes)
        prototype_tokens = self.norm(prototype_tokens)
        prototype_tokens = self.proj(prototype_tokens.permute(0, 2, 1)).permute(0, 2, 1)
        prototype_tokens = self.fusion(prototype_tokens * class_prototypes) * self.alpha.reshape(1, 1, -1) + prototype_tokens

        attn_embed = torch.einsum("bqc,bchw->bqhw", prototype_tokens, image_embed)
        if masks != None:
            masks = torch.softmax(masks, dim=1)
            image_embed = image_embed.unsqueeze(1) * masks.unsqueeze(2)
            image_embed = rearrange(image_embed, 'b n c h w -> b n (h w) c')
        else:
            image_embed = rearrange(image_embed, 'b c h w -> b (h w) c')
            image_embed = image_embed.unsqueeze(1).expand(-1, self.num_classes, -1, -1)
        image_embed = image_embed + self.mask_embed.weight.reshape(1, self.num_classes, 1, c)
        attn_embed = rearrange(attn_embed, 'b c h w -> b c (h w)')
        attn_embed = image_embed * attn_embed.unsqueeze(-1) + image_embed # [b, num_cls, h*w, 256]

        # compute out
        attn_embed = attn_embed.reshape(b, -1, h, w)
        attn_embed = self.conv(attn_embed) + res

        # compute dense embeddings
        dense_embeddings = self.dense_conv(attn_embed)
        # compute sparse embeddings
        sparse_embed = attn_embed.reshape(b, -1, c)
        sparse_embeddings = self.sparse_conv(sparse_embed)
        sparse_embeddings = sparse_embeddings.reshape(b, -1, c)

        return attn_embed, dense_embeddings, sparse_embeddings, prototype_tokens


