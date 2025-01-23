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
import torch.nn.functional as F

from .module import ConvLayer2d, UpSample2d
from .common import MLP, get_norm, get_act


class PrePrompt(nn.Module):
    def __init__(
        self,
        feat_dim=256,
        num_classes=9,
        num_tokens=8,
    ):
        super().__init__()
        
        self.mask_embed = nn.Embedding(1, feat_dim)
        self.csm = CSM(feat_dim)
        
        self.prompt_generator = PromptGenerator(feat_dim=feat_dim,
                                                num_classes=num_classes,
                                                num_tokens=num_tokens)

        self.num_classes = num_classes
        self.num_tokens = num_tokens
        
    def forward(self, image_embed: torch.Tensor, inter_prototypes: torch.Tensor, intra_prototypes: torch.Tensor, inter_embed: torch.Tensor, intra_embed: torch.Tensor, mask_decoder: torch.Tensor):

        out_embed = image_embed + self.mask_embed.weight.reshape(1, -1, 1, 1)
        out_embed = self.csm(out_embed)
        
        dense_prompts, sparse_prompts = self.prompt_generator(out_embed, inter_prototypes, intra_prototypes, inter_embed, intra_embed, None)

        low_res_masks, mask_embed = mask_decoder(
            image_embeddings=image_embed,                   # [b, c, h, w]
            dense_prompt_embeddings=dense_prompts,          # [b, c, h, w]
            sparse_prompt_embeddings=sparse_prompts,        # [b, q, c]
            up_embeds=None,
            mask_embeds=None,
            ps_masks=None,
        )
        
        return low_res_masks, mask_embed, out_embed


class CSM(nn.Module):
    def __init__(
        self,
        hidden_dim,
        factor=2,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            ConvLayer2d(hidden_dim, hidden_dim//factor, 1, bias=False, norm='LN', act_func='leaky'),
            ConvLayer2d(hidden_dim//factor, hidden_dim, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_avg = torch.mean(x, dim=[2, 3], keepdim=True)
        x_w = self.mlp(x_avg)
        x_w = self.sigmoid(x_w)
        
        return x + x * x_w


class PrototypePromptEncoder(nn.Module):
    def __init__(
        self, 
        stage=2,
        embed_dim=768,
        feat_dim=256,
        num_heads=8,
        num_classes=9,
        num_tokens=8,
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
                    out_dim=feat_dim,
                )
            )
            self.prototype_refinement.append(
                PrototypeRefinement(
                    feat_dim=feat_dim,
                    num_heads=num_heads,
                    num_classes=num_classes,
                    num_tokens=num_tokens,
                )
            )
            self.prompt_generator.append(
                PromptGenerator(
                    feat_dim=feat_dim,
                    num_classes=num_classes,
                    num_tokens=num_tokens
                    )
            )

        # learnable level embedding 
        self.level_image = nn.Embedding(stage, feat_dim)
        self.level_mask = nn.Embedding(stage, num_classes)

        self.num_classes = num_classes
        self.num_tokens = num_tokens
        
    def forward(self, idx, interm_embed: torch.Tensor, out_embed: torch.Tensor, mask_embed: torch.Tensor, inter_prototypes: torch.Tensor, intra_prototypes: torch.Tensor, inter_embed: torch.Tensor, intra_embed: torch.Tensor, masks: torch.Tensor):

        out_embed, up_embed = self.feature_refinement[idx](interm_embed, out_embed)
        
        masks = torch.softmax(masks, dim=1)
        out_embed = out_embed + self.level_image.weight[idx].reshape(1, -1, 1, 1)
        mask_embed = mask_embed + self.level_mask.weight[idx].reshape(1, -1, 1, 1)
        
        inter_prototypes, intra_prototypes = self.prototype_refinement[idx](out_embed, mask_embed, inter_prototypes, intra_prototypes, inter_embed, intra_embed, masks)
        
        dense_prompts, sparse_prompts = self.prompt_generator[idx](out_embed, inter_prototypes, intra_prototypes, inter_embed, intra_embed, masks)
        
        return out_embed, up_embed, mask_embed, inter_prototypes, intra_prototypes, dense_prompts, sparse_prompts


class FeatureRefinement(nn.Module):
    def __init__(
        self, 
        in_dim=768,
        out_dim=256,
    ):  
        super().__init__()
        
        self.proj = ConvLayer2d(in_dim, out_dim, 1, bias=False, norm='LN')
        
        self.scm = SCM(out_dim)
        self.final = FFC(out_dim, out_dim)

        self.up_sample = nn.Sequential(
            UpSample2d(out_dim, out_dim // 4, norm='LN', act_func='leaky'),
            UpSample2d(out_dim // 4, out_dim // 8)
        )

    def forward(self, interm_embed, out_embed):
        
        interm_features = self.proj(interm_embed.permute(0,3,1,2))

        x = self.scm(interm_features) + out_embed
        x = self.final(x)
        
        x_up = self.up_sample(x)

        return x, x_up


class SCM(nn.Module):
    def __init__(
        self,
        dim,
    ):
        super().__init__()

        # SEM: Spatial Infor Enhancement Module
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.sem_conv = nn.Sequential(
            ConvLayer2d(dim, dim, (1, 5), groups=dim, bias=False, norm='LN', act_func='leaky'),
            ConvLayer2d(dim, dim, (5, 1), groups=dim, bias=False),
            nn.Sigmoid(),
        )
        
        # CEM: Channel Infor Enhancement Module
        self.pool_c = nn.AdaptiveAvgPool2d((1, 1))
        self.cem_conv = nn.Sequential(
            ConvLayer2d(dim, dim//2, 1, bias=False, norm='LN', act_func='leaky'),
            ConvLayer2d(dim//2, dim, 1, bias=False),
            nn.Sigmoid(),
        )
        
        self.conv = nn.Sequential(
            ConvLayer2d(2*dim, dim, 1, groups=dim, bias=False, norm='LN', act_func='leaky'),
            ConvLayer2d(dim, dim, 1),
        )
        
        self.alpha = nn.Parameter(torch.ones(dim) * 1e-3)
        self.beta = nn.Parameter(torch.ones(dim) * 1e-3)

    def sem(self, x):
        x_h = self.pool_h(x)
        x_w = self.pool_w(x)
        x_gather = x_h + x_w
        sptial_x = self.sem_conv(x_gather)

        return sptial_x * x
    
    def cem(self, x):
        x_c = self.pool_c(x)
        channel_x = self.cem_conv(x_c)

        return channel_x * x
    
    def forward(self, x):
        res = x
        
        sptial_x = x + self.sem(x) * self.alpha.reshape(1, -1, 1, 1)
        channel_x = x + self.cem(x) * self.beta.reshape(1, -1, 1, 1)
        
        x = torch.cat([sptial_x, channel_x], dim=1)
        x = self.conv(x)
        x = x + res
        
        return x


class FFC(nn.Module):
    def __init__(
        self, 
        in_dim, 
        out_dim
    ):
        super().__init__()

        self.norm = get_norm('LN', in_dim)
        self.project_in = ConvLayer2d(in_dim, in_dim*2, 1)
        self.dwconv = ConvLayer2d(in_dim*2, in_dim*2, 3, groups=in_dim*2, bias=False)
        self.act_func = get_act('leaky')
        self.project_out = ConvLayer2d(in_dim, out_dim, 1)

    def forward(self, x):
        res = x
        
        x = self.norm(x)
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = self.act_func(x1) * x2
        x = self.project_out(x) + res

        return x


class PrototypeRefinement(nn.Module):
    def __init__(
        self, 
        feat_dim=256,
        num_heads=8,
        num_classes=9,
        num_tokens=8,
    ):
        super().__init__()

        self.in_proj1 = nn.Linear(feat_dim, feat_dim)
        self.in_proj2 = nn.Linear(feat_dim, feat_dim)

        in_feature, out_feature = 1024, 32
        self.class_attn = ClassAttention(feat_dim, in_feature, out_feature, num_heads, num_classes, num_tokens)
        self.inter = nn.Parameter(torch.ones(feat_dim) * 1e-3)
        self.intra = nn.Parameter(torch.ones(feat_dim) * 1e-3)
        
        self.linear1 = nn.Linear(num_classes*num_tokens + num_classes, num_classes*num_tokens)
        self.linear2 = nn.Linear(feat_dim + num_classes, feat_dim)

        self.out_proj1 = FFL(feat_dim, feat_dim)
        self.out_proj2 = FFL(feat_dim, feat_dim)
        
        self.num_classes = num_classes
        self.num_tokens = num_tokens
        
        self._reset_parameters()

    def _reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
           
    def forward(self, image_embed: torch.Tensor, mask_embed: torch.Tensor, inter_prototypes: torch.Tensor, intra_prototypes: torch.Tensor, inter_embed: torch.Tensor, intra_embed: torch.Tensor, masks: torch.Tensor):
        inter_prototypes = inter_prototypes + inter_embed
        intra_prototypes = intra_prototypes + intra_embed
        
        inter_prototypes = self.in_proj1(inter_prototypes)
        intra_prototypes = self.in_proj2(intra_prototypes)

        # Most-similar Prototypes
        inter_class_norm = F.normalize(inter_prototypes, dim=-1)  # [b, query, 256]
        intra_class_norm = F.normalize(intra_prototypes, dim=-1)  # [b, class, 256]

        cos_similarity = torch.einsum('bcd,bqd->bcq', intra_class_norm, inter_class_norm)               # [b, class, query]
        _, topk_indices = torch.topk(cos_similarity, k=self.num_tokens//2, dim=-1)

        expanded_indices = topk_indices.unsqueeze(-1).expand(-1, -1, -1, inter_prototypes.shape[-1])    # [b, class, k, 256]
        inter_prototypes_expand = inter_prototypes.unsqueeze(1).expand(-1, self.num_classes, -1, -1)    # [b, class, query, 256]
        sim_prototypes = torch.gather(inter_prototypes_expand, 2, expanded_indices)                     # [b, class, k, 256]
        
        # Class Attention
        B, _, C = intra_prototypes.shape
        prototypes = torch.cat((intra_prototypes.unsqueeze(2), sim_prototypes), dim=2)                  # [b, class, k+1, 256]
        prototypes = prototypes.view(B, -1, C)
        prototypes = self.class_attn(prototypes,            # [b, class*(k+1), c]
                                     image_embed,           # [b, c, h, w]
                                     mask_embed,            # [b, class, h, w]
                                     masks                  # [b, class, h, w]
                                     )

        # Prototypes Fusion
        prototypes = prototypes.view(B, self.num_classes, -1, C)
        inter_class_prototypes = prototypes[:, :, 1:, :]    
        topk_indices = topk_indices.reshape(B, -1)
        inter_class_prototypes = inter_class_prototypes.reshape(B, -1, C)
        inter_updates = torch.zeros_like(inter_prototypes)
        inter_updates.scatter_add_(dim=1, index=topk_indices.unsqueeze(-1).expand(-1, -1, C), src=inter_class_prototypes)
        inter_prototypes = inter_prototypes + inter_updates*self.inter.reshape(1, 1, -1)

        intra_class_prototypes = prototypes[:, :, 0, :]
        intra_prototypes = intra_prototypes + intra_class_prototypes*self.intra.reshape(1, 1, -1)
        
        # Mask Prototypes
        mask_inter_prototypes = torch.einsum('bchw,bnhw->bnc', image_embed, masks)
        inter_prototypes = self.linear1(torch.cat((inter_prototypes, mask_inter_prototypes), dim=1).permute(0, 2, 1)).permute(0, 2, 1)
        mask_intra_prototypes = torch.einsum('bkhw,bnhw->bnk', mask_embed, masks)
        intra_prototypes = self.linear2(torch.cat((intra_prototypes, mask_intra_prototypes), dim=-1))
        
        inter_prototypes = self.out_proj1(inter_prototypes)
        intra_prototypes = self.out_proj2(intra_prototypes)

        return inter_prototypes, intra_prototypes


class ClassAttention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        in_feature: int,
        out_feature: int,
        num_heads: int,
        num_classes: int,
        num_tokens: int
    ):
        super().__init__()
        
        self.q_norm = get_norm('LN', embedding_dim, d=False)
        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.x_proj = nn.Sequential(
            ConvLayer2d(embedding_dim + num_classes, embedding_dim, 1, act_func='gelu'),
            ConvLayer2d(embedding_dim, embedding_dim, 1, norm='LN')
        )
        self.x_conv = ConvLayer2d(embedding_dim, embedding_dim, 3, groups=embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)

        self.image_conv = ConvLayer2d(embedding_dim, 1, 1)
        self.mask_conv = ConvLayer2d(num_classes, 1, 1)
        self.image_proj = nn.Linear(in_feature, out_feature)
        self.mask_proj = nn.Linear(in_feature, out_feature)
        self.conv = ConvLayer2d(1, num_classes*(num_tokens//2+1), 1)
        
        self.alpha = nn.Parameter(torch.ones(in_feature) * 1e-3)
        self.beta = nn.Parameter(torch.ones(in_feature) * 1e-3)
        
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

        self.num_heads = num_heads
        self.num_classes = num_classes
        self.num_tokens = num_tokens
        self._reset_parameters()

    def _reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
           
    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)

    def forward(self, query_prototypes: Tensor, image_embed: Tensor, mask_embed: Tensor, masks: Tensor) -> Tensor:
        
        # Input
        query = self.q_proj(self.q_norm(query_prototypes))
        x = self.x_proj(torch.cat((image_embed, mask_embed), dim=1))
        x = self.x_conv(x).flatten(2).permute(0, 2, 1)
        key = self.k_proj(x)
        value = self.v_proj(x)
        
        # Multi-head Attn
        query = self._separate_heads(query, self.num_heads)         # [b, n, q, c]
        key = self._separate_heads(key, self.num_heads)             # [b, n, d, c]
        value = self._separate_heads(value, self.num_heads)         # [b, n, d, c]
        B, N, Q, C = query.shape

        # Class Attn
        att_image = rearrange(self.image_proj(rearrange(self.image_conv(image_embed), 'b 1 h w -> b (h w)')), 'b h -> b h 1')
        att_mask = rearrange(self.mask_proj(rearrange(self.mask_conv(mask_embed), 'b 1 h w -> b (h w)')), 'b w -> b 1 w')
        att_class = rearrange(torch.matmul(att_image, att_mask), 'b h w -> b 1 h w')
        att_class = self.conv(att_class)                            # [b, q, h, w]
        att_class = att_class.flatten(2).unsqueeze(1)               # [b, 1, q, d]

        # Query Attn
        att_query = query @ key.permute(0, 1, 3, 2)                 # [b, n, q, d]
        att_query = att_query.view(B, N, self.num_classes, self.num_tokens//2+1, -1) * masks.flatten(2).reshape(B, 1, self.num_classes, 1, -1)
        att_query = att_query.reshape(B, N, Q, -1)
        
        # Output
        att_weight = (att_class * self.alpha + att_query * self.beta) / math.sqrt(C)
        att_weight = torch.softmax(att_weight, dim=-1)
        out = att_weight @ value
        out = self._recombine_heads(out)                            # [b, q, c]
        out = self.out_proj(out)
        
        return out


class FFL(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim
    ): 
        super().__init__()
        
        self.norm = get_norm('LN', in_dim, d=False)
        self.ffl = nn.Sequential(
            nn.Linear(in_dim, in_dim*4),
            get_act('gelu'),
            nn.Linear(in_dim*4, out_dim)
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
           
    def forward(self, prototypes):
        res = prototypes
        
        prototypes = self.norm(prototypes)
        prototypes = self.ffl(prototypes)

        prototypes = prototypes + res
        
        return prototypes


class PromptGenerator(nn.Module):
    def __init__(
        self, 
        feat_dim=256,
        num_classes=9,
        num_tokens=8
    ):  
        super().__init__()
        
        self.dense_prompt_generator = DensePromptGenerator(feat_dim=feat_dim,
                                                           num_classes=num_classes,
                                                           num_tokens=num_tokens)
        self.sparse_prompt_generator = SparsePromptGenerator(feat_dim=feat_dim,
                                                             num_classes=num_classes,
                                                             num_tokens=num_tokens)

    def forward(self, image_embed, inter_prototypes=None, intra_prototypes=None, inter_embed=None, intra_embed=None, masks=None):
        
        inter_prototypes = inter_prototypes + inter_embed
        intra_prototypes = intra_prototypes + intra_embed

        dense_prompts = self.dense_prompt_generator(image_embed, inter_prototypes, intra_prototypes, masks)
        sparse_prompts = self.sparse_prompt_generator(image_embed, inter_prototypes, intra_prototypes, masks)
        
        return dense_prompts, sparse_prompts


class DensePromptGenerator(nn.Module):
    def __init__(
        self, 
        feat_dim=256,
        num_classes=9,
        num_tokens=8
    ):  
        super().__init__()
        
        self.proj = nn.Linear(feat_dim + num_classes, feat_dim)
        self.act_func = get_act('gelu')
        self.linear = nn.Linear(feat_dim, feat_dim)
        self.norm = get_norm('LN', feat_dim, d=False)
        
        self.dcn = DeformAttn(feat_dim)
        
        self.out = nn.Sequential(
            ConvLayer2d(feat_dim, feat_dim//2, 1),
            get_act('gelu'),
            ConvLayer2d(feat_dim//2, feat_dim, 1)
        )
        
        self.alpha = nn.Parameter(torch.ones(feat_dim) * 1e-3)
        self.scale = feat_dim ** -0.5
        self.num_tokens = num_tokens
        self._reset_parameters()

    def _reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
           
    def forward(self, image_embed, inter_prototypes=None, intra_prototypes=None, masks=None):
                
        prompt_tokens = inter_prototypes @ intra_prototypes.permute(0, 2, 1)                # [b, q, n]
        prototype_tokens = self.proj(torch.cat((inter_prototypes, prompt_tokens), dim=-1))
        prototype_tokens = self.act_func(prototype_tokens)
        prototype_tokens = self.linear(prototype_tokens)
        prototype_tokens = self.norm(prototype_tokens)
        
        if masks != None:
            prompt_embed = torch.einsum('bnc,bnhw->bchw', intra_prototypes, masks)
            image_embed = torch.mul(image_embed, prompt_embed) + image_embed
        dense_embed = self.dcn(image_embed)

        attn_embed = torch.einsum("bchw,bqc->bqhw", dense_embed, prototype_tokens)
        attn_embed = attn_embed * self.scale
        attn_embed = torch.softmax(attn_embed, dim=1)
        attn_embed = torch.einsum("bqhw,bqc->bchw", attn_embed, prototype_tokens)
        attn_embed = attn_embed + dense_embed * self.alpha.reshape(1, -1, 1, 1)

        dense_prompts = self.out(attn_embed)
        
        return dense_prompts


class DeformAttn(nn.Module):
    def __init__(
        self,
        dim
    ):
        super().__init__()

        self.in_proj = ConvLayer2d(dim, dim, 1)
        self.act_func = get_act('gelu')
        self.deform_conv0 = DeformConv(dim, 3, groups=dim, dilation=1)
        self.deform_conv1 = DeformConv(dim, 3, groups=dim, dilation=3)
        self.conv = ConvLayer2d(dim, dim, 1)
        
        self.norm = get_norm('LN', dim)
        self.out_proj = nn.Sequential(
            ConvLayer2d(dim, dim, 1),
            ConvLayer2d(dim, dim, 3, groups=dim, act_func='gelu'),
            ConvLayer2d(dim, dim, 1)
        )
        
    def forward(self, image_embed):
        
        x = self.in_proj(image_embed)
        x = self.act_func(x)
        x = self.deform_attn(x) + image_embed
        
        x = self.norm(x)
        x = self.out_proj(x) + x
        
        return x
        
    def deform_attn(self, x):
        attn = self.deform_conv0(x)
        attn = self.deform_conv1(attn)
        attn = self.conv(attn)
        
        return x * attn
    

class DeformConv(nn.Module):
    def __init__(
        self, 
        d_modal, 
        kernel_size=3, 
        groups=1, 
        dilation=1
    ):
        super().__init__()
        
        padding = kernel_size//2 * dilation
        offset_channels = 3 * kernel_size * kernel_size
        self.dcn_offset = nn.Conv2d(in_channels=d_modal,
                                    out_channels=offset_channels,
                                    kernel_size=kernel_size,
                                    padding=padding,
                                    dilation=dilation)

        self.deform_conv = torchvision.ops.DeformConv2d(in_channels=d_modal,
                                                        out_channels=d_modal,
                                                        kernel_size=kernel_size,
                                                        padding=padding,
                                                        groups=groups,
                                                        dilation=dilation)

        self._reset_parameters()

    def _reset_parameters(self):
        weight_init.c2_msra_fill(self.deform_conv)

        nn.init.constant_(self.dcn_offset.weight, 0)
        nn.init.constant_(self.dcn_offset.bias, 0)
 
    def forward(self, x):
        offsets = self.dcn_offset(x)
        offset_x, offset_y, mask = torch.chunk(offsets, 3, dim=1)
        offset = torch.cat((offset_x, offset_y), dim=1)
        mask = mask.sigmoid()
        
        out = self.deform_conv(x, offset, mask)
        return out


class SparsePromptGenerator(nn.Module):
    def __init__(
        self, 
        feat_dim=256,
        num_classes=9,
        num_tokens=8
    ):  
        super().__init__()
        
        self.dconv = ConvLayer2d(feat_dim, feat_dim, 3, dilation=2)
        
        self.prototype_adapter = PrototypeAdapter(feat_dim, num_classes, num_tokens)

        self.out = nn.Sequential(
            nn.Linear(feat_dim, feat_dim//2),
            get_act('gelu'),
            nn.Linear(feat_dim//2, feat_dim)
        )
        
        self.alpha = nn.Parameter(torch.ones(feat_dim) * 1e-3)
        self.scale = feat_dim ** -0.5
        self._reset_parameters()

    def _reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
           
    def forward(self, image_embed, inter_prototypes=None, intra_prototypes=None, masks=None):
        
        image_embed = self.dconv(image_embed)

        sparse_embed = self.prototype_adapter(inter_prototypes, intra_prototypes)
        
        attn_embed = torch.einsum("bqc,bcd->bqd", sparse_embed, image_embed.flatten(2))
        attn_embed = attn_embed * self.scale
        attn_embed = torch.softmax(attn_embed, dim=-1)
        if masks != None:
            attn_embed = torch.mul(attn_embed, masks.flatten(2))
        attn_embed = torch.einsum("bqd,bcd->bqc", attn_embed, image_embed.flatten(2))
        attn_embed = attn_embed + sparse_embed * self.alpha.reshape(1, 1, -1)
        
        sparse_prompts = self.out(attn_embed)

        return sparse_prompts


class PrototypeAdapter(nn.Module):
    def __init__(
        self,
        feat_dim=256,
        num_classes=9,
        num_tokens=8
    ):
        super().__init__()
        
        num_queries = num_classes * num_tokens
        self.in_proj = nn.Linear(feat_dim + num_queries, feat_dim)
        self.act_func = get_act('gelu')
        self.linear = nn.Linear(feat_dim, feat_dim)
        self.norm = get_norm('LN', feat_dim, d=False)
        
        self.adapter_layer = nn.ModuleList()
        for _ in range(num_tokens):
            self.adapter_layer.append(
                nn.Sequential(
                    nn.Linear(feat_dim, feat_dim//4),
                    get_act('gelu'),
                )
            )

        hidden_dim = feat_dim//4 * num_tokens
        self.out_proj = MLP(hidden_dim, feat_dim, feat_dim, 3)
        
        self._reset_parameters()

    def _reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
           
    def forward(self, inter_prototypes, intra_prototypes):
        
        class_tokens = intra_prototypes @ inter_prototypes.transpose(2,1)
        adapt_prototypes = self.in_proj(torch.cat((intra_prototypes, class_tokens), dim=-1))
        adapt_prototypes = self.act_func(adapt_prototypes)
        adapt_prototypes = self.linear(adapt_prototypes)
        adapt_prototypes = self.norm(adapt_prototypes)
        
        adapt_prototypes = torch.cat([adapter_layer(adapt_prototypes) for adapter_layer in self.adapter_layer], dim=-1)
        
        sparse_prompts = self.out_proj(adapt_prototypes)

        return sparse_prompts

