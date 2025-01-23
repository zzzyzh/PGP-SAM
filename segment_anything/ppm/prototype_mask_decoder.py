# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Modified by Zhonghao Yan from: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/mask_decoder.py

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import List, Tuple

import torch
from torch import nn, Tensor

from .module import ConvLayer2d, UpSample2d
from .common import PositionEmbeddingSine, MLP, get_norm, get_act


class HierMaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer: nn.Module,
        embed_dim: int,
        num_multimask_outputs: int = 3,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        tranformer architecture.

        Args:
            transformer (nn.Module): the transformer used to predict masks
            embed_dim (int): the channel dimension of the transformer
            num_multimask_outputs (int): the number of masks to predict when disambiguating masks
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.transformer = transformer
        
        # positional encoding
        N_steps = embed_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        # learnable token p.e.
        self.pos_token = nn.Embedding(1, embed_dim)
        
        self.num_mask_tokens = num_multimask_outputs
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, embed_dim)
        
        dim = 1024
        self.norm = get_norm('LN', num_multimask_outputs)
        self.attn = Attention(dim, 8)
        self.act_func = get_act('leaky')
        
        
        self.mask_norm = get_norm('LN', dim, d=False)
        self.mask_proj1 = nn.Linear(num_multimask_outputs, 1)
        self.mask_act = get_act('leaky')
        self.mask_proj2 = ConvLayer2d(1, 1, 1)

        self.output_upscaling = nn.Sequential(
            UpSample2d(embed_dim, embed_dim // 4, norm='LN', act_func='gelu'),
            UpSample2d(embed_dim // 4, embed_dim // 8, act_func='gelu')
        )
        
        self.skip_connect = nn.Sequential(
            ConvLayer2d(embed_dim // 4, embed_dim // 8, 1, norm='LN', act_func='gelu'),
            ConvLayer2d(embed_dim // 8, embed_dim // 8, 1, norm='LN', act_func='gelu')
        )
        
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(embed_dim, embed_dim, embed_dim // 8, 3)
                for _ in range(self.num_mask_tokens)
            ]
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        up_embeds: torch.Tensor = None,
        mask_embeds: torch.Tensor = None,
        ps_masks: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Args:
            image_embeddings (torch.Tensor): the embeddings from the image encoder
            dense_prompt_embeddings (torch.Tensor): the dense prompt embeddings generated from the image
            sparse_prompt_embeddings (torch.Tensor): the sparse prompt embeddings generated from the image
            up_embeds (torch.Tensor): upsampled image embeddings to match the output dimensions
            mask_embeds (torch.Tensor): mask class features from the previous output
            ps_masks (torch.Tensor): predicted masks from the previous iteration
        """
        masks, mask_embeds = self.predict_masks(
            image_embeddings=image_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            up_embeds=up_embeds,
            mask_embeds=mask_embeds,
            ps_masks=ps_masks
        )
        
        return masks, mask_embeds

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        up_embeds: torch.Tensor = None,
        mask_embeds: torch.Tensor = None,
        ps_masks: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.
        """
        # Concatenate output tokens
        pos_tokens = self.pos_token.weight.reshape(1, 1, -1)
        mask_tokens = self.mask_tokens.weight.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((mask_tokens, sparse_prompt_embeddings), dim=1)                              # [b, num_cls * 2, c]

        # Expand per-image data in batch direction to be per-mask
        pos_src = self.pe_layer(image_embeddings, None)
        src = image_embeddings + dense_prompt_embeddings                                                # [b, c, h, w]
        b, c, h, w = src.shape
        
        # Enhance the src with mask_embeds
        if mask_embeds != None:
            mask_embeds = self.norm(mask_embeds).reshape(b, -1, h*w)
            mask_embeds = self.attn(mask_embeds, mask_embeds, mask_embeds)
            attn = torch.einsum('bcd,bnd->bcn', src.flatten(2), ps_masks.flatten(2))
            attn = self.act_func(attn)
            attn = torch.einsum('bcn,bnd->bcd', attn, mask_embeds)
            src = src + attn.reshape(b, c, h, w)
        
        # Run the transformer
        hs, src, _ = self.transformer(src, pos_src, tokens, pos_tokens, ps_masks)
        mask_tokens_out = hs[:, : self.num_mask_tokens, :]                                              # [b, num_cls, c]
        mask_embeds = torch.matmul(mask_tokens_out, src.transpose(2, 1))                                # [b, num_cls, d]
        mask_embeds = mask_embeds.reshape(b, -1, h, w)

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)                                                 # [b, 32, 128, 128]
        if up_embeds != None:
            up_embeds = torch.cat((upscaled_embedding, up_embeds), dim=1)
            upscaled_embedding = self.skip_connect(up_embeds) + upscaled_embedding
            
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)                                                    # [b, num_cls, 32]

        b, c, h, w = upscaled_embedding.shape                                                           # [b, 32, h, w]
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)                     # [b, num_cls, 128, 128]

        return masks, mask_embeds


class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 2,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out

