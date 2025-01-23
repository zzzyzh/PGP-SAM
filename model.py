# Copyright (c) Zhonghao Yan.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch 
import torch.nn as nn 
from torch.nn import functional as F

from segment_anything import sam_model_registry
from segment_anything.ppm import PrePrompt, PrototypePromptEncoder


class PGP_SAM(nn.Module):
    def __init__(
        self,
        sam_checkpoint,
        sam_mode,
        model_type,
        stage,
        mask_size = 512,
        resolution = 512,
        feat_dim = 256,
        num_classes = 9,
        num_tokens = 8,
    ):
        """
        PGP_SAM is a model designed to enhance prompt generation 
        using inter_prototypes to guide the SAM (Segment Anything Model) for 
        efficient medical image segmentation. It leverages prototype tokens 
        and prompt embeddings to improve few-shot segmentation performance 
        across multiple classes in medical imaging tasks.
        
        Args:
            sam_checkpoint (str): path to the sam checkpoint file
            sam_mode (str): size of the sam model weights, options are ['vit-h', 'vit-l', 'vit-b']
            model_type (str): method for fine-tuning the image encoder
            stage (str): total stage of the model
            mask_size (int): size of the output segmentation mask
            resolution (int): resolution of the input image
            feat_dim (int): dimension of the feature embeddings
            num_classes (int): number of target classes for segmentation
            num_tokens (int): number of query tokens for inter_prototypes
        """
        super().__init__()
        
        # ======> Load SAM
        self.image_encoder, self.mask_decoder = sam_model_registry[sam_mode](checkpoint=sam_checkpoint,
                                                                             model_type=model_type,
                                                                             image_size=resolution,
                                                                             num_classes=num_classes)             
        
        embed_dim = {
            'vit_b': 768,
            'vit_l': 1024,
            'vit_h': 1280
        }
        vit_dim = embed_dim[sam_mode]
        
        # ======> Load Prototype-based Prompt Encoder
        self.prototype_prompt_encoder = PrototypePromptEncoder(stage=stage,
                                                               embed_dim=vit_dim,
                                                               feat_dim=feat_dim,
                                                               num_classes=num_classes,
                                                               num_tokens=num_tokens)
        
        # ======> Pre Prompt Generator and Mask Decoder
        self.pre_prompt = PrePrompt(feat_dim=feat_dim,
                                    num_classes=num_classes,
                                    num_tokens=num_tokens)

        # ======> Load Global Class Prototypes and Query Prototypes
        self.global_prototypes = GlobalPrototypes(feat_dim=feat_dim,
                                                  num_classes=num_classes,
                                                  num_tokens=num_tokens)
        
        self.stage = stage
        self.mask_size = mask_size
        
    def forward(self, images: torch.Tensor):
        """
        Refines the segmentation results iteratively by leveraging 
        multi-scale information.

        Args:
            images (torch.Tensor): a batch of input images after data augmentation.
            gts (torch.Tensor, optional): ground truth used to generate class embeddings 
                                        during training. Defaults to None.
        """
        image_embeddings, interm_embeddings = self.image_encoder(images)
        interm_embeddings = [interm_embeddings[-1], interm_embeddings[0]]
        
        inter_prototypes, intra_prototypes, inter_embed, intra_embed = self.global_prototypes()
        
        B, _, H, W = image_embeddings.shape
        inter_prototypes = inter_prototypes.unsqueeze(0).expand(B, -1, -1)
        intra_prototypes = intra_prototypes.unsqueeze(0).expand(B, -1, -1)
        inter_embed = inter_embed.unsqueeze(0).expand(B, -1, -1)
        intra_embed = intra_embed.unsqueeze(0).expand(B, -1, -1)
        
        low_res_masks, mask_embed, out_embed = self.pre_prompt(image_embeddings, inter_prototypes, intra_prototypes, inter_embed, intra_embed, self.mask_decoder)
        pred_masks = [low_res_masks]

        for idx, (interm_embed) in enumerate(interm_embeddings):
            
            ps_masks = F.interpolate(low_res_masks, (H,W), mode="bilinear", align_corners=False)
            
            (out_embed, up_embed, mask_embed, inter_prototypes, intra_prototypes, 
             dense_prompts, sparse_prompts) = self.prototype_prompt_encoder(idx,
                                                                            interm_embed,
                                                                            out_embed=out_embed,
                                                                            mask_embed=mask_embed,
                                                                            inter_prototypes=inter_prototypes,
                                                                            intra_prototypes=intra_prototypes,
                                                                            inter_embed=inter_embed,
                                                                            intra_embed=intra_embed,
                                                                            masks=ps_masks)
            
            low_res_masks, mask_embed = self.mask_decoder(
                image_embeddings=out_embed,                     # [b, c, h, w]
                dense_prompt_embeddings=dense_prompts,          # [b, c, h, w]
                sparse_prompt_embeddings=sparse_prompts,        # [b, q, c]
                up_embeds=up_embed,
                mask_embeds=mask_embed,
                ps_masks=ps_masks,
            )
            
        masks = F.interpolate(low_res_masks, (self.mask_size, self.mask_size), mode="bilinear", align_corners=False)
        pred_masks.append(masks)

        return pred_masks
   

class GlobalPrototypes(nn.Module):
    def __init__(
        self,
        feat_dim = 256,
        num_classes = 9,
        num_tokens = 8,
    ):
        super().__init__()
        # learnable inter inter_prototypes
        self.inter_prototypes = nn.Embedding(num_classes*num_tokens, feat_dim)
        # learnable intra inter_prototypes
        self.intra_prototypes = nn.Embedding(num_classes, feat_dim)
        # learnable inter p.e.
        self.inter_embed = nn.Embedding(num_classes*num_tokens, feat_dim)
        # learnable intra p.e.
        self.intra_embed = nn.Embedding(num_classes, feat_dim)
        
    def forward(self):
        inter_prototypes = self.inter_prototypes.weight
        intra_prototypes = self.intra_prototypes.weight
        inter_embed = self.inter_embed.weight
        intra_embed = self.intra_embed.weight
                
        return inter_prototypes, intra_prototypes, inter_embed, intra_embed

