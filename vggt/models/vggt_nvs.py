# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from vggt.models.aggregator import Aggregator, NVSAggregator
from vggt.heads.nvs_head import PluckerEncoder
from vggt.heads.dpt_head import DPTHead
from vggt.utils.plucker_rays import generate_plucker_rays, plucker_rays_to_image


class VGGT_NVS(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024, enable_rgb=True):
        super().__init__()

        self.aggregator = NVSAggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.plucker_encoder = PluckerEncoder(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, in_channels=6) 
        
        # choose sigmoid to ensure output is in [0, 1] range
        # output_dim -> preds / conf
        self.rgb_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="sigmoid", conf_activation="expp1") if enable_rgb else None
        

    def forward(self, input_images, target_intrinsics, target_extrinsics):
        """
        Forward pass for novel view synthesis.
        
        Args:
            input_images (torch.Tensor): Input images with shape [B, S_in, 3, H, W]
                where S_in is the number of input views (typically 4)
            target_intrinsics (torch.Tensor): Target view intrinsics with shape [B, S_out, 3, 3]
                where S_out is the number of target views to synthesize
            target_extrinsics (torch.Tensor): Target view extrinsics with shape [B, S_out, 3, 4]
                in OpenCV convention (camera from world)
        
        Returns:
            torch.Tensor: Synthesized RGB images for target views with shape [B, S_out, H, W, 3]
        """ 
        # If without batch dimension, add it
        if len(input_images.shape) == 4:
            input_images = input_images.unsqueeze(0)
            
        if len(target_intrinsics.shape) == 3:
            target_intrinsics = target_intrinsics.unsqueeze(0)
            target_extrinsics = target_extrinsics.unsqueeze(0)
            
        B, S_in, C, H, W = input_images.shape
        _, S_out, _, _ = target_intrinsics.shape
            
        # Generate Plücker rays for target views
        plucker_rays_list = []
        for b in range(B):
            batch_rays = []
            for s in range(S_out):
                rays = generate_plucker_rays(
                    height=H,
                    width=W,
                    intrinsics=target_intrinsics[b, s],
                    extrinsics=target_extrinsics[b, s]
                )
                batch_rays.append(plucker_rays_to_image(rays))
            plucker_rays_list.append(torch.stack(batch_rays, dim=0))
        
        plucker_images = torch.stack(plucker_rays_list, dim=0)  # (B, S_out, 6, H, W)
        output_images = torch.zeros((B, S_out, 3, H, W), device=input_images.device)
        
        # Encode Plücker rays for target views
        target_tokens = self.plucker_encoder(plucker_images) # (B, S_out, P, C)

        # Process input images and target tokens together through AA transformer
        aggregated_tokens_list, patch_start_idx = self.aggregator(input_images, target_tokens)
        
        # Extract only target frame patch tokens for RGB head
        aggregated_tokens_list = [layer[:, S_in:, :, :] for layer in aggregated_tokens_list]
        
        predictions = {}

        with torch.cuda.amp.autocast(enabled=False):
            if self.rgb_head is not None:
                # images:                  [B, S_out, 3, H, W]
                # aggregated_tokens_list[i]:  [B, S_out, N_tokens, C]
                rgb, rgb_conf = self.rgb_head(
                    aggregated_tokens_list, images=output_images, patch_start_idx=patch_start_idx
                )
                predictions["rgb"] = rgb  # (B, S_out, H, W, 3)
                predictions["rgb_conf"] = rgb_conf  # (B, S_out, H, W)

        if not self.training:
            predictions["images"] = input_images  # store the images for visualization during inference

        return predictions

    