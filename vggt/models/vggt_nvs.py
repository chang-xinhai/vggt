# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from vggt.models.aggregator import Aggregator, NVSAggregator
from vggt.heads.nvs_head import PluckerEncoder, RGBRegressionHead
from vggt.utils.plucker_rays import generate_plucker_rays, plucker_rays_to_image


class VGGT_NVS(nn.Module, PyTorchModelHubMixin):
    """
    VGGT model for Feed-forward Novel View Synthesis.
    
    Following the approach described in the VGGT paper:
    - Takes 4 input images (encoded with DINO)
    - Takes target view Plücker rays (encoded with convolutional layer)
    - Processes concatenated tokens through AA transformer
    - Outputs RGB images for target views using DPT head
    
    Key difference from standard VGGT: Does NOT require camera parameters for input frames.
    
    Args:
        img_size (int): Image size (default: 518)
        patch_size (int): Patch size (default: 14)
        embed_dim (int): Embedding dimension (default: 1024)
    """
    
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # NVS-specific aggregator that processes fused input and target tokens
        self.aggregator = NVSAggregator(
            img_size=img_size, 
            patch_size=patch_size, 
            embed_dim=embed_dim
        )
        
        # Plücker ray encoder for target views
        self.plucker_encoder = PluckerEncoder(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            in_channels=6,  # Plücker coordinates have 6 dimensions
        )
        
        # RGB regression head using DPT architecture
        self.rgb_head = RGBRegressionHead(
            dim_in=2 * embed_dim,  # Aggregator outputs 2x embed_dim
            patch_size=patch_size,
            img_size=img_size,
        )
    
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
        
        # Encode Plücker rays for target views
        target_tokens = self.plucker_encoder(plucker_images)  # (B, S_out, N_patches, D)
        
        # Process input images and target tokens together through AA transformer
        # The NVSAggregator concatenates them before AA transformer processing
        combined_tokens_list, patch_start_idx = self.aggregator(input_images, target_tokens)
        
        # Extract only target frame PATCH tokens for RGB head (no special tokens)
        # Source and target tokens embed different information:
        # - Source tokens: from real images (visual content)
        # - Target tokens: from Plücker rays (viewpoint info)
        # RGB head should focus on decoding target tokens only
        
        # Calculate patches per frame from image dimensions
        patches_per_frame = (H // self.patch_size) ** 2
        
        # Extract patches for each target frame
        target_tokens_list = []
        for tokens in combined_tokens_list:
            # Extract patch tokens for all target frames
            frame_patches = []
            for frame_idx in range(S_in, S_in + S_out):
                start = patch_start_idx[frame_idx]
                # End is start + patches_per_frame (NOT patch_start_idx[frame_idx + 1])
                end = start + patches_per_frame
                frame_patches.append(tokens[:, start:end, :])
            # Concatenate all target frame patches: [B, S_out*patches, 2C]
            target_tokens_list.append(torch.cat(frame_patches, dim=1))
        
        # Create patch indices relative to 0 for target frames only
        target_patch_start_idx = [i * patches_per_frame for i in range(S_out + 1)]
        
        # Regress RGB colors for target views only
        rgb_target = self.rgb_head(target_tokens_list, target_patch_start_idx)
        
        return rgb_target
    
    def forward_with_separate_encoding(self, input_images, target_plucker_images):
        """
        Alternative forward pass that takes pre-computed Plücker ray images.
        
        This is useful when Plücker rays are pre-computed during data loading.
        
        Args:
            input_images (torch.Tensor): Input images with shape [B, S_in, 3, H, W]
            target_plucker_images (torch.Tensor): Plücker ray images with shape [B, S_out, 6, H, W]
        
        Returns:
            torch.Tensor: Synthesized RGB images for target views with shape [B, S_out, H, W, 3]
        """
        B, S_in, C, H, W = input_images.shape
        _, S_out, _, _, _ = target_plucker_images.shape
        
        # Encode Plücker rays for target views
        target_tokens = self.plucker_encoder(target_plucker_images)  # (B, S_out, N_patches, D)
        
        # Process input images and target tokens together through AA transformer
        # The NVSAggregator concatenates them before AA transformer processing
        combined_tokens_list, patch_start_idx = self.aggregator(input_images, target_tokens)
        
        # Extract only target frame PATCH tokens for RGB head (no special tokens)
        # Source and target tokens embed different information:
        # - Source tokens: from real images (visual content)
        # - Target tokens: from Plücker rays (viewpoint info)
        # RGB head should focus on decoding target tokens only
        
        # Calculate patches per frame from image dimensions
        patches_per_frame = (H // self.patch_size) ** 2
        
        # Extract patches for each target frame
        target_tokens_list = []
        for tokens in combined_tokens_list:
            # Extract patch tokens for all target frames
            frame_patches = []
            for frame_idx in range(S_in, S_in + S_out):
                start = patch_start_idx[frame_idx]
                # End is start + patches_per_frame (NOT patch_start_idx[frame_idx + 1])
                end = start + patches_per_frame
                frame_patches.append(tokens[:, start:end, :])
            # Concatenate all target frame patches: [B, S_out*patches, 2C]
            target_tokens_list.append(torch.cat(frame_patches, dim=1))
        
        # Create patch indices relative to 0 for target frames only
        target_patch_start_idx = [i * patches_per_frame for i in range(S_out + 1)]
        
        # Regress RGB colors for target views only
        rgb_target = self.rgb_head(target_tokens_list, target_patch_start_idx)
        
        return rgb_target
