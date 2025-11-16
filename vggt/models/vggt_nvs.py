# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from vggt.models.aggregator import Aggregator
from vggt.heads.nvs_head import PluckerEncoder
from vggt.heads.dpt_head import DPTHead
from vggt.utils.plucker_rays import generate_plucker_rays, plucker_rays_to_image


class VGGT_NVS(nn.Module, PyTorchModelHubMixin):
    """
    VGGT model for Feed-forward Novel View Synthesis.
    
    Following the approach described in the VGGT paper:
    - Takes 4 input images (encoded with DINO)
    - Takes target view Plücker rays (encoded with convolutional layer)
    - Concatenates tokens and processes them together through AA transformer
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
        
        # Aggregator for processing tokens with AA transformer
        self.aggregator = Aggregator(
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
        # DPTHead configured for RGB output with feature_only=True
        # We'll add our own output head for RGB
        self.rgb_dpt = DPTHead(
            dim_in=2 * embed_dim,  # Aggregator outputs 2x embed_dim
            patch_size=patch_size,
            output_dim=3,  # Not used when feature_only=True
            features=256,
            out_channels=[256, 512, 1024, 1024],
            intermediate_layer_idx=[4, 11, 17, 23],
            pos_embed=True,
            feature_only=True,  # Get features, we'll add our own RGB head
        )
        
        # Simple output head for RGB
        self.rgb_output = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=1),
            nn.Sigmoid(),  # RGB values in [0, 1]
        )
        
        # Register normalization constants as buffers (for input images)
        _RESNET_MEAN = [0.485, 0.456, 0.406]
        _RESNET_STD = [0.229, 0.224, 0.225]
        self.register_buffer("_resnet_mean", torch.FloatTensor(_RESNET_MEAN).view(1, 1, 3, 1, 1), persistent=False)
        self.register_buffer("_resnet_std", torch.FloatTensor(_RESNET_STD).view(1, 1, 3, 1, 1), persistent=False)
    
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
        
        # Use the alternative forward that takes pre-computed Plücker images
        return self.forward_with_separate_encoding(input_images, plucker_images)
    
    def forward_with_separate_encoding(self, input_images, target_plucker_images):
        """
        Alternative forward pass that takes pre-computed Plücker ray images.
        
        This is the main forward implementation that properly fuses input and target tokens
        before processing through the AA transformer.
        
        Args:
            input_images (torch.Tensor): Input images with shape [B, S_in, 3, H, W]
            target_plucker_images (torch.Tensor): Plücker ray images with shape [B, S_out, 6, H, W]
        
        Returns:
            torch.Tensor: Synthesized RGB images for target views with shape [B, S_out, H, W, 3]
        """
        B, S_in, C, H, W = input_images.shape
        _, S_out, _, _, _ = target_plucker_images.shape
        
        # Normalize input images
        input_images_norm = (input_images - self._resnet_mean) / self._resnet_std
        
        # Encode input images through patch embedding
        # Reshape to [B*S_in, C, H, W] for patch embedding
        input_images_flat = input_images_norm.view(B * S_in, C, H, W)
        input_patch_tokens = self.aggregator.patch_embed(input_images_flat)
        
        if isinstance(input_patch_tokens, dict):
            input_patch_tokens = input_patch_tokens["x_norm_patchtokens"]
        
        # input_patch_tokens: (B*S_in, N_patches, D)
        
        # Encode Plücker rays for target views
        # Reshape to [B*S_out, 6, H, W]
        plucker_flat = target_plucker_images.view(B * S_out, 6, H, W)
        
        # Apply convolutional projection
        target_patch_tokens = self.plucker_encoder.proj(plucker_flat)  # (B*S_out, D, h, w)
        
        # Flatten spatial dimensions to get tokens
        target_patch_tokens = target_patch_tokens.flatten(2)  # (B*S_out, D, N_patches)
        target_patch_tokens = target_patch_tokens.transpose(1, 2)  # (B*S_out, N_patches, D)
        
        # Concatenate input and target tokens along the sequence dimension
        # input_patch_tokens: (B*S_in, N_patches, D)
        # target_patch_tokens: (B*S_out, N_patches, D)
        # We need to concatenate them as (B*(S_in+S_out), N_patches, D)
        
        # First reshape to separate batch dimension
        N_patches = input_patch_tokens.shape[1]
        D = input_patch_tokens.shape[2]
        
        input_patch_tokens = input_patch_tokens.view(B, S_in, N_patches, D)
        target_patch_tokens = target_patch_tokens.view(B, S_out, N_patches, D)
        
        # Concatenate along sequence dimension
        combined_patch_tokens = torch.cat([input_patch_tokens, target_patch_tokens], dim=1)  # (B, S_in+S_out, N_patches, D)
        
        # Flatten back to (B*(S_in+S_out), N_patches, D)
        S_total = S_in + S_out
        combined_patch_tokens = combined_patch_tokens.view(B * S_total, N_patches, D)
        
        # Process through AA transformer using the new method
        aggregated_tokens_list, patch_start_idx = self.aggregator.forward_with_tokens_and_sequence(
            patch_tokens=combined_patch_tokens,
            B=B,
            S=S_total,
            img_size=H,
        )
        
        # Create a "dummy" images tensor for DPT head (just for shape/device information)
        # The DPT head needs the original image tensor for some operations
        dummy_images = torch.cat([input_images, torch.zeros(B, S_out, 3, H, W, device=input_images.device)], dim=1)
        
        # Get DPT features
        dpt_features = self.rgb_dpt(
            aggregated_tokens_list=aggregated_tokens_list,
            images=dummy_images,
            patch_start_idx=patch_start_idx,
        )
        
        # dpt_features shape: (B, S_total, C, H, W) where C=256 (features)
        # Reshape to process all views at once
        B, S_total, C_feat, H_feat, W_feat = dpt_features.shape
        dpt_features_flat = dpt_features.view(B * S_total, C_feat, H_feat, W_feat)
        
        # Apply RGB output head
        rgb_output_flat = self.rgb_output(dpt_features_flat)  # (B*S_total, 3, H_feat, W_feat)
        
        # Reshape back
        rgb_output = rgb_output_flat.view(B, S_total, 3, H_feat, W_feat)
        
        # Extract only the target views (last S_out views)
        rgb_target = rgb_output[:, S_in:, :, :, :]  # (B, S_out, 3, H, W)
        
        # Convert to (B, S_out, H, W, 3) format
        rgb_target = rgb_target.permute(0, 1, 3, 4, 2)
        
        return rgb_target
