# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F


class PluckerEncoder(nn.Module):
    """
    Encoder for Plücker ray coordinates into tokens.
    
    This module takes Plücker ray images (6 channels representing ray direction and moment)
    and encodes them into tokens compatible with the VGGT aggregator.
    
    Args:
        img_size (int): Input image size (height/width, assumes square)
        patch_size (int): Patch size for tokenization
        embed_dim (int): Output embedding dimension
        in_channels (int): Number of input channels (6 for Plücker coordinates)
    """
    
    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        in_channels=6,
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        
        # Calculate number of patches
        self.num_patches_per_side = img_size // patch_size
        self.num_patches = self.num_patches_per_side ** 2
        
        # Convolutional layer to encode Plücker rays into tokens
        # Following the paper: "we use a convolutional layer to encode their Plücker ray images into tokens"
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize the convolutional layer weights."""
        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0)
    
    def forward(self, plucker_images):
        """
        Forward pass to encode Plücker ray images into tokens.
        
        Args:
            plucker_images (torch.Tensor): Plücker ray images with shape (B, S, 6, H, W)
                where B is batch size, S is number of target views,
                6 is the Plücker coordinate dimension (direction + moment)
        
        Returns:
            torch.Tensor: Encoded tokens with shape (B, S, N, D)
                where N is the number of patches and D is the embedding dimension
        """
        B, S, C, H, W = plucker_images.shape
        
        # Reshape to process all views together
        plucker_flat = plucker_images.view(B * S, C, H, W)
        
        # Apply convolutional projection
        x = self.proj(plucker_flat)  # (B*S, embed_dim, num_patches_h, num_patches_w)
        
        # Flatten spatial dimensions to get tokens
        x = x.flatten(2)  # (B*S, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B*S, num_patches, embed_dim)
        
        # Reshape back to batch and sequence dimensions
        x = x.view(B, S, -1, self.embed_dim)  # (B, S, num_patches, embed_dim)
        
        return x
