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


class RGBRegressionHead(nn.Module):
    """
    RGB regression head for novel view synthesis.
    
    Uses a DPT-style architecture to regress RGB colors for target views.
    This is adapted from the existing DPTHead but outputs RGB values instead of depth/points.
    
    Args:
        dim_in (int): Input dimension from aggregator tokens
        patch_size (int): Patch size used in the aggregator
        img_size (int): Output image size
    """
    
    def __init__(
        self,
        dim_in=2048,
        patch_size=14,
        img_size=518,
        features=256,
        out_channels=[256, 512, 1024, 1024],
        intermediate_layer_idx=[4, 11, 17, 23],
    ):
        super().__init__()
        
        self.patch_size = patch_size
        self.img_size = img_size
        self.intermediate_layer_idx = intermediate_layer_idx
        
        self.norm = nn.LayerNorm(dim_in)
        
        # Projection layers for each output channel from tokens
        self.projects = nn.ModuleList(
            [nn.Conv2d(in_channels=dim_in, out_channels=oc, kernel_size=1, stride=1, padding=0) 
             for oc in out_channels]
        )
        
        # Resize layers for upsampling feature maps
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0], out_channels=out_channels[0], 
                kernel_size=4, stride=4, padding=0
            ),
            nn.ConvTranspose2d(
                in_channels=out_channels[1], out_channels=out_channels[1],
                kernel_size=2, stride=2, padding=0
            ),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3], out_channels=out_channels[3],
                kernel_size=3, stride=2, padding=1
            ),
        ])
        
        # 1x1 convolutions to project to features dimension
        self.scratch_layers = nn.ModuleList([
            nn.Conv2d(out_channels[0], features, kernel_size=1),
            nn.Conv2d(out_channels[1], features, kernel_size=1),
            nn.Conv2d(out_channels[2], features, kernel_size=1),
            nn.Conv2d(out_channels[3], features, kernel_size=1),
        ])
        
        # Refinement blocks
        self.refinenet1 = FeatureFusionBlock(features, activation=nn.ReLU(True))
        self.refinenet2 = FeatureFusionBlock(features, activation=nn.ReLU(True))
        self.refinenet3 = FeatureFusionBlock(features, activation=nn.ReLU(True))
        self.refinenet4 = FeatureFusionBlock(features, activation=nn.ReLU(True))
        
        # Output head for RGB (3 channels)
        self.head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(features // 2, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 3, kernel_size=1),  # Output 3 channels for RGB
            nn.Sigmoid(),  # Ensure output is in [0, 1] range
        )
        
    def forward(self, aggregated_tokens_list, patch_start_idx):
        """
        Forward pass to regress RGB colors for target views.
        
        Args:
            aggregated_tokens_list (list): List of token tensors from each AA transformer layer
            patch_start_idx (list): Starting indices for patches of each frame
        
        Returns:
            torch.Tensor: Predicted RGB images with shape (B, S, H, W, 3)
        """
        # Get intermediate features
        tokens = [aggregated_tokens_list[i] for i in self.intermediate_layer_idx]
        
        batch_size = tokens[0].shape[0]
        num_frames = len(patch_start_idx) - 1
        
        # Process each layer
        layers = []
        for idx, (token, proj, resize, scratch) in enumerate(zip(tokens, self.projects, self.resize_layers, self.scratch_layers)):
            # Normalize tokens
            token = self.norm(token)
            
            # Reshape tokens to spatial format for each frame
            layer_features = []
            for frame_idx in range(num_frames):
                start = patch_start_idx[frame_idx]
                end = patch_start_idx[frame_idx + 1]
                frame_tokens = token[:, start:end, :]  # (B, N_patches, D)
                
                # Calculate spatial dimensions
                num_patches = end - start
                h = w = int(num_patches ** 0.5)
                
                # Reshape to spatial grid
                frame_tokens = frame_tokens.permute(0, 2, 1)  # (B, D, N_patches)
                frame_tokens = frame_tokens.reshape(batch_size, -1, h, w)  # (B, D, h, w)
                
                # Project, resize, and convert to features dimension
                frame_tokens = proj(frame_tokens)
                frame_tokens = resize(frame_tokens)
                frame_tokens = scratch(frame_tokens)  # Project to features dimension
                
                layer_features.append(frame_tokens)
            
            layers.append(layer_features)
        
        # Process each frame through refinement network
        output_images = []
        for frame_idx in range(num_frames):
            # Get features for this frame from all layers
            layer_1_feat = layers[0][frame_idx]
            layer_2_feat = layers[1][frame_idx]
            layer_3_feat = layers[2][frame_idx]
            layer_4_feat = layers[3][frame_idx]
            
            # Refinement network (pyramid fusion)
            path_4 = self.refinenet4(layer_4_feat)
            path_3 = self.refinenet3(path_4, layer_3_feat)
            path_2 = self.refinenet2(path_3, layer_2_feat)
            path_1 = self.refinenet1(path_2, layer_1_feat)
            
            # Generate output
            out = self.head(path_1)  # (B, 3, H, W)
            
            # Interpolate to target size if needed
            if out.shape[-2:] != (self.img_size, self.img_size):
                out = F.interpolate(out, size=(self.img_size, self.img_size), 
                                   mode='bilinear', align_corners=False)
            
            output_images.append(out)
        
        # Stack all frames
        output = torch.stack(output_images, dim=1)  # (B, S, 3, H, W)
        output = output.permute(0, 1, 3, 4, 2)  # (B, S, H, W, 3)
        
        return output


class FeatureFusionBlock(nn.Module):
    """Feature fusion block for the refinement network."""
    
    def __init__(self, features, activation=nn.ReLU(True)):
        super().__init__()
        
        self.block1 = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
            activation,
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
            activation,
        )
        
    def forward(self, *inputs):
        """
        Args:
            inputs: One or more feature tensors to fuse
        """
        # Resize all inputs to the same size (use the size of the first input)
        target_size = inputs[0].shape[-2:]
        
        # Sum all inputs after resizing
        x = inputs[0]
        for inp in inputs[1:]:
            inp_resized = F.interpolate(inp, size=target_size, mode='bilinear', align_corners=False)
            x = x + inp_resized
        
        x = self.block1(x)
        x = self.block2(x)
        
        return x
