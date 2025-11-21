# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np


def generate_plucker_rays(height, width, intrinsics, extrinsics):
    """
    Generate Plücker ray coordinates for a camera view.
    
    Plücker coordinates represent 3D lines using 6 parameters:
    - Direction vector (3D): the ray direction in world coordinates
    - Moment vector (3D): the cross product of a point on the line with the direction
    
    Args:
        height (int): Image height in pixels
        width (int): Image width in pixels
        intrinsics (torch.Tensor): Camera intrinsic matrix of shape (3, 3) or (B, 3, 3)
        extrinsics (torch.Tensor): Camera extrinsic matrix of shape (3, 4) or (B, 3, 4)
            In OpenCV convention (camera from world)
    
    Returns:
        torch.Tensor: Plücker rays of shape (H, W, 6) or (B, H, W, 6)
            where the last dimension contains [direction (3), moment (3)]
    """
    device = intrinsics.device
    dtype = intrinsics.dtype
    
    # Check if batched
    is_batched = len(intrinsics.shape) == 3
    if not is_batched:
        intrinsics = intrinsics.unsqueeze(0)
        extrinsics = extrinsics.unsqueeze(0)
    
    batch_size = intrinsics.shape[0]
    
    # Create pixel coordinates grid
    i, j = torch.meshgrid(
        torch.arange(height, dtype=dtype, device=device),
        torch.arange(width, dtype=dtype, device=device),
        indexing='ij'
    )
    
    # Pixel coordinates in homogeneous form (H, W, 3)
    pixels = torch.stack([j, i, torch.ones_like(i)], dim=-1)  # (H, W, 3)
    pixels = pixels.unsqueeze(0).expand(batch_size, -1, -1, -1)  # (B, H, W, 3)
    
    # Get camera parameters
    fx = intrinsics[:, 0, 0]  # (B,)
    fy = intrinsics[:, 1, 1]  # (B,)
    cx = intrinsics[:, 0, 2]  # (B,)
    cy = intrinsics[:, 1, 2]  # (B,)
    
    # Convert to normalized camera coordinates
    # x_cam = (u - cx) / fx, y_cam = (v - cy) / fy, z_cam = 1
    x_cam = (pixels[..., 0] - cx[:, None, None]) / fx[:, None, None]
    y_cam = (pixels[..., 1] - cy[:, None, None]) / fy[:, None, None]
    z_cam = torch.ones_like(x_cam)
    
    # Ray directions in camera space (not normalized yet)
    rays_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)  # (B, H, W, 3)
    
    # Extract rotation and translation from extrinsics
    R_c2w = extrinsics[:, :3, :3]  # Camera to world rotation (B, 3, 3)
    t_c2w = extrinsics[:, :3, 3]   # Camera to world translation (B, 3)
    
    # Convert from camera-from-world to world-from-camera
    # If extrinsic is camera-from-world: T_cw, then world-from-camera is T_wc = T_cw^-1
    R_w2c = R_c2w.transpose(-2, -1)  # (B, 3, 3)
    t_w2c = -torch.bmm(R_w2c, t_c2w.unsqueeze(-1)).squeeze(-1)  # (B, 3)
    
    # Camera origin in world coordinates (the translation of world-from-camera)
    camera_origin = t_w2c  # (B, 3)
    
    # Transform ray directions to world space
    rays_world = torch.einsum('bij,bhwj->bhwi', R_w2c, rays_cam)  # (B, H, W, 3)
    
    # Normalize ray directions
    rays_world = rays_world / (torch.norm(rays_world, dim=-1, keepdim=True) + 1e-8)  # (B, H, W, 3)
    
    # Compute moment: m = o × d (cross product of origin and direction)
    # camera_origin: (B, 3) -> (B, 1, 1, 3)
    origin_expanded = camera_origin[:, None, None, :]  # (B, 1, 1, 3)
    moments = torch.cross(origin_expanded.expand_as(rays_world), rays_world, dim=-1)  # (B, H, W, 3)
    
    # Concatenate direction and moment to form Plücker coordinates
    plucker_rays = torch.cat([rays_world, moments], dim=-1)  # (B, H, W, 6)
    
    if not is_batched:
        plucker_rays = plucker_rays.squeeze(0)
    
    return plucker_rays


def plucker_rays_to_image(plucker_rays):
    """
    Convert Plücker rays from (H, W, 6) to (6, H, W) for use as CNN input.
    
    Args:
        plucker_rays (torch.Tensor): Plücker rays of shape (H, W, 6) or (B, H, W, 6)
    
    Returns:
        torch.Tensor: Plücker rays reshaped to (6, H, W) or (B, 6, H, W)
    """
    if len(plucker_rays.shape) == 3:
        # (H, W, 6) -> (6, H, W)
        return plucker_rays.permute(2, 0, 1)
    else:
        # (B, H, W, 6) -> (B, 6, H, W)
        return plucker_rays.permute(0, 3, 1, 2)
