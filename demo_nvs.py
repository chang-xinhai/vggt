#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Simple demo script for Feed-forward Novel View Synthesis with VGGT.

This script demonstrates how to use the VGGT-NVS model to synthesize novel views
from a set of input images.
"""

import argparse
import torch
import numpy as np
from PIL import Image
import os

from vggt.models.vggt_nvs import VGGT_NVS
from vggt.utils.load_fn import load_and_preprocess_images


def load_camera_params_from_json(camera_json_path, view_id):
    """Load camera parameters from JSON file."""
    import json
    with open(camera_json_path, 'r') as f:
        cameras = json.load(f)
    
    cam_data = cameras[view_id]
    intrinsics = np.array(cam_data['intrinsics'], dtype=np.float32)
    extrinsics = np.array(cam_data['extrinsics'], dtype=np.float32)
    
    return intrinsics, extrinsics


def create_default_cameras(num_views=1, img_size=518):
    """Create default camera parameters for testing."""
    # Default intrinsics (pinhole camera)
    intrinsics = np.eye(3, dtype=np.float32)
    intrinsics[0, 0] = img_size * 0.8  # fx
    intrinsics[1, 1] = img_size * 0.8  # fy
    intrinsics[0, 2] = img_size / 2    # cx
    intrinsics[1, 2] = img_size / 2    # cy
    
    # Default extrinsics (camera looking at origin from distance)
    extrinsics = np.eye(3, 4, dtype=np.float32)
    extrinsics[2, 3] = 3.0  # Move camera back along z-axis
    
    # Replicate for multiple views
    intrinsics_batch = np.stack([intrinsics] * num_views, axis=0)
    extrinsics_batch = np.stack([extrinsics] * num_views, axis=0)
    
    return intrinsics_batch, extrinsics_batch


def main():
    parser = argparse.ArgumentParser(
        description='Feed-forward Novel View Synthesis Demo'
    )
    parser.add_argument(
        '--input_images', 
        type=str, 
        nargs='+', 
        required=True,
        help='Paths to 4 input images'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='synthesized_view.png',
        help='Output path for synthesized view'
    )
    parser.add_argument(
        '--checkpoint', 
        type=str, 
        help='Path to model checkpoint (optional, uses untrained model if not provided)'
    )
    parser.add_argument(
        '--target_camera_json',
        type=str,
        help='Path to JSON file containing target camera parameters'
    )
    parser.add_argument(
        '--target_view_id',
        type=str,
        default='000',
        help='View ID in the camera JSON file to use as target'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (cuda or cpu)'
    )
    
    args = parser.parse_args()
    
    # Validate input
    if len(args.input_images) != 4:
        print(f"Error: Expected 4 input images, got {len(args.input_images)}")
        print("This implementation uses 4 input views as described in the paper")
        return 1
    
    print("=" * 60)
    print("VGGT Feed-forward Novel View Synthesis Demo")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Input images: {len(args.input_images)}")
    print(f"Output: {args.output}")
    
    # Initialize model
    print("\nInitializing model...")
    model = VGGT_NVS(img_size=518, patch_size=14, embed_dim=1024)
    
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        print("Warning: No checkpoint provided. Using untrained model.")
        print("Results will not be meaningful without a trained checkpoint.")
    
    model = model.to(args.device)
    model.eval()
    
    # Load input images
    print("\nLoading input images...")
    input_images = load_and_preprocess_images(args.input_images)
    input_images = input_images.unsqueeze(0).to(args.device)  # Add batch dimension
    print(f"Input images shape: {input_images.shape}")
    
    # Load or create target camera parameters
    print("\nSetting up target camera...")
    if args.target_camera_json:
        print(f"Loading camera from {args.target_camera_json}, view {args.target_view_id}")
        intrinsics, extrinsics = load_camera_params_from_json(
            args.target_camera_json, 
            args.target_view_id
        )
        intrinsics = intrinsics[np.newaxis, np.newaxis, :, :]  # (1, 1, 3, 3)
        extrinsics = extrinsics[np.newaxis, np.newaxis, :, :]  # (1, 1, 3, 4)
    else:
        print("Using default camera parameters")
        intrinsics, extrinsics = create_default_cameras(num_views=1, img_size=518)
        intrinsics = intrinsics[np.newaxis, :, :, :]  # (1, 1, 3, 3)
        extrinsics = extrinsics[np.newaxis, :, :, :]  # (1, 1, 3, 4)
    
    target_intrinsics = torch.from_numpy(intrinsics).to(args.device)
    target_extrinsics = torch.from_numpy(extrinsics).to(args.device)
    
    # Synthesize novel view
    print("\nSynthesizing novel view...")
    with torch.no_grad():
        rgb_output = model(input_images, target_intrinsics, target_extrinsics)
    
    print(f"Output shape: {rgb_output.shape}")
    
    # Save output
    print(f"\nSaving to {args.output}...")
    output_image = rgb_output[0, 0].cpu().numpy()  # (H, W, 3)
    output_image = (output_image * 255).astype(np.uint8)
    output_pil = Image.fromarray(output_image)
    output_pil.save(args.output)
    
    print("✓ Done!")
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    exit(main())
