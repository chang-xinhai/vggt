# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Evaluation script for Feed-forward Novel View Synthesis on GSO dataset.

This script evaluates the VGGT-NVS model on the Google Scanned Objects (GSO) dataset,
following the evaluation protocol from LVSM.
"""

import os
import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm

from vggt.models.vggt_nvs import VGGT_NVS
from vggt.utils.plucker_rays import generate_plucker_rays, plucker_rays_to_image


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_gso_data(gso_dir, object_name, view_ids):
    """
    Load images and camera parameters for a GSO object.
    
    Args:
        gso_dir (str): Path to GSO dataset directory
        object_name (str): Name of the object
        view_ids (list): List of view IDs to load
    
    Returns:
        tuple: (images, intrinsics, extrinsics)
    """
    obj_dir = os.path.join(gso_dir, object_name)
    images_dir = os.path.join(obj_dir, "images")
    cameras_file = os.path.join(obj_dir, "cameras.json")
    
    # Load camera data
    with open(cameras_file, 'r') as f:
        cameras_data = json.load(f)
    
    images = []
    intrinsics = []
    extrinsics = []
    
    for view_id in view_ids:
        # Load image
        img_path = os.path.join(images_dir, f"{view_id}.png")
        if not os.path.exists(img_path):
            img_path = os.path.join(images_dir, f"{view_id}.jpg")
        
        img = Image.open(img_path).convert('RGB')
        img = img.resize((518, 518), Image.LANCZOS)
        img = np.array(img).astype(np.float32) / 255.0
        images.append(img)
        
        # Load camera parameters
        cam_data = cameras_data[view_id]
        intrinsic = np.array(cam_data['intrinsics'], dtype=np.float32)
        extrinsic = np.array(cam_data['extrinsics'], dtype=np.float32)
        
        intrinsics.append(intrinsic)
        extrinsics.append(extrinsic)
    
    images = np.stack(images, axis=0)
    intrinsics = np.stack(intrinsics, axis=0)
    extrinsics = np.stack(extrinsics, axis=0)
    
    return images, intrinsics, extrinsics


def compute_psnr(pred, target):
    """Compute PSNR between predicted and target images."""
    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return 100.0
    return -10 * np.log10(mse)


def compute_ssim(pred, target):
    """
    Compute SSIM between predicted and target images.
    
    Simple implementation - for production use, consider using skimage.metrics.structural_similarity
    """
    # Convert to grayscale for SSIM
    pred_gray = np.mean(pred, axis=-1)
    target_gray = np.mean(target, axis=-1)
    
    # Constants for SSIM
    C1 = (0.01 * 1.0) ** 2
    C2 = (0.03 * 1.0) ** 2
    
    mu1 = pred_gray.mean()
    mu2 = target_gray.mean()
    
    sigma1_sq = np.var(pred_gray)
    sigma2_sq = np.var(target_gray)
    sigma12 = np.mean((pred_gray - mu1) * (target_gray - mu2))
    
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim


def evaluate_model(model, gso_dir, num_input_views=4, num_target_views=1, device='cuda'):
    """
    Evaluate the model on GSO dataset.
    
    Args:
        model: VGGT_NVS model
        gso_dir (str): Path to GSO dataset
        num_input_views (int): Number of input views
        num_target_views (int): Number of target views to synthesize
        device (str): Device to run evaluation on
    
    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    
    # Get list of objects
    object_list = [d for d in os.listdir(gso_dir) 
                   if os.path.isdir(os.path.join(gso_dir, d))]
    
    logger.info(f"Evaluating on {len(object_list)} objects")
    
    all_psnr = []
    all_ssim = []
    
    with torch.no_grad():
        for obj_name in tqdm(object_list):
            try:
                obj_dir = os.path.join(gso_dir, obj_name)
                cameras_file = os.path.join(obj_dir, "cameras.json")
                
                if not os.path.exists(cameras_file):
                    continue
                
                # Load camera data to get available views
                with open(cameras_file, 'r') as f:
                    cameras_data = json.load(f)
                
                available_views = sorted(cameras_data.keys())
                
                if len(available_views) < num_input_views + num_target_views:
                    continue
                
                # Sample views (deterministic for evaluation)
                np.random.seed(42)
                sampled_views = np.random.choice(
                    available_views, 
                    size=num_input_views + num_target_views,
                    replace=False
                )
                
                input_view_ids = sampled_views[:num_input_views]
                target_view_ids = sampled_views[num_input_views:]
                
                # Load data
                input_images, _, _ = load_gso_data(gso_dir, obj_name, input_view_ids)
                target_images, target_intrinsics, target_extrinsics = load_gso_data(
                    gso_dir, obj_name, target_view_ids
                )
                
                # Convert to tensors
                input_images_tensor = torch.from_numpy(input_images).permute(0, 3, 1, 2).unsqueeze(0).to(device)
                target_intrinsics_tensor = torch.from_numpy(target_intrinsics).unsqueeze(0).to(device)
                target_extrinsics_tensor = torch.from_numpy(target_extrinsics).unsqueeze(0).to(device)
                
                # Run model
                pred_rgb = model(
                    input_images_tensor,
                    target_intrinsics_tensor,
                    target_extrinsics_tensor
                )
                
                # Convert predictions to numpy
                pred_rgb = pred_rgb.squeeze(0).cpu().numpy()  # (S_out, H, W, 3)
                
                # Compute metrics for each target view
                for i in range(num_target_views):
                    pred = pred_rgb[i]
                    target = target_images[i]
                    
                    psnr = compute_psnr(pred, target)
                    ssim = compute_ssim(pred, target)
                    
                    all_psnr.append(psnr)
                    all_ssim.append(ssim)
                
            except Exception as e:
                logger.warning(f"Error processing {obj_name}: {e}")
                continue
    
    # Compute statistics
    metrics = {
        'psnr_mean': np.mean(all_psnr),
        'psnr_std': np.std(all_psnr),
        'ssim_mean': np.mean(all_ssim),
        'ssim_std': np.std(all_ssim),
        'num_samples': len(all_psnr),
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate VGGT-NVS on GSO dataset')
    parser.add_argument('--gso_dir', type=str, required=True, help='Path to GSO dataset')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--num_input_views', type=int, default=4, help='Number of input views')
    parser.add_argument('--num_target_views', type=int, default=1, help='Number of target views')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--output', type=str, default='gso_results.json', help='Output file for results')
    
    args = parser.parse_args()
    
    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    model = VGGT_NVS()
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    model = model.to(args.device)
    
    # Evaluate
    logger.info("Starting evaluation...")
    metrics = evaluate_model(
        model,
        args.gso_dir,
        num_input_views=args.num_input_views,
        num_target_views=args.num_target_views,
        device=args.device
    )
    
    # Print results
    logger.info("Evaluation Results:")
    logger.info(f"  PSNR: {metrics['psnr_mean']:.2f} ± {metrics['psnr_std']:.2f}")
    logger.info(f"  SSIM: {metrics['ssim_mean']:.4f} ± {metrics['ssim_std']:.4f}")
    logger.info(f"  Number of samples: {metrics['num_samples']}")
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()
