# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import os.path as osp
import json
import logging
import random
import numpy as np
import cv2
from PIL import Image

import torch
from torch.utils.data import Dataset

from data.base_dataset import BaseDataset
from data.dataset_util import *
from vggt.utils.plucker_rays import generate_plucker_rays, plucker_rays_to_image


class ObjaverseDataset(BaseDataset):
    """
    Dataset for training VGGT on Objaverse-like 3D object data for novel view synthesis.
    
    Expected data structure:
        OBJAVERSE_DIR/
            object_id_1/
                images/
                    000.png
                    001.png
                    ...
                cameras.json  # Contains intrinsics and extrinsics for each view
            object_id_2/
                ...
    
    The cameras.json format:
        {
            "000": {
                "intrinsics": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                "extrinsics": [[r11, r12, r13, tx], [r21, r22, r23, ty], [r31, r32, r33, tz]]
            },
            ...
        }
    """
    
    def __init__(
        self,
        common_conf,
        split: str = "train",
        OBJAVERSE_DIR: str = None,
        num_input_views: int = 4,
        num_target_views: int = 1,
        min_num_images: int = 10,
        len_train: int = 100000,
        len_test: int = 10000,
    ):
        """
        Initialize the ObjaverseDataset.
        
        Args:
            common_conf: Configuration object with common settings
            split (str): Dataset split, either 'train' or 'test'
            OBJAVERSE_DIR (str): Directory path to Objaverse data
            num_input_views (int): Number of input views (default: 4, as in paper)
            num_target_views (int): Number of target views to synthesize (default: 1)
            min_num_images (int): Minimum number of images per object
            len_train (int): Length of the training dataset
            len_test (int): Length of the test dataset
        """
        super().__init__(common_conf=common_conf)
        
        self.debug = common_conf.debug
        self.training = common_conf.training
        self.num_input_views = num_input_views
        self.num_target_views = num_target_views
        
        if OBJAVERSE_DIR is None:
            raise ValueError("OBJAVERSE_DIR must be specified.")
        
        self.OBJAVERSE_DIR = OBJAVERSE_DIR
        self.min_num_images = min_num_images
        
        if split == "train":
            self.len_train = len_train
            self.split = "train"
        elif split == "test":
            self.len_train = len_test
            self.split = "test"
        else:
            raise ValueError(f"Invalid split: {split}")
        
        # Load object list
        self.object_list = self._load_object_list()
        
        logging.info(f"Loaded {len(self.object_list)} objects from {OBJAVERSE_DIR}")
    
    def _load_object_list(self):
        """Load list of valid objects from the dataset directory."""
        object_list = []
        
        if not osp.exists(self.OBJAVERSE_DIR):
            logging.error(f"Objaverse directory not found: {self.OBJAVERSE_DIR}")
            return object_list
        
        # List all subdirectories as potential objects
        for obj_id in os.listdir(self.OBJAVERSE_DIR):
            obj_dir = osp.join(self.OBJAVERSE_DIR, obj_id)
            if not osp.isdir(obj_dir):
                continue
            
            # Check if it has required files
            images_dir = osp.join(obj_dir, "images")
            cameras_file = osp.join(obj_dir, "cameras.json")
            
            if not osp.exists(images_dir) or not osp.exists(cameras_file):
                continue
            
            # Check minimum number of images
            image_files = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            if len(image_files) < self.min_num_images:
                continue
            
            object_list.append(obj_id)
        
        # Split into train/test (80/20 split)
        random.seed(42)  # For reproducibility
        random.shuffle(object_list)
        split_idx = int(len(object_list) * 0.8)
        
        if self.split == "train":
            return object_list[:split_idx]
        else:
            return object_list[split_idx:]
    
    def get_data(self, seq_index=None, seq_name=None, ids=None, aspect_ratio=1.0):
        """
        Get training data for a specific object.
        
        Args:
            seq_index (int): Index of the object in the list
            seq_name (str): Name of the object (not used if seq_index is provided)
            ids (list): Specific image IDs to load (not used, we sample randomly)
            aspect_ratio (float): Target aspect ratio
        
        Returns:
            dict: Dictionary containing:
                - images: Input images [S_in, 3, H, W]
                - target_images: Target view images [S_out, 3, H, W]
                - target_plucker_rays: Plücker rays for target views [S_out, 6, H, W]
                - target_intrinsics: Intrinsics for target views [S_out, 3, 3]
                - target_extrinsics: Extrinsics for target views [S_out, 3, 4]
        """
        # Select object
        if seq_index is not None:
            obj_id = self.object_list[seq_index % len(self.object_list)]
        elif seq_name is not None:
            obj_id = seq_name
        else:
            obj_id = random.choice(self.object_list)
        
        obj_dir = osp.join(self.OBJAVERSE_DIR, obj_id)
        images_dir = osp.join(obj_dir, "images")
        cameras_file = osp.join(obj_dir, "cameras.json")
        
        # Load camera data
        with open(cameras_file, 'r') as f:
            cameras_data = json.load(f)
        
        # Get list of available views
        available_views = sorted(cameras_data.keys())
        
        # Sample input and target views
        total_views_needed = self.num_input_views + self.num_target_views
        if len(available_views) < total_views_needed:
            # If not enough views, sample with replacement
            sampled_views = random.choices(available_views, k=total_views_needed)
        else:
            sampled_views = random.sample(available_views, k=total_views_needed)
        
        input_view_ids = sampled_views[:self.num_input_views]
        target_view_ids = sampled_views[self.num_input_views:]
        
        # Load and process images
        input_images = []
        target_images = []
        target_intrinsics = []
        target_extrinsics = []
        
        # Get target shape
        target_H, target_W = self.get_target_shape(aspect_ratio)
        
        # Load input views
        for view_id in input_view_ids:
            img_path = osp.join(images_dir, f"{view_id}.png")
            if not osp.exists(img_path):
                img_path = osp.join(images_dir, f"{view_id}.jpg")
            
            img = Image.open(img_path).convert('RGB')
            img = img.resize((target_W, target_H), Image.LANCZOS)
            img = np.array(img).astype(np.float32) / 255.0
            input_images.append(img)
        
        # Load target views
        for view_id in target_view_ids:
            img_path = osp.join(images_dir, f"{view_id}.png")
            if not osp.exists(img_path):
                img_path = osp.join(images_dir, f"{view_id}.jpg")
            
            img = Image.open(img_path).convert('RGB')
            img = img.resize((target_W, target_H), Image.LANCZOS)
            img = np.array(img).astype(np.float32) / 255.0
            target_images.append(img)
            
            # Get camera parameters for target view
            cam_data = cameras_data[view_id]
            intrinsic = np.array(cam_data['intrinsics'], dtype=np.float32)
            extrinsic = np.array(cam_data['extrinsics'], dtype=np.float32)
            
            # Scale intrinsics if image was resized
            orig_H, orig_W = img.shape[:2]  # After resize
            # Assuming original cameras were calibrated for a different size
            # This would need to be adjusted based on your actual data
            
            target_intrinsics.append(intrinsic)
            target_extrinsics.append(extrinsic)
        
        # Convert to tensors and stack
        input_images = torch.from_numpy(np.stack(input_images, axis=0))  # (S_in, H, W, 3)
        input_images = input_images.permute(0, 3, 1, 2)  # (S_in, 3, H, W)
        
        target_images = torch.from_numpy(np.stack(target_images, axis=0))  # (S_out, H, W, 3)
        target_images = target_images.permute(0, 3, 1, 2)  # (S_out, 3, H, W)
        
        target_intrinsics = torch.from_numpy(np.stack(target_intrinsics, axis=0))  # (S_out, 3, 3)
        target_extrinsics = torch.from_numpy(np.stack(target_extrinsics, axis=0))  # (S_out, 3, 4)
        
        # Generate Plücker rays for target views
        target_plucker_rays = []
        for i in range(self.num_target_views):
            rays = generate_plucker_rays(
                height=target_H,
                width=target_W,
                intrinsics=target_intrinsics[i],
                extrinsics=target_extrinsics[i]
            )
            rays_img = plucker_rays_to_image(rays)
            target_plucker_rays.append(rays_img)
        
        target_plucker_rays = torch.stack(target_plucker_rays, dim=0)  # (S_out, 6, H, W)
        
        return {
            'images': input_images,
            'target_images': target_images,
            'target_plucker_rays': target_plucker_rays,
            'target_intrinsics': target_intrinsics,
            'target_extrinsics': target_extrinsics,
        }
