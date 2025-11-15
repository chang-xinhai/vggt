# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass(eq=False)
class NovelViewSynthesisLoss(nn.Module):
    """
    Loss module for novel view synthesis training.
    
    Combines RGB reconstruction loss with optional perceptual loss.
    """
    
    def __init__(self, rgb=None, perceptual=None, **kwargs):
        super().__init__()
        self.rgb = rgb
        self.perceptual = perceptual
        
        # Initialize perceptual loss network if needed
        if perceptual is not None and perceptual.get('use_vgg', False):
            self.vgg = VGGPerceptualLoss()
        else:
            self.vgg = None
    
    def forward(self, predictions, batch):
        """
        Compute the total novel view synthesis loss.
        
        Args:
            predictions: Dict containing model predictions
                - 'rgb': Predicted RGB images (B, S_out, H, W, 3)
            batch: Dict containing ground truth data
                - 'target_images': Ground truth target images (B, S_out, 3, H, W)
        
        Returns:
            Dict containing individual losses and total objective
        """
        total_loss = 0
        loss_dict = {}
        
        # RGB reconstruction loss
        pred_rgb = predictions['rgb']  # (B, S_out, H, W, 3)
        target_rgb = batch['target_images']  # (B, S_out, 3, H, W)
        
        # Permute target to match prediction format
        target_rgb = target_rgb.permute(0, 1, 3, 4, 2)  # (B, S_out, H, W, 3)
        
        # Compute RGB loss
        if self.rgb['loss_type'] == 'l1':
            rgb_loss = F.l1_loss(pred_rgb, target_rgb)
        elif self.rgb['loss_type'] == 'l2':
            rgb_loss = F.mse_loss(pred_rgb, target_rgb)
        else:
            raise ValueError(f"Unknown loss type: {self.rgb['loss_type']}")
        
        rgb_loss = rgb_loss * self.rgb['weight']
        loss_dict['loss_rgb'] = rgb_loss
        total_loss = total_loss + rgb_loss
        
        # PSNR metric (for logging, not for optimization)
        with torch.no_grad():
            mse = F.mse_loss(pred_rgb, target_rgb)
            psnr = -10 * torch.log10(mse + 1e-8)
            loss_dict['loss_psnr'] = psnr
        
        # Perceptual loss (optional)
        if self.vgg is not None:
            # Reshape for VGG: (B*S_out, 3, H, W)
            B, S_out, H, W, C = pred_rgb.shape
            pred_rgb_vgg = pred_rgb.permute(0, 1, 4, 2, 3).reshape(B * S_out, C, H, W)
            target_rgb_vgg = target_rgb.permute(0, 1, 4, 2, 3).reshape(B * S_out, C, H, W)
            
            perceptual_loss = self.vgg(pred_rgb_vgg, target_rgb_vgg)
            perceptual_loss = perceptual_loss * self.perceptual['weight']
            
            loss_dict['loss_perceptual'] = perceptual_loss
            total_loss = total_loss + perceptual_loss
        
        loss_dict['objective'] = total_loss
        
        return loss_dict


class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG features.
    
    Measures perceptual similarity between images using intermediate
    features from a pre-trained VGG network.
    """
    
    def __init__(self):
        super().__init__()
        # Use torchvision's VGG16
        try:
            import torchvision.models as models
            vgg = models.vgg16(pretrained=True).features
        except:
            # Fallback if torchvision not available
            import torchvision.models as models
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        
        # Extract specific layers for perceptual loss
        self.slice1 = nn.Sequential(*list(vgg.children())[:4])   # relu1_2
        self.slice2 = nn.Sequential(*list(vgg.children())[4:9])  # relu2_2
        self.slice3 = nn.Sequential(*list(vgg.children())[9:16]) # relu3_3
        
        # Freeze VGG parameters
        for param in self.parameters():
            param.requires_grad = False
        
        # VGG normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def forward(self, pred, target):
        """
        Compute perceptual loss.
        
        Args:
            pred (torch.Tensor): Predicted images (B, 3, H, W) in range [0, 1]
            target (torch.Tensor): Target images (B, 3, H, W) in range [0, 1]
        
        Returns:
            torch.Tensor: Perceptual loss value
        """
        # Normalize inputs
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std
        
        # Extract features
        pred_feat1 = self.slice1(pred)
        pred_feat2 = self.slice2(pred_feat1)
        pred_feat3 = self.slice3(pred_feat2)
        
        target_feat1 = self.slice1(target)
        target_feat2 = self.slice2(target_feat1)
        target_feat3 = self.slice3(target_feat2)
        
        # Compute L1 loss on features
        loss = 0
        loss += F.l1_loss(pred_feat1, target_feat1)
        loss += F.l1_loss(pred_feat2, target_feat2)
        loss += F.l1_loss(pred_feat3, target_feat3)
        
        return loss
