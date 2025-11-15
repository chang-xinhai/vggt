# Feed-forward Novel View Synthesis - Reproduction Explanation

## Overview

This document explains how we reproduced the Feed-forward Novel View Synthesis experiment from the VGGT paper (Section on "Feed-forward Novel View Synthesis"). This implementation follows the approach described in the paper as closely as possible.

## Background

The paper states:

> "Feed-forward Novel View Synthesis is progressing rapidly. Most existing methods take images with known camera parameters as input and predict the target image corresponding to a new camera viewpoint. Instead of relying on an explicit 3D representation, we follow LVSM and modify VGGT to directly output the target image. However, we do not assume known camera parameters for the input frames."

## Key Implementation Components

### 1. Plücker Ray Encoding (`vggt/utils/plucker_rays.py`)

**Purpose**: Represent target viewpoints using Plücker coordinates.

**Implementation Details**:
- Plücker coordinates represent 3D lines using 6 parameters:
  - Direction vector (3D): ray direction in world coordinates
  - Moment vector (3D): cross product of camera origin with direction
- Function `generate_plucker_rays()` creates Plücker rays for each pixel given camera intrinsics and extrinsics
- Function `plucker_rays_to_image()` converts to CNN-friendly format (6, H, W)

**Mathematical Formulation**:
```
For each pixel (u, v):
1. Convert to normalized camera coordinates using intrinsics
2. Transform to world space using camera-to-world transformation
3. Normalize ray direction: d = normalize(rays_world)
4. Compute moment: m = camera_origin × d
5. Plücker coordinates: [d, m] (6D vector)
```

### 2. Plücker Encoder (`vggt/heads/nvs_head.py` - PluckerEncoder)

**Purpose**: "Use a convolutional layer to encode their Plücker ray images into tokens"

**Implementation Details**:
- Convolutional layer: Conv2d(in_channels=6, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)
- Input: Plücker ray images (B, S_target, 6, H, W)
- Output: Tokens (B, S_target, num_patches, embed_dim)
- Compatible with DINO token format from the aggregator

**Design Choice**: Simple convolutional projection as stated in the paper, initialized with Xavier uniform.

### 3. RGB Regression Head (`vggt/heads/nvs_head.py` - RGBRegressionHead)

**Purpose**: "A DPT head is used to regress the RGB colors for the target views"

**Implementation Details**:
- Based on DPT (Dense Prediction Transformer) architecture
- Multi-scale feature fusion with refinement blocks
- Output: 3-channel RGB images with sigmoid activation (range [0, 1])
- Uses intermediate features from layers [4, 11, 17, 23] of the AA transformer

**Architecture**:
```
Input: Aggregated tokens from AA transformer
→ Project to multi-scale features
→ Feature fusion blocks (pyramid)
→ Refinement network
→ Output head (Conv layers + Sigmoid)
→ Output: RGB image (H, W, 3)
```

### 4. VGGT-NVS Model (`vggt/models/vggt_nvs.py`)

**Purpose**: Main model combining all components.

**Implementation Details**:
- Uses existing VGGT aggregator for input image encoding (DINO tokens)
- Input images: 4 views (as mentioned in paper) - NO camera parameters required
- Target views: Represented as Plücker rays
- Token concatenation: Input tokens + Target Plücker tokens
- Processing: Through AA (Alternating Attention) transformer
- Output: RGB images for target views

**Forward Pass**:
```
1. Encode input images → DINO tokens (via aggregator)
2. Generate Plücker rays for target views
3. Encode Plücker rays → tokens (via PluckerEncoder)
4. Concatenate input and target tokens
5. Process through AA transformer (existing aggregator)
6. Regress RGB colors (via RGBRegressionHead)
```

### 5. Dataset (`training/data/datasets/objaverse.py`)

**Purpose**: Dataset loader for Objaverse-like data.

**Implementation Details**:
- Expects structure: object_id/images/*.png + cameras.json
- Samples 4 input views + 1 target view per iteration
- Generates Plücker rays on-the-fly for target views
- Returns: input images, target images, target Plücker rays, camera parameters

**Note**: The paper mentions training on an internal dataset "approximately 20% the size of Objaverse". Our implementation is designed to work with Objaverse-style data structure.

### 6. Training Configuration (`training/config/nvs_default.yaml`)

**Key Settings**:
- Number of input views: 4 (as stated in paper)
- Frozen aggregator: Yes (transfer learning from pre-trained VGGT)
- Learning rate: 1e-4 with warmup and cosine decay
- Loss: RGB reconstruction (L1) + optional perceptual loss (VGG)
- Batch size: Adjusted for memory (16 images per GPU)

### 7. Loss Function (`training/loss_nvs.py`)

**Components**:
- RGB Loss: L1 or L2 distance between predicted and target RGB
- Perceptual Loss (optional): VGG-based perceptual loss using features from relu1_2, relu2_2, relu3_3
- PSNR: Logged for monitoring (not used for optimization)

**Implementation**:
```python
loss_rgb = L1(pred_rgb, target_rgb)
loss_perceptual = VGG_loss(pred_rgb, target_rgb)
total_loss = w_rgb * loss_rgb + w_perceptual * loss_perceptual
```

### 8. Evaluation Script (`eval_nvs_gso.py`)

**Purpose**: Evaluate on GSO (Google Scanned Objects) dataset as mentioned in paper.

**Metrics**:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)

**Protocol**:
- Use 4 input views
- Synthesize novel target views
- Compare with ground truth
- Report mean and standard deviation

## Key Design Choices

### 1. No Input Camera Parameters
As explicitly stated in the paper:
> "We do not input the Plücker rays for the source images. Hence, the model is not given the camera parameters for these input frames."

**Implementation**: Only target views use Plücker rays; input images are encoded directly with DINO without any camera information.

### 2. Transfer Learning
The paper mentions following LVSM's protocol and likely starting from a pre-trained VGGT model.

**Implementation**: 
- Load pre-trained VGGT aggregator
- Freeze aggregator weights during training
- Only train new components (Plücker encoder + RGB head)

### 3. Plücker Ray Representation
Following LVSM and other novel view synthesis works, we use Plücker coordinates which provide a canonical 6D representation of 3D rays.

**Advantages**:
- Permutation invariant
- Handles parallel rays
- Geometric consistency
- Works well with CNNs

## Training Protocol

1. **Data Preparation**:
   - Organize data in Objaverse-like structure
   - Ensure cameras.json contains intrinsics and extrinsics

2. **Pre-training**:
   - Start from pre-trained VGGT checkpoint
   - Aggregator already trained on reconstruction tasks

3. **Fine-tuning**:
   - Freeze aggregator
   - Train Plücker encoder and RGB head
   - Use 4 input views + 1 target view
   - Optimize RGB reconstruction loss

4. **Evaluation**:
   - Test on GSO dataset
   - Report PSNR and SSIM
   - Compare with LVSM baseline

## Differences from Original Experiment

### Known Differences
1. **Dataset Size**: We provide code for Objaverse, but the paper uses an internal dataset (~20% of Objaverse)
2. **Exact Architecture Details**: Some implementation details (layer normalization, initialization) are our best interpretation
3. **Training Hyperparameters**: Some details not specified in paper (e.g., exact learning rate schedule)

### Maintained Consistency
1. **4 Input Views**: As specified in paper
2. **Plücker Ray Encoding**: Following LVSM approach
3. **No Input Camera Parameters**: Key design choice maintained
4. **DPT Head**: Using DPT-style architecture for RGB regression
5. **AA Transformer**: Using existing VGGT aggregator
6. **GSO Evaluation**: Target dataset for evaluation

## Expected Results

According to the paper (Table 7), on GSO dataset:
- Model achieves "competitive results" compared to LVSM
- Despite not using input camera parameters
- Despite using less training data (~20% of Objaverse)

The paper suggests:
> "We expect that better results would be obtained using a larger training dataset."

## Usage

### Training
```bash
# 1. Prepare Objaverse-like dataset
# 2. Update paths in training/config/nvs_default.yaml
# 3. Run training
torchrun --nproc_per_node=4 training/launch.py --config-name nvs_default
```

### Evaluation
```bash
python eval_nvs_gso.py \
    --gso_dir /path/to/gso \
    --checkpoint /path/to/checkpoint.pt \
    --num_input_views 4 \
    --num_target_views 1
```

## References

- VGGT Paper: "Visual Geometry Grounded Transformer"
- LVSM: Referenced for training protocol
- Objaverse: Dataset structure inspiration
- GSO: Evaluation dataset
- DPT: "Vision Transformers for Dense Prediction"
- Plücker Coordinates: Classical 3D line representation

## Conclusion

This implementation reproduces the Feed-forward Novel View Synthesis experiment from VGGT as faithfully as possible based on the paper description. The key innovation is performing novel view synthesis WITHOUT requiring camera parameters for input frames, while using Plücker rays only for target viewpoints. The implementation uses standard components (DINO encoding, AA transformer, DPT head) combined in the specific way described in the paper.
