# Feed-forward Novel View Synthesis with VGGT

This directory contains the implementation of Feed-forward Novel View Synthesis as described in the VGGT paper. This approach synthesizes novel views without requiring camera parameters for input frames, following the methodology of LVSM.

## Overview

The implementation modifies VGGT to directly output target images instead of relying on explicit 3D representations:
- Takes 4 input views (encoded with DINO)
- Takes target viewpoints (represented as Plücker rays)
- Processes through AA transformer
- Outputs RGB images for target views using DPT head

**Key Innovation**: No camera parameters required for input frames!

## Quick Start

### 1. Installation

```bash
# Install VGGT
pip install -e .

# Install additional dependencies for training
pip install fvcore omegaconf hydra-core
```

### 2. Prepare Dataset

Organize your dataset in Objaverse-like structure:
```
DATASET_DIR/
  object_id_1/
    images/
      000.png
      001.png
      ...
    cameras.json
  object_id_2/
    ...
```

The `cameras.json` format:
```json
{
  "000": {
    "intrinsics": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
    "extrinsics": [[r11, r12, r13, tx], [r21, r22, r23, ty], [r31, r32, r33, tz]]
  },
  ...
}
```

### 3. Training

```bash
# Update paths in training/config/nvs_default.yaml
# Then run training
cd training
torchrun --nproc_per_node=4 launch.py --config-name nvs_default
```

### 4. Evaluation on GSO

```bash
python eval_nvs_gso.py \
    --gso_dir /path/to/gso \
    --checkpoint /path/to/checkpoint.pt \
    --num_input_views 4 \
    --num_target_views 1 \
    --output gso_results.json
```

## Usage Example

```python
import torch
from vggt.models.vggt_nvs import VGGT_NVS
from vggt.utils.load_fn import load_and_preprocess_images

device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize model
model = VGGT_NVS(img_size=518, patch_size=14, embed_dim=1024)
model.load_state_dict(torch.load("checkpoint.pt"))
model = model.to(device)
model.eval()

# Load input images (4 views)
input_image_paths = ["view1.png", "view2.png", "view3.png", "view4.png"]
input_images = load_and_preprocess_images(input_image_paths).to(device)
input_images = input_images.unsqueeze(0)  # Add batch dimension

# Define target view camera parameters
target_intrinsics = torch.tensor([...]).unsqueeze(0).to(device)  # (1, 1, 3, 3)
target_extrinsics = torch.tensor([...]).unsqueeze(0).to(device)  # (1, 1, 3, 4)

# Synthesize novel view
with torch.no_grad():
    rgb_output = model(input_images, target_intrinsics, target_extrinsics)

# Save output
output_image = rgb_output[0, 0].cpu().numpy()  # (H, W, 3)
```

## Files

- `vggt/utils/plucker_rays.py` - Plücker ray generation
- `vggt/heads/nvs_head.py` - Plücker encoder and RGB head
- `vggt/models/vggt_nvs.py` - Main NVS model
- `training/data/datasets/objaverse.py` - Dataset loader
- `training/config/nvs_default.yaml` - Training config
- `training/loss_nvs.py` - Loss function
- `eval_nvs_gso.py` - GSO evaluation script
- `test_nvs_components.py` - Component tests

## Documentation

- **[Reproduction Explanation](docs/NVS_Reproduction_Explanation.md)**: Detailed explanation of how the implementation reproduces the paper
- **[Verification Checklist](docs/NVS_Verification_Checklist.md)**: Comprehensive to-do list for verification

## Architecture Details

### Plücker Ray Encoding
Plücker coordinates represent 3D rays using 6 parameters:
- Direction vector (3D): normalized ray direction in world space
- Moment vector (3D): cross product of camera origin with direction

### Model Architecture
```
Input Images (4 views)
    ↓
DINO Encoding (via VGGT Aggregator)
    ↓
Input Tokens

Target Viewpoints
    ↓
Plücker Ray Images (6 channels)
    ↓
Convolutional Encoder
    ↓
Target Tokens

Input Tokens + Target Tokens
    ↓
AA Transformer (Alternating Attention)
    ↓
DPT Head (RGB Regression)
    ↓
RGB Images for Target Views
```

## Training Configuration

Key hyperparameters:
- Number of input views: 4
- Learning rate: 1e-4 with warmup and cosine decay
- Batch size: 16 images per GPU
- Loss: RGB reconstruction (L1) + optional perceptual (VGG)
- Freeze: VGGT aggregator (transfer learning)
- Train: Plücker encoder + RGB head only

## Expected Results

According to the paper (Table 7), on GSO dataset:
- Achieves competitive results compared to LVSM
- Despite not using input camera parameters
- Despite using less training data (~20% of Objaverse)

## Testing

Run component tests:
```bash
python test_nvs_components.py
```

All tests should pass:
- ✓ Plücker ray generation
- ✓ Plücker encoder
- ✓ RGB regression head
- ✓ VGGT-NVS model structure

## Citation

If you use this implementation, please cite the VGGT paper:

```bibtex
@inproceedings{wang2025vggt,
  title={VGGT: Visual Geometry Grounded Transformer},
  author={Wang, Jianyuan and Chen, Minghao and Karaev, Nikita and Vedaldi, Andrea and Rupprecht, Christian and Novotny, David},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```

## License

See the [LICENSE](../LICENSE.txt) file for details.
