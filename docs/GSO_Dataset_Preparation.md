# GSO Dataset Preparation Guide

This guide provides detailed instructions for preparing the Google Scanned Objects (GSO) dataset for Novel View Synthesis (NVS) evaluation with VGGT.

## Table of Contents

1. [Overview](#overview)
2. [Dataset Specifications](#dataset-specifications)
3. [Prerequisites](#prerequisites)
4. [Step-by-Step Guide](#step-by-step-guide)
5. [Data Format](#data-format)
6. [Evaluation Protocol](#evaluation-protocol)
7. [Troubleshooting](#troubleshooting)

## Overview

The GSO dataset rendering pipeline follows the exact specifications from Instant3D, GS-LRM, and LVSM papers to ensure fair comparison with state-of-the-art methods.

### Key Features

- **Standardized camera views**: 64 views following Instant3D protocol
- **Structured evaluation**: 4 input views + 10 target views per object
- **High-quality rendering**: 512×512 resolution with proper lighting
- **Automatic normalization**: Objects scaled to [-1, 1]³ bounding box
- **Compatible format**: Direct integration with `eval_nvs_gso.py`

## Dataset Specifications

### Camera Configuration (Following Instant3D)

```
Total Views: 64
├── Elevation 0°: 16 views (azimuth: 0°, 22.5°, ..., 337.5°)
├── Elevation 20°: 16 views (azimuth: 0°, 22.5°, ..., 337.5°)
├── Elevation 40°: 16 views (azimuth: 0°, 22.5°, ..., 337.5°)
└── Elevation 60°: 16 views (azimuth: 0°, 22.5°, ..., 337.5°)
```

**Parameters**:
- Resolution: 512 × 512 pixels
- Camera distance: 2.0 units from object center
- Focal length: 35mm (sensor width: 32mm)
- Lighting: Uniform white (0.5 intensity)
- Background: Transparent (RGBA)

### Object Normalization (Following GS-LRM)

Each 3D object is:
1. **Centered**: Moved to origin (0, 0, 0)
2. **Scaled**: Fit within [-1, 1]³ bounding box
3. **Preserved aspect ratio**: Uniform scaling in all dimensions

### Evaluation Protocol (Following LVSM)

For each object:
- **4 input views**: Elevation 20°, azimuths [45°, 135°, 225°, 315°]
  - View IDs: [011, 013, 015, 017] (assuming 16 azimuths per elevation)
- **10 target views**: Randomly sampled from remaining 60 views
- **Metrics**: PSNR, SSIM, LPIPS

## Prerequisites

### 1. Software Requirements

#### Blender 4.x (Required)

```bash
# Download from official website
https://www.blender.org/download/

# Verify installation
blender --version
# Expected output: Blender 4.x.x

# Add to PATH (if needed)
# Linux/Mac:
export PATH="/path/to/blender:$PATH"

# Windows:
# Add to System Environment Variables
```

#### Python Packages

```bash
# Already included in VGGT requirements
pip install numpy tqdm pillow
```

### 2. Download GSO Dataset

#### Option A: Official Download Script (Recommended)

```bash
# Clone GSO-Data-Utils
git clone https://github.com/TO-Hitori/GSO-Data-Utils.git
cd GSO-Data-Utils

# Install dependencies
pip install -r requirements.txt

# Download dataset (this will take a while)
python download_collection.py -d "./GSO_data"

# Resume interrupted download
python download_collection.py -d "./GSO_data" -c [last_successful_id]
```

#### Option B: Google Drive (Alternative)

```bash
# Download from provided Google Drive link
# https://drive.google.com/drive/folders/1Dtqiyt0QP9dabiaTN5qONdb8avc0aNg6
```

### 3. Convert to GLB Format

GSO objects are provided as OBJ files with textures. Convert them to GLB format for Blender:

```bash
cd GSO-Data-Utils

# Batch convert all objects
python obj2glb_batch.py \
    --input_path "./GSO_data" \
    --output_path "./GSO_GLB"
```

This creates a `GSO_GLB` folder with one `.glb` file per object.

## Step-by-Step Guide

### Step 1: Setup Environment

```bash
# Navigate to VGGT repository
cd /path/to/vggt

# Verify Blender is installed
blender --version

# Check Python environment
python -c "import numpy, tqdm; print('Dependencies OK')"
```

### Step 2: Prepare GLB Files

Ensure you have converted GSO objects to GLB format (see Prerequisites).

```bash
# Your GLB folder should look like:
GSO_GLB/
  ├── object_001.glb
  ├── object_002.glb
  └── ...
```

### Step 3: Test Single Object Rendering

Before batch processing, test with a single object:

```bash
blender --background --python scripts/gso/render_gso_dataset.py -- \
    --glb_path /path/to/GSO_GLB/test_object.glb \
    --out_path /path/to/test_output
```

Expected output:
```
test_output/
  test_object/
    images/
      000.png, 001.png, ..., 063.png  # 64 images
    cameras.json
```

Verify:
- All 64 images are generated
- Images look correct (centered, proper scale)
- cameras.json contains all view parameters

### Step 4: Batch Render All Objects

```bash
# Sequential processing (safer, easier to debug)
python scripts/gso/batch_render_gso.py \
    --glb_folder /path/to/GSO_GLB \
    --out_path /path/to/GSO_rendered

# Parallel processing (faster)
python scripts/gso/batch_render_gso.py \
    --glb_folder /path/to/GSO_GLB \
    --out_path /path/to/GSO_rendered \
    --num_workers 4
```

**Expected time**:
- Per object: 2-5 minutes (depends on complexity, GPU)
- Full dataset (~1030 objects): 34-86 hours (sequential) or 8-22 hours (4 workers)

### Step 5: Verify Rendered Dataset

```bash
# Check number of rendered objects
ls /path/to/GSO_rendered | wc -l

# Verify structure of a random object
ls -R /path/to/GSO_rendered/object_001/

# Check cameras.json
cat /path/to/GSO_rendered/object_001/cameras.json | head -20
```

### Step 6: Run Evaluation

```bash
python eval_nvs_gso.py \
    --gso_dir /path/to/GSO_rendered \
    --checkpoint /path/to/vggt_nvs_checkpoint.pt \
    --num_input_views 4 \
    --num_target_views 10 \
    --output gso_results.json
```

## Data Format

### Directory Structure

```
GSO_rendered/
├── object_001/
│   ├── images/
│   │   ├── 000.png  # Elevation 0°, Azimuth 0°
│   │   ├── 001.png  # Elevation 0°, Azimuth 22.5°
│   │   ├── ...
│   │   ├── 015.png  # Elevation 0°, Azimuth 337.5°
│   │   ├── 016.png  # Elevation 20°, Azimuth 0°
│   │   ├── ...
│   │   └── 063.png  # Elevation 60°, Azimuth 337.5°
│   └── cameras.json
├── object_002/
│   └── ...
└── ...
```

### cameras.json Schema

```json
{
  "000": {
    "intrinsics": [
      [fx, 0, cx],
      [0, fy, cy],
      [0, 0, 1]
    ],
    "extrinsics": [
      [r11, r12, r13, tx],
      [r21, r22, r23, ty],
      [r31, r32, r33, tz]
    ],
    "azimuth": 0.0,
    "elevation": 0.0,
    "distance": 2.0
  },
  "001": {
    ...
  },
  ...
}
```

**Field descriptions**:
- `intrinsics`: 3×3 camera intrinsic matrix (OpenCV convention)
  - `fx`, `fy`: Focal lengths in pixels
  - `cx`, `cy`: Principal point (image center)
- `extrinsics`: 3×4 camera extrinsic matrix (world-to-camera transformation)
  - Rotation matrix (3×3) + translation vector (3×1)
  - OpenCV convention (camera looks down -Z axis)
- `azimuth`: Azimuth angle in degrees [0, 360)
- `elevation`: Elevation angle in degrees [-90, 90]
- `distance`: Camera distance from object center

### View ID Mapping

Views are numbered sequentially by elevation then azimuth:

```python
view_id = elevation_idx * num_azimuths + azimuth_idx

# Example: elevation 20° (idx=1), azimuth 45° (idx=2)
# view_id = 1 * 16 + 2 = 18
```

### Input View Selection (LVSM Protocol)

```python
# 4 input views at elevation 20°
input_azimuths = [45, 135, 225, 315]  # degrees
input_view_ids = [
    1 * 16 + 2,   # view 018: elevation 20°, azimuth 45°
    1 * 16 + 6,   # view 022: elevation 20°, azimuth 135°
    1 * 16 + 10,  # view 026: elevation 20°, azimuth 225°
    1 * 16 + 14,  # view 030: elevation 20°, azimuth 315°
]
```

## Evaluation Protocol

### Metrics

1. **PSNR (Peak Signal-to-Noise Ratio)**
   - Measures reconstruction quality
   - Higher is better
   - Typical range: 20-40 dB

2. **SSIM (Structural Similarity Index)**
   - Measures perceptual similarity
   - Range: [0, 1], higher is better
   - Typical range: 0.8-0.99

3. **LPIPS (Learned Perceptual Image Patch Similarity)**
   - Deep learning-based perceptual metric
   - Lower is better
   - Typical range: 0.01-0.3

### Evaluation Settings

- **Input views**: 4 (elevation 20°, specific azimuths)
- **Target views**: 10 (randomly sampled from remaining 60)
- **Random seed**: 42 (for reproducibility)
- **Aggregation**: Mean across all objects

### Running Evaluation

```bash
# Full evaluation on all objects
python eval_nvs_gso.py \
    --gso_dir /path/to/GSO_rendered \
    --checkpoint /path/to/checkpoint.pt \
    --output gso_results.json

# Results are saved in gso_results.json:
{
  "psnr_mean": 25.34,
  "psnr_std": 2.15,
  "ssim_mean": 0.892,
  "ssim_std": 0.043,
  "num_samples": 10300  # 1030 objects × 10 views
}
```

## Troubleshooting

### Issue: Blender not found

```bash
# Check if Blender is in PATH
which blender  # Linux/Mac
where blender  # Windows

# If not found, specify path explicitly
python scripts/gso/batch_render_gso.py \
    --blender_path /path/to/blender/blender \
    ...
```

### Issue: Out of memory (GPU)

```bash
# Reduce rendering quality
python scripts/gso/batch_render_gso.py \
    --samples 64 \  # Instead of default 128
    --resolution 256 \  # Instead of default 512
    ...
```

### Issue: Objects appear too small/large

```bash
# Adjust camera distance
blender --background --python scripts/gso/render_gso_dataset.py -- \
    --distance 2.5 \  # Increase if objects appear too large
    ...
```

### Issue: Rendering fails for specific objects

```bash
# Skip problematic objects and continue
python scripts/gso/batch_render_gso.py \
    --skip_existing \
    ...

# Or render specific range
python scripts/gso/batch_render_gso.py \
    --start_index 100 \
    --max_objects 50 \
    ...
```

### Issue: Evaluation fails

```bash
# Check data format
python -c "
import json
from pathlib import Path

gso_dir = Path('/path/to/GSO_rendered')
obj_dir = gso_dir / 'object_001'

# Check structure
assert (obj_dir / 'images').exists(), 'images/ missing'
assert (obj_dir / 'cameras.json').exists(), 'cameras.json missing'

# Check cameras.json
with open(obj_dir / 'cameras.json') as f:
    cameras = json.load(f)
    print(f'Views: {len(cameras)}')
    print(f'First view: {list(cameras.keys())[0]}')
    print(f'Keys: {list(cameras[\"000\"].keys())}')
"
```

## Performance Optimization

### Speed up rendering

1. **Use parallel processing**:
   ```bash
   --num_workers 8  # Use 8 parallel Blender instances
   ```

2. **Reduce quality** (if acceptable):
   ```bash
   --samples 64  # Instead of 128
   ```

3. **Use GPU** (if available):
   - GPU rendering is enabled by default
   - Ensure CUDA is properly configured

### Reduce storage

1. **Compress images** (after rendering):
   ```bash
   # Use PNG compression
   find GSO_rendered -name "*.png" -exec optipng {} \;
   ```

2. **Remove intermediate files**:
   - Keep only images/ and cameras.json
   - Remove Blender cache files

## References

- **GSO Dataset**: [Google Scanned Objects](https://goo.gle/scanned-objects)
- **GSO Paper**: [arXiv:2204.11918](https://arxiv.org/abs/2204.11918)
- **GSO-Data-Utils**: [GitHub Repository](https://github.com/TO-Hitori/GSO-Data-Utils)
- **Instant3D**: Li et al., 2023
- **GS-LRM**: Zhang et al., 2024
- **VGGT**: Wang et al., CVPR 2025

## Support

For questions or issues:
1. Check this documentation
2. Review the [scripts/gso/README.md](scripts/gso/README.md)
3. Open an issue on GitHub
