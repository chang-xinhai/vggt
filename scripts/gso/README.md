# GSO Dataset Rendering Scripts

This directory contains scripts for rendering the Google Scanned Objects (GSO) dataset following the Instant3D/GS-LRM/LVSM protocol for novel view synthesis evaluation.

## Overview

These scripts implement the GSO dataset preparation pipeline described in the VGGT paper, following the exact specifications from:
- **Instant3D**: 64 views at 4 elevations (0°, 20°, 40°, 60°) × 16 azimuths per elevation
- **GS-LRM**: Objects centered and scaled to [-1, 1]³ bounding box
- **LVSM**: 4 structured input views + 10 random target views for evaluation

## Requirements

### Software Dependencies

1. **Blender 4.x** (required for rendering)
   - Download from: https://www.blender.org/download/
   - Add to system PATH or specify path with `--blender_path`
   - Verify installation: `blender --version`

2. **Python packages** (already in requirements.txt)
   - numpy
   - tqdm (for progress bars)

### Dataset Requirements

1. **Download GSO Dataset**
   - Official website: https://goo.gle/scanned-objects
   - Or use the GSO-Data-Utils download script: https://github.com/TO-Hitori/GSO-Data-Utils

2. **Convert to GLB format**
   - Use GSO-Data-Utils conversion scripts:
     ```bash
     git clone https://github.com/TO-Hitori/GSO-Data-Utils.git
     cd GSO-Data-Utils
     python obj2glb_batch.py --input_path "GSO_data" --output_path "GSO_GLB"
     ```

## Quick Start

### 1. Render a Single Object

```bash
blender --background --python scripts/gso/render_gso_dataset.py -- \
    --glb_path /path/to/object.glb \
    --out_path /path/to/output
```

This will create:
```
output/
  object_name/
    images/
      000.png  # elevation=0°, azimuth=0°
      001.png  # elevation=0°, azimuth=22.5°
      ...
      063.png  # elevation=60°, azimuth=337.5°
    cameras.json  # Camera parameters for all views
```

### 2. Batch Render All Objects

```bash
python scripts/gso/batch_render_gso.py \
    --glb_folder /path/to/GSO_GLB \
    --out_path /path/to/GSO_rendered
```

For parallel processing (faster):
```bash
python scripts/gso/batch_render_gso.py \
    --glb_folder /path/to/GSO_GLB \
    --out_path /path/to/GSO_rendered \
    --num_workers 4
```

### 3. Resume Interrupted Rendering

If the batch rendering is interrupted, you can resume:

```bash
python scripts/gso/batch_render_gso.py \
    --glb_folder /path/to/GSO_GLB \
    --out_path /path/to/GSO_rendered \
    --skip_existing
```

Or start from a specific index:
```bash
python scripts/gso/batch_render_gso.py \
    --glb_folder /path/to/GSO_GLB \
    --out_path /path/to/GSO_rendered \
    --start_index 500
```

## Rendering Specifications

### Camera Configuration (Instant3D Protocol)

- **Total views**: 64 (4 elevations × 16 azimuths)
- **Elevations**: 0°, 20°, 40°, 60°
- **Azimuths**: 0°, 22.5°, 45°, ..., 337.5° (evenly distributed)
- **Camera distance**: 2.0 units (default, can be adjusted)
- **Resolution**: 512×512 pixels
- **Samples**: 128 (for quality rendering)

### Input Views for Evaluation (LVSM Protocol)

Following the paper, for evaluation:
- **4 input views**: elevation 20°, azimuths [45°, 135°, 225°, 315°]
  - These are views: 011, 013, 015, 017 (based on 16 azimuths per elevation)
- **10 target views**: randomly sampled from the remaining 60 views

This is handled automatically by `eval_nvs_gso.py`.

## Output Format

### Directory Structure
```
GSO_rendered/
  object_001/
    images/
      000.png, 001.png, ..., 063.png
    cameras.json
  object_002/
    ...
```

### cameras.json Format

```json
{
  "000": {
    "intrinsics": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
    "extrinsics": [[r11, r12, r13, tx], [r21, r22, r23, ty], [r31, r32, r33, tz]],
    "azimuth": 0.0,
    "elevation": 0.0,
    "distance": 2.0
  },
  ...
}
```

- **intrinsics**: 3×3 camera intrinsic matrix (OpenCV convention)
- **extrinsics**: 3×4 camera extrinsic matrix (world-to-camera, OpenCV convention)
- **azimuth**: Azimuth angle in degrees
- **elevation**: Elevation angle in degrees
- **distance**: Camera distance from object center

## Advanced Usage

### Custom Camera Configuration

```bash
blender --background --python scripts/gso/render_gso_dataset.py -- \
    --glb_path object.glb \
    --out_path output \
    --elevations 0 20 40 60 \
    --num_azimuths 16 \
    --distance 2.5 \
    --resolution 1024 \
    --samples 256
```

### Render Subset of Objects

```bash
python scripts/gso/batch_render_gso.py \
    --glb_folder /path/to/GSO_GLB \
    --out_path /path/to/output \
    --start_index 0 \
    --max_objects 100
```

### Performance Tuning

For faster rendering (lower quality):
```bash
python scripts/gso/batch_render_gso.py \
    --glb_folder /path/to/GSO_GLB \
    --out_path /path/to/output \
    --samples 64 \
    --num_workers 8
```

## Evaluation

Once rendering is complete, evaluate the VGGT-NVS model:

```bash
python eval_nvs_gso.py \
    --gso_dir /path/to/GSO_rendered \
    --checkpoint /path/to/checkpoint.pt \
    --num_input_views 4 \
    --num_target_views 10 \
    --output gso_results.json
```

This will:
1. Load each object from the rendered dataset
2. Use 4 structured input views (elevation 20°, specific azimuths)
3. Synthesize 10 random target views
4. Compute PSNR, SSIM, and LPIPS metrics
5. Save results to JSON file

## Troubleshooting

### Blender not found
```bash
# Linux/Mac: Add to PATH
export PATH="/path/to/blender:$PATH"

# Or specify explicitly
python batch_render_gso.py --blender_path /path/to/blender/blender ...
```

### Out of memory (GPU)
- Reduce `--samples` (e.g., 64 or 32)
- Reduce `--resolution` (e.g., 256)
- Use CPU rendering (slower): modify `use_gpu=False` in script

### Rendering is slow
- Use parallel processing: `--num_workers 4` (or more)
- Reduce quality: `--samples 64`
- Use GPU if available (default)

### Objects appear incorrect
- Check normalization: Objects should fit in [-1, 1]³ box
- Verify camera distance is appropriate for your objects
- Adjust `--distance` parameter if needed

## GSO Dataset Information

- **Total objects**: ~1,030 objects
- **Format**: High-quality 3D scans with textures
- **License**: Various (check individual object licenses)
- **Paper**: [Google Scanned Objects: A High-Quality Dataset of 3D Scanned Household Items](https://arxiv.org/abs/2204.11918)

## References

1. **VGGT**: Visual Geometry Grounded Transformer (Wang et al., CVPR 2025)
2. **Instant3D**: Instant Text-to-3D Generation (Li et al., 2023)
3. **GS-LRM**: Large Reconstruction Model for 3D Gaussian Splatting (Zhang et al., 2024)
4. **LVSM**: Large View Synthesis Models (various)
5. **GSO**: Google Scanned Objects (Downs et al., 2022)

## Contact

For issues or questions, please open an issue on the GitHub repository.
