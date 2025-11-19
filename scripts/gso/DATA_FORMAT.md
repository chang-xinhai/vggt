# GSO Dataset Structure Example

This document shows the expected directory structure and data format for the GSO dataset after rendering.

## Directory Structure

```
GSO_rendered/
├── object_001/
│   ├── images/
│   │   ├── 000.png          # Elevation 0°, Azimuth 0°
│   │   ├── 001.png          # Elevation 0°, Azimuth 22.5°
│   │   ├── 002.png          # Elevation 0°, Azimuth 45°
│   │   ├── ...
│   │   ├── 015.png          # Elevation 0°, Azimuth 337.5°
│   │   ├── 016.png          # Elevation 20°, Azimuth 0°
│   │   ├── 017.png          # Elevation 20°, Azimuth 22.5°
│   │   ├── 018.png          # Elevation 20°, Azimuth 45° ← Input view 1
│   │   ├── 019.png          # Elevation 20°, Azimuth 67.5°
│   │   ├── 020.png          # Elevation 20°, Azimuth 90°
│   │   ├── 021.png          # Elevation 20°, Azimuth 112.5°
│   │   ├── 022.png          # Elevation 20°, Azimuth 135° ← Input view 2
│   │   ├── ...
│   │   ├── 026.png          # Elevation 20°, Azimuth 225° ← Input view 3
│   │   ├── ...
│   │   ├── 030.png          # Elevation 20°, Azimuth 315° ← Input view 4
│   │   ├── 031.png          # Elevation 20°, Azimuth 337.5°
│   │   ├── 032.png          # Elevation 40°, Azimuth 0°
│   │   ├── ...
│   │   ├── 047.png          # Elevation 40°, Azimuth 337.5°
│   │   ├── 048.png          # Elevation 60°, Azimuth 0°
│   │   ├── ...
│   │   └── 063.png          # Elevation 60°, Azimuth 337.5°
│   └── cameras.json
├── object_002/
│   └── ...
└── ...
```

## cameras.json Format

```json
{
  "000": {
    "intrinsics": [
      [576.0, 0.0, 256.0],
      [0.0, 576.0, 256.0],
      [0.0, 0.0, 1.0]
    ],
    "extrinsics": [
      [1.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, -1.0, 2.0],
      [0.0, 1.0, 0.0, 0.0]
    ],
    "azimuth": 0.0,
    "elevation": 0.0,
    "distance": 2.0
  },
  "001": {
    "intrinsics": [
      [576.0, 0.0, 256.0],
      [0.0, 576.0, 256.0],
      [0.0, 0.0, 1.0]
    ],
    "extrinsics": [
      [0.9239, 0.0, 0.3827, -0.7654],
      [0.0, 0.0, -1.0, 2.0],
      [-0.3827, 1.0, 0.0, 0.0]
    ],
    "azimuth": 22.5,
    "elevation": 0.0,
    "distance": 2.0
  },
  ...
  "018": {
    "intrinsics": [
      [576.0, 0.0, 256.0],
      [0.0, 576.0, 256.0],
      [0.0, 0.0, 1.0]
    ],
    "extrinsics": [
      [0.6533, 0.0, 0.7572, -1.3285],
      [0.2588, 0.9397, -0.2233, -0.6840],
      [-0.7115, 0.3420, 0.6139, 0.0]
    ],
    "azimuth": 45.0,
    "elevation": 20.0,
    "distance": 2.0
  },
  ...
}
```

## View ID Mapping

Views are numbered sequentially by elevation then azimuth:

```
view_id = elevation_index * 16 + azimuth_index

Elevations: [0°, 20°, 40°, 60°] → indices [0, 1, 2, 3]
Azimuths: [0°, 22.5°, 45°, ..., 337.5°] → indices [0, 1, 2, ..., 15]
```

### Examples:

| View ID | Elevation | Azimuth | Description |
|---------|-----------|---------|-------------|
| 000     | 0°        | 0°      | First view (elevation 0°) |
| 015     | 0°        | 337.5°  | Last view at elevation 0° |
| 016     | 20°       | 0°      | First view at elevation 20° |
| 018     | 20°       | 45°     | **Input view 1** |
| 022     | 20°       | 135°    | **Input view 2** |
| 026     | 20°       | 225°    | **Input view 3** |
| 030     | 20°       | 315°    | **Input view 4** |
| 031     | 20°       | 337.5°  | Last view at elevation 20° |
| 032     | 40°       | 0°      | First view at elevation 40° |
| 048     | 60°       | 0°      | First view at elevation 60° |
| 063     | 60°       | 337.5°  | Last view (final) |

## Input Views for Evaluation

Following the Instant3D protocol:
- **4 structured input views** at elevation 20° with azimuths [45°, 135°, 225°, 315°]
- These correspond to view IDs: **[018, 022, 026, 030]**

## Target Views for Evaluation

Following the LVSM protocol:
- **10 random target views** sampled from the remaining 60 views
- Excludes the 4 input views
- Random seed 42 for reproducibility

## Camera Coordinate System

The camera matrices follow OpenCV convention:

### Intrinsic Matrix K (3×3)

```
K = [fx  0  cx]
    [0  fy  cy]
    [0   0   1]
```

Where:
- `fx`, `fy`: Focal lengths in pixels
- `cx`, `cy`: Principal point (image center)

### Extrinsic Matrix [R|t] (3×4)

```
[R|t] = [r11 r12 r13 tx]
        [r21 r22 r23 ty]
        [r31 r32 r33 tz]
```

Where:
- `R` (3×3): Rotation matrix (world to camera)
- `t` (3×1): Translation vector (world to camera)

**Coordinate system**:
- +X: Right
- +Y: Down
- +Z: Forward (camera looking direction)

## Image Format

- **Format**: PNG with RGBA channels
- **Resolution**: 512 × 512 pixels
- **Color space**: sRGB
- **Alpha channel**: Transparent background (for masking if needed)
- **Value range**: [0, 255] uint8

## Example Loading Code

```python
import json
import numpy as np
from PIL import Image
from pathlib import Path

# Load object data
gso_dir = Path("/path/to/GSO_rendered")
obj_name = "object_001"
obj_dir = gso_dir / obj_name

# Load cameras
with open(obj_dir / "cameras.json") as f:
    cameras = json.load(f)

# Load specific view
view_id = "018"  # Input view 1
img = Image.open(obj_dir / "images" / f"{view_id}.png").convert('RGB')
img_array = np.array(img).astype(np.float32) / 255.0  # Normalize to [0, 1]

# Get camera parameters
intrinsics = np.array(cameras[view_id]["intrinsics"], dtype=np.float32)
extrinsics = np.array(cameras[view_id]["extrinsics"], dtype=np.float32)
azimuth = cameras[view_id]["azimuth"]
elevation = cameras[view_id]["elevation"]

print(f"View {view_id}: {elevation}° elevation, {azimuth}° azimuth")
print(f"Image shape: {img_array.shape}")
print(f"Intrinsics shape: {intrinsics.shape}")
print(f"Extrinsics shape: {extrinsics.shape}")
```

## Validation Checklist

After rendering, verify:

- [ ] Each object has an `images/` directory with 64 PNG files
- [ ] Images are named `000.png` through `063.png`
- [ ] Each object has a `cameras.json` file
- [ ] `cameras.json` contains 64 entries with keys "000" through "063"
- [ ] Each camera entry has `intrinsics`, `extrinsics`, `azimuth`, `elevation`, `distance`
- [ ] Images are 512×512 pixels
- [ ] Objects appear centered and properly scaled
- [ ] No rendering artifacts or missing textures

## Quick Validation Script

```bash
# Count objects
echo "Number of objects: $(ls -d GSO_rendered/*/ | wc -l)"

# Check random object
OBJ="GSO_rendered/object_001"
echo "Images: $(ls $OBJ/images/*.png | wc -l)"
echo "Camera entries: $(python -c "import json; print(len(json.load(open('$OBJ/cameras.json'))))")"

# Verify image dimensions
python -c "from PIL import Image; img = Image.open('$OBJ/images/000.png'); print(f'Image size: {img.size}')"
```
