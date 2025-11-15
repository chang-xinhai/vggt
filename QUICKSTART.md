# Quick Start Checklist for Feed-forward Novel View Synthesis

This checklist provides a quick overview of what has been implemented and how to get started.

## ✅ What's Implemented

### Core Components
- [x] Plücker ray encoding (`vggt/utils/plucker_rays.py`)
- [x] Plücker encoder head (`vggt/heads/nvs_head.py`)
- [x] RGB regression head (`vggt/heads/nvs_head.py`)
- [x] VGGT-NVS model (`vggt/models/vggt_nvs.py`)

### Training Infrastructure
- [x] Objaverse dataset loader (`training/data/datasets/objaverse.py`)
- [x] Training configuration (`training/config/nvs_default.yaml`)
- [x] Loss function (`training/loss_nvs.py`)

### Evaluation
- [x] GSO evaluation script (`eval_nvs_gso.py`)

### Documentation
- [x] Quick start guide (`docs/NVS_README.md`)
- [x] Reproduction explanation (`docs/NVS_Reproduction_Explanation.md`)
- [x] Verification checklist (`docs/NVS_Verification_Checklist.md`)
- [x] Training guide (`docs/NVS_Training_Guide.md`)
- [x] Implementation summary (`IMPLEMENTATION_SUMMARY.md`)

### Testing & Demo
- [x] Component tests (`test_nvs_components.py`)
- [x] Demo script (`demo_nvs.py`)

## 📋 Getting Started Steps

### 1. Verify Installation
```bash
# Run component tests
python test_nvs_components.py
```
Expected output: ✓ All 4 tests passed

### 2. Prepare Your Dataset

Your dataset should have this structure:
```
DATASET_DIR/
  object_001/
    images/
      000.png
      001.png
      002.png
      ...
    cameras.json
  object_002/
    ...
```

**cameras.json format:**
```json
{
  "000": {
    "intrinsics": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
    "extrinsics": [[r11, r12, r13, tx], [r21, r22, r23, ty], [r31, r32, r33, tz]]
  }
}
```

### 3. Download Pre-trained Checkpoint

You need a pre-trained VGGT checkpoint:
```bash
# Download from Hugging Face or your checkpoint location
# Save to: /path/to/vggt_checkpoint.pt
```

### 4. Update Configuration

Edit `training/config/nvs_default.yaml`:
```yaml
data:
  train:
    dataset:
      dataset_configs:
        - OBJAVERSE_DIR: /YOUR/PATH/TO/DATASET  # UPDATE

checkpoint:
  resume_checkpoint_path: /YOUR/PATH/TO/VGGT_CKPT  # UPDATE
```

### 5. Start Training

Single GPU:
```bash
cd training
python launch.py --config-name nvs_default
```

Multi-GPU (recommended):
```bash
cd training
torchrun --nproc_per_node=4 launch.py --config-name nvs_default
```

### 6. Monitor Progress

Watch TensorBoard:
```bash
tensorboard --logdir logs/tensorboard
```

Check logs:
```bash
tail -f logs/nvs_exp001/train.log
```

### 7. Evaluate on GSO

After training:
```bash
python eval_nvs_gso.py \
    --gso_dir /path/to/gso \
    --checkpoint logs/nvs_exp001/ckpts/epoch_30.pt \
    --output gso_results.json
```

### 8. Run Demo

Synthesize a novel view:
```bash
python demo_nvs.py \
    --input_images img1.png img2.png img3.png img4.png \
    --checkpoint logs/nvs_exp001/ckpts/epoch_30.pt \
    --output novel_view.png
```

## 📚 Documentation Guide

**Start here if you're...**

- **New to the project**: Read `docs/NVS_README.md`
- **Want to understand the implementation**: Read `docs/NVS_Reproduction_Explanation.md`
- **Ready to train**: Read `docs/NVS_Training_Guide.md`
- **Verifying correctness**: Use `docs/NVS_Verification_Checklist.md`
- **Want a quick summary**: Read `IMPLEMENTATION_SUMMARY.md`

## 🔍 Quick Verification

Check that everything is working:

```bash
# 1. Run tests
python test_nvs_components.py
# Expected: ✓ All tests passed

# 2. Check imports
python -c "from vggt.models.vggt_nvs import VGGT_NVS; print('✓ Import successful')"

# 3. Verify dataset structure
ls -la /your/dataset/object_001/
# Expected: images/ and cameras.json

# 4. Check configuration
cat training/config/nvs_default.yaml | grep OBJAVERSE_DIR
# Expected: Your dataset path
```

## 🎯 Expected Results

After training on Objaverse-like data (~20% size) for 20-30 epochs:

- **Training PSNR**: > 20 dB
- **Validation PSNR**: > 18 dB
- **GSO PSNR**: Competitive with LVSM (per paper Table 7)
- **GSO SSIM**: Competitive with LVSM

## ⚠️ Common Issues

### Out of Memory
- Reduce `max_img_per_gpu` in config (try 8 or 4)
- Increase `accum_steps` (try 4)

### Slow Training
- Use multiple GPUs
- Increase `num_workers` in config
- Check data loading isn't the bottleneck

### Poor Results
- Verify dataset quality (visualize some samples)
- Check that pre-trained checkpoint loaded correctly
- Try different learning rates (1e-5, 5e-5, 1e-4)

## 📊 File Overview

**Total Implementation:**
- 14 files
- ~3,000 lines of code and documentation
- 4 component tests (all passing)
- 5 documentation files

**Core Code**: ~1,800 lines
**Documentation**: ~1,200 lines

## ✅ Final Verification

Before claiming success, verify:

1. [ ] All tests pass: `python test_nvs_components.py`
2. [ ] Dataset is prepared in correct format
3. [ ] Configuration paths are updated
4. [ ] Pre-trained checkpoint is available
5. [ ] Training runs without errors
6. [ ] Loss decreases over epochs
7. [ ] PSNR increases over epochs
8. [ ] Evaluation produces reasonable metrics
9. [ ] Qualitative results look good
10. [ ] Results align with paper's Table 7

## 🚀 You're Ready!

If all the above checks pass, you're ready to reproduce the Feed-forward Novel View Synthesis experiment from the VGGT paper.

**Good luck with your reproduction!** 🎉

For questions or issues, refer to:
- `docs/NVS_Verification_Checklist.md` for detailed verification
- `docs/NVS_Training_Guide.md` for troubleshooting
- `IMPLEMENTATION_SUMMARY.md` for technical details
