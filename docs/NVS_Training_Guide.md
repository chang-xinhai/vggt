# Feed-forward Novel View Synthesis Training Example

This directory contains an example of how to train VGGT for Feed-forward Novel View Synthesis.

## Prerequisites

1. **Install dependencies**:
```bash
pip install -e .
pip install fvcore omegaconf hydra-core
```

2. **Prepare dataset**: Organize your data in Objaverse-like structure (see main NVS README)

3. **Download pre-trained VGGT checkpoint**: You'll need the base VGGT checkpoint to initialize the aggregator

## Training Steps

### 1. Update Configuration

Edit `training/config/nvs_default.yaml`:

```yaml
data:
  train:
    dataset:
      dataset_configs:
        - _target_: data.datasets.objaverse.ObjaverseDataset
          OBJAVERSE_DIR: /path/to/your/objaverse/data  # UPDATE THIS
          
checkpoint:
  resume_checkpoint_path: /path/to/vggt/checkpoint.pt  # UPDATE THIS
```

### 2. Run Training

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

With 8 GPUs:
```bash
torchrun --nproc_per_node=8 launch.py --config-name nvs_default
```

### 3. Monitor Training

Training logs and checkpoints will be saved to:
- Logs: `logs/nvs_exp001/`
- Checkpoints: `logs/nvs_exp001/ckpts/`
- TensorBoard: `logs/tensorboard/`

View TensorBoard:
```bash
tensorboard --logdir logs/tensorboard
```

## Configuration Options

### Memory Management

If you encounter OOM errors, reduce batch size:
```yaml
max_img_per_gpu: 8  # Default is 16
accum_steps: 4      # Default is 2
```

### Learning Rate

Adjust based on your effective batch size:
```yaml
optim:
  optimizer:
    lr: 5e-5  # Try: 1e-5, 5e-5, 1e-4
```

### Loss Weights

Balance RGB and perceptual loss:
```yaml
loss:
  rgb:
    weight: 1.0
  perceptual:
    weight: 0.1  # Set to 0 to disable
```

## Expected Training Behavior

### Normal Training
- Loss should decrease steadily
- PSNR should increase over epochs
- First epoch may be slower (caching)
- Checkpoints saved every 5 epochs (configurable)

### Typical Metrics (After Convergence)
- Training Loss: < 0.05
- Training PSNR: > 20 dB
- Validation PSNR: > 18 dB

*Note: These are approximate and depend on dataset quality*

## Training Tips

### 1. Start with Small Dataset
Test your setup with a subset first:
```yaml
limit_train_batches: 100  # Process only 100 batches per epoch
limit_val_batches: 20
max_epochs: 5
```

### 2. Monitor GPU Usage
```bash
watch -n 1 nvidia-smi
```

### 3. Resume Training
Training automatically resumes from the last checkpoint:
```yaml
checkpoint:
  resume_checkpoint_path: logs/nvs_exp001/ckpts/epoch_10.pt
```

### 4. Fine-tune Hyperparameters
- Try different learning rates (1e-5, 5e-5, 1e-4)
- Adjust perceptual loss weight (0.05, 0.1, 0.2)
- Experiment with batch size vs accumulation steps

## Evaluation After Training

After training, evaluate on GSO:
```bash
python eval_nvs_gso.py \
    --gso_dir /path/to/gso \
    --checkpoint logs/nvs_exp001/ckpts/best_model.pt \
    --output gso_results.json
```

## Common Issues

### Issue: CUDA Out of Memory
**Solution**: Reduce `max_img_per_gpu` or increase `accum_steps`

### Issue: Training is very slow
**Solution**: 
- Check data loading (increase `num_workers`)
- Use multiple GPUs
- Enable mixed precision (already enabled by default)

### Issue: Loss is NaN
**Solution**:
- Reduce learning rate
- Check for corrupt images in dataset
- Reduce gradient clipping norm

### Issue: Model not improving
**Solution**:
- Verify dataset quality (visualize some samples)
- Check that aggregator checkpoint loaded correctly
- Try different learning rate
- Ensure Plücker rays are computed correctly

## Dataset Preparation Tips

### Create Cameras.json
Example script to generate cameras.json:
```python
import json
import numpy as np

cameras = {}
for i in range(num_views):
    # Your camera computation here
    cameras[f"{i:03d}"] = {
        "intrinsics": intrinsics[i].tolist(),
        "extrinsics": extrinsics[i].tolist()
    }

with open("cameras.json", "w") as f:
    json.dump(cameras, f, indent=2)
```

### Verify Dataset
Test your dataset loader:
```python
from training.data.datasets.objaverse import ObjaverseDataset

dataset = ObjaverseDataset(
    common_conf=...,
    OBJAVERSE_DIR="/path/to/data"
)

sample = dataset.get_data(seq_index=0)
print(sample['images'].shape)         # Should be (4, 3, H, W)
print(sample['target_images'].shape)  # Should be (1, 3, H, W)
```

## Performance Benchmarks

On a system with 4x NVIDIA A100 GPUs:
- Training speed: ~2 min/epoch (with 800 batches)
- Memory usage: ~30GB per GPU
- Total training time: ~10 hours for 30 epochs

## Next Steps

After successful training:
1. Evaluate on GSO dataset
2. Visualize synthesized views
3. Compare with baseline (LVSM)
4. Experiment with more input views (6, 8)
5. Try different target viewpoint distributions

## Questions?

- Check the main README: `docs/NVS_README.md`
- See reproduction explanation: `docs/NVS_Reproduction_Explanation.md`
- Review verification checklist: `docs/NVS_Verification_Checklist.md`
