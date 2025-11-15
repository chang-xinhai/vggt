# Feed-forward Novel View Synthesis - To-Do List for Verification

## Purpose
This document provides a comprehensive checklist for verifying that the Feed-forward Novel View Synthesis implementation is consistent with the original VGGT paper experiment. Use this list to ensure all components match the paper's description.

---

## 1. Architecture Verification

### 1.1 Input Processing
- [ ] **Verify**: Model accepts 4 input views (as stated in paper)
- [ ] **Check**: Input images are encoded using DINO (via existing VGGT aggregator)
- [ ] **Verify**: NO camera parameters are used for input frames
- [ ] **Check**: Input images are preprocessed to 518x518 resolution
- [ ] **Verify**: Input images are tokenized with patch_size=14

### 1.2 Plücker Ray Encoding
- [ ] **Verify**: Plücker rays are 6-dimensional (direction + moment)
- [ ] **Check**: Mathematical formulation follows standard Plücker coordinates
  - Direction vector: normalized ray direction in world space
  - Moment vector: cross product of origin and direction
- [ ] **Verify**: Plücker ray images have shape (6, H, W)
- [ ] **Check**: Convolutional encoder projects 6 channels to embed_dim
- [ ] **Verify**: Kernel size and stride match patch_size (14)
- [ ] **Check**: Output tokens are compatible with aggregator token format

### 1.3 Token Processing
- [ ] **Verify**: Input tokens (from DINO) and target tokens (from Plücker) are concatenated
- [ ] **Check**: Concatenation happens along the sequence/token dimension
- [ ] **Verify**: Combined tokens are processed through AA (Alternating Attention) transformer
- [ ] **Check**: AA transformer uses existing VGGT aggregator architecture
- [ ] **Verify**: No modifications to AA transformer itself (just different input)

### 1.4 RGB Regression Head
- [ ] **Verify**: Uses DPT (Dense Prediction Transformer) architecture
- [ ] **Check**: Extracts features from intermediate layers [4, 11, 17, 23]
- [ ] **Verify**: Multi-scale feature fusion with refinement blocks
- [ ] **Check**: Output has 3 channels (RGB)
- [ ] **Verify**: Output activation is sigmoid (range [0, 1])
- [ ] **Check**: Output resolution matches input (518x518)

---

## 2. Training Setup Verification

### 2.1 Dataset
- [ ] **Verify**: Dataset structure matches Objaverse-style organization
  - object_id/images/*.png
  - object_id/cameras.json
- [ ] **Check**: Dataset size is approximately 20% of Objaverse (or note if different)
- [ ] **Verify**: Train/test split is reasonable (e.g., 80/20)
- [ ] **Check**: Each sample contains 4 input views + target views
- [ ] **Verify**: Camera parameters (intrinsics/extrinsics) are in OpenCV format
- [ ] **Check**: Plücker rays are generated on-the-fly or pre-computed

### 2.2 Training Protocol
- [ ] **Verify**: Pre-trained VGGT aggregator is loaded
- [ ] **Check**: Aggregator weights are frozen during training
- [ ] **Verify**: Only Plücker encoder and RGB head are trained
- [ ] **Check**: Learning rate follows warmup + cosine decay schedule
- [ ] **Verify**: Batch size is adjusted for GPU memory
- [ ] **Check**: Gradient clipping is applied (max_norm=1.0)
- [ ] **Verify**: Mixed precision training (bfloat16 or float16) is used

### 2.3 Loss Function
- [ ] **Verify**: Primary loss is RGB reconstruction (L1 or L2)
- [ ] **Check**: Optional perceptual loss using VGG features
- [ ] **Verify**: Loss is computed only on target views, not input views
- [ ] **Check**: PSNR is logged for monitoring
- [ ] **Verify**: Loss weights are balanced (e.g., rgb=1.0, perceptual=0.1)

### 2.4 Training Configuration
- [ ] **Verify**: Configuration file exists (nvs_default.yaml)
- [ ] **Check**: All paths are configurable (OBJAVERSE_DIR, checkpoint path)
- [ ] **Verify**: Number of input/target views is configurable
- [ ] **Check**: Training can run on multiple GPUs with DDP
- [ ] **Verify**: Checkpoints are saved periodically
- [ ] **Check**: TensorBoard logging is enabled

---

## 3. Evaluation Verification

### 3.1 GSO Dataset Evaluation
- [ ] **Verify**: Evaluation script exists (eval_nvs_gso.py)
- [ ] **Check**: GSO dataset structure is supported
- [ ] **Verify**: Evaluation uses 4 input views (consistent with training)
- [ ] **Check**: PSNR metric is computed correctly
- [ ] **Verify**: SSIM metric is computed correctly
- [ ] **Check**: Results are averaged over all objects
- [ ] **Verify**: Standard deviation is reported alongside mean

### 3.2 Evaluation Protocol
- [ ] **Verify**: Model is in eval mode (no dropout, batch norm in eval mode)
- [ ] **Check**: No gradient computation during evaluation
- [ ] **Verify**: Deterministic sampling for reproducibility (fixed seed)
- [ ] **Check**: Results are saved to JSON file
- [ ] **Verify**: Comparison with LVSM baseline is possible

---

## 4. Implementation Details Verification

### 4.1 Plücker Ray Generation
- [ ] **Verify**: Pixel coordinates are correctly converted to camera space
- [ ] **Check**: Camera-to-world transformation is applied correctly
- [ ] **Verify**: Ray directions are normalized
- [ ] **Check**: Moment vectors use correct cross product
- [ ] **Verify**: Handles both single and batch inputs
- [ ] **Check**: Compatible with PyTorch tensors

### 4.2 Model Forward Pass
- [ ] **Verify**: Input shape handling is correct (with/without batch dim)
- [ ] **Check**: Token concatenation preserves spatial structure
- [ ] **Verify**: Patch indices are updated correctly for target tokens
- [ ] **Check**: RGB head receives correct tokens
- [ ] **Verify**: Output shape matches expected (B, S_target, H, W, 3)

### 4.3 Code Quality
- [ ] **Verify**: All files have proper copyright headers
- [ ] **Check**: Docstrings explain purpose and arguments
- [ ] **Verify**: Type hints are used where appropriate
- [ ] **Check**: Code follows existing VGGT style conventions
- [ ] **Verify**: No hardcoded paths or magic numbers
- [ ] **Check**: Error handling for missing files/data

---

## 5. Consistency with Paper

### 5.1 Key Claims
- [ ] **Verify**: "4 input views" - confirmed in code
- [ ] **Check**: "Plücker rays to represent target viewpoints" - implemented
- [ ] **Verify**: "Do not assume known camera parameters for input frames" - confirmed
- [ ] **Check**: "Concatenate tokens and process by AA transformer" - implemented
- [ ] **Verify**: "DPT head to regress RGB colors" - implemented

### 5.2 Design Choices
- [ ] **Verify**: Following LVSM approach (as stated in paper)
- [ ] **Check**: Using existing VGGT aggregator (not creating new one)
- [ ] **Verify**: Transfer learning from pre-trained VGGT
- [ ] **Check**: Training on Objaverse-like dataset
- [ ] **Verify**: Evaluation on GSO dataset

### 5.3 Expected Results
- [ ] **Verify**: Results should be "competitive" with LVSM (as per paper)
- [ ] **Check**: Acknowledge smaller dataset (~20% of Objaverse)
- [ ] **Verify**: Note that larger dataset would improve results (per paper)

---

## 6. Testing and Validation

### 6.1 Unit Tests
- [ ] **Verify**: Plücker ray generation produces correct shape
- [ ] **Check**: Plücker rays satisfy mathematical properties
  - Ray directions are normalized (norm ≈ 1)
  - Moment vectors are orthogonal to directions (dot product ≈ 0)
- [ ] **Verify**: PluckerEncoder output shape is correct
- [ ] **Check**: RGBRegressionHead output is in [0, 1] range
- [ ] **Verify**: VGGT_NVS forward pass completes without errors

### 6.2 Integration Tests
- [ ] **Verify**: Model can load pre-trained VGGT checkpoint
- [ ] **Check**: Training loop runs for multiple iterations
- [ ] **Verify**: Loss decreases over training
- [ ] **Check**: Evaluation script runs on sample data
- [ ] **Verify**: Generated images are visually reasonable

### 6.3 Sanity Checks
- [ ] **Verify**: Model outputs have expected shape
- [ ] **Check**: No NaN or Inf values in outputs/gradients
- [ ] **Verify**: Memory usage is reasonable
- [ ] **Check**: Training speed is reasonable (not too slow)
- [ ] **Verify**: Checkpoints can be loaded and resumed

---

## 7. Documentation Verification

### 7.1 Code Documentation
- [ ] **Verify**: Reproduction Explanation document exists
- [ ] **Check**: All components are explained in detail
- [ ] **Verify**: Mathematical formulations are included
- [ ] **Check**: Design choices are justified
- [ ] **Verify**: Known differences from paper are noted

### 7.2 Usage Documentation
- [ ] **Verify**: Training instructions are clear
- [ ] **Check**: Evaluation instructions are clear
- [ ] **Verify**: Example commands are provided
- [ ] **Check**: Expected outputs are described
- [ ] **Verify**: Troubleshooting tips are included

### 7.3 Configuration Documentation
- [ ] **Verify**: All config options are documented
- [ ] **Check**: Default values are reasonable
- [ ] **Verify**: Required vs optional settings are clear
- [ ] **Check**: Path configurations are explained

---

## 8. Reproducibility Checklist

### 8.1 Dependencies
- [ ] **Verify**: All required packages are listed
- [ ] **Check**: Version requirements are specified where critical
- [ ] **Verify**: Installation instructions are provided
- [ ] **Check**: GPU requirements are documented

### 8.2 Random Seeds
- [ ] **Verify**: Random seeds are set for reproducibility
- [ ] **Check**: Seeds are used in data sampling
- [ ] **Verify**: Seeds are used in initialization
- [ ] **Check**: Seeds are documented in config

### 8.3 Checkpoints
- [ ] **Verify**: Pre-trained checkpoint path is specified
- [ ] **Check**: Checkpoint loading is robust (handles missing keys)
- [ ] **Verify**: Trained checkpoints can be saved
- [ ] **Check**: Saved checkpoints can be loaded for inference

---

## 9. Paper Alignment Checklist

### 9.1 Section References
- [ ] **Verify**: Implementation matches "Feed-forward Novel View Synthesis" section
- [ ] **Check**: References to LVSM are consistent
- [ ] **Verify**: Table 7 metrics can be reproduced
- [ ] **Check**: Figure 6 qualitative examples can be generated

### 9.2 Cited Methods
- [ ] **Verify**: LVSM training protocol is followed
- [ ] **Check**: Plücker ray representation matches referenced works
- [ ] **Verify**: DPT architecture follows original paper
- [ ] **Check**: GSO evaluation follows standard protocol

### 9.3 Claims and Limitations
- [ ] **Verify**: All claims in paper are supported by code
- [ ] **Check**: Limitations are acknowledged (smaller dataset)
- [ ] **Verify**: Future improvements are noted (larger dataset)
- [ ] **Check**: Competitive results claim is testable

---

## 10. Final Verification Steps

### Before Claiming Reproduction Complete:
1. [ ] **Run full training for at least 10 epochs**
2. [ ] **Evaluate on GSO dataset and record metrics**
3. [ ] **Compare with paper's reported results**
4. [ ] **Generate qualitative examples (similar to Figure 6)**
5. [ ] **Verify all checkboxes above are completed**
6. [ ] **Document any discrepancies or issues**
7. [ ] **Prepare final reproduction report**

### Success Criteria:
- [ ] Model trains without errors
- [ ] Loss decreases consistently
- [ ] PSNR/SSIM metrics are reasonable (within expected range)
- [ ] Qualitative results show coherent novel views
- [ ] Results are "competitive" with LVSM (as paper states)

---

## Notes Section

Use this space to document any issues, discrepancies, or important observations:

```
Date: ___________
Observer: ___________

Issues Found:
-

Resolutions:
-

Additional Notes:
-
```

---

## Conclusion

This checklist ensures comprehensive verification of the Feed-forward Novel View Synthesis implementation. Each item should be checked carefully and any discrepancies documented. The goal is to reproduce the experiment as faithfully as possible while noting any necessary deviations or limitations.
